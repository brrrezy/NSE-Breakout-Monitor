"""
Core Engine — Main orchestrator for the NSE Breakout Detection Engine.

Coordinates all modules:
  1. Load config & state
  2. Check market context (regime, VIX, breadth)
  3. Load universe + watchlist
  4. Download data with caching
  5. Compute all indicators
  6. Run analysis modules (structure, liquidity, patterns, base, breakout)
  7. Fetch news & sentiment
  8. Score all candidates
  9. Rank and filter
  10. Apply risk management
  11. Format output
  12. Send alerts
  13. Track predictions
  14. Save state
"""

import sys
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from config.settings import Settings
from core.data_provider import DataProvider
from core.universe import Universe, StateManager

# Indicators
from indicators.trend import add_all_trend_indicators
from indicators.momentum import add_all_momentum_indicators
from indicators.volume import add_all_volume_indicators
from indicators.volatility import add_all_volatility_indicators

# Analysis
from analysis.base_detection import find_base, detect_uptrend
from analysis.breakout import (is_breakout_candle, confluence_score,
                               calc_risk, calculate_entry_zone)
from analysis.market_structure import analyze_market_structure
from analysis.liquidity import analyze_liquidity
from analysis.patterns import detect_all_patterns
from analysis.strength import analyze_strength
from analysis.market_context import analyze_market_context

# Scoring
from scoring.scorer import compute_composite_score
from scoring.ranker import (rank_candidates, filter_by_threshold,
                            separate_actionable_watchlist)
from scoring.risk_manager import apply_risk_filters

# Alerts
from alerts.telegram import send_telegram
from alerts.formatter import (format_breakout_telegram,
                              format_watchlist_telegram,
                              format_market_context_telegram,
                              format_detailed)

# News
from news.sentiment import analyze_sentiment_batch, get_groq_verdict

# Learning
from learning.tracker import PerformanceTracker
from learning.optimizer import WeightOptimizer


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicator modules to a DataFrame."""
    df = df.copy()
    df = add_all_trend_indicators(df)
    df = add_all_momentum_indicators(df)
    df = add_all_volume_indicators(df)
    df = add_all_volatility_indicators(df)
    return df


def analyze_single_ticker(sym: str,
                           df: pd.DataFrame,
                           nifty_df: Optional[pd.DataFrame],
                           market_context: Dict,
                           news_data: Dict,
                           sector: str = "",
                           ) -> Optional[Dict[str, Any]]:
    """
    Full analysis pipeline for one stock.
    Returns a candidate dict or None if rejected.
    """
    cfg = Settings.get()

    try:
        if len(df) < cfg.min_data_days:
            return None

        # Compute indicators
        df = compute_all_indicators(df)
        latest = df.iloc[-1]
        price = float(latest["Close"])

        # Price filter
        if price < cfg.min_price or price > cfg.max_price:
            return None

        # Phase 1: Must be in uptrend
        if not detect_uptrend(df):
            return None

        # Phase 2: Find a base
        has_base, base_info = find_base(df)
        if not has_base:
            return None

        pivot = base_info["pivot"]
        end_offset = base_info["base_end_offset"]

        # Determine status (ACTIONABLE vs WATCHLIST)
        if end_offset == 1 and is_breakout_candle(df, pivot):
            conf = confluence_score(latest)
            status = "ACTIONABLE" if conf >= cfg.confluence_min else "WATCHLIST"
        elif end_offset == 0 and price <= pivot:
            status = "WATCHLIST"
        else:
            return None

        # Risk calculation
        bo_low = float(latest["Low"]) if status == "ACTIONABLE" else 0
        atr = float(latest.get("atr", 0)) if not pd.isna(
            latest.get("atr", 0)) else 0
        risk_data = calc_risk(price, bo_low, atr)
        entry_zone = calculate_entry_zone(pivot, atr)

        # Relative strength analysis
        strength_data = analyze_strength(df, nifty_df, sector)

        # Market structure analysis
        structure_data = {}
        if cfg.enable_liquidity_analysis:
            try:
                structure_data = analyze_market_structure(df)
            except Exception:
                pass

        # Liquidity analysis
        liquidity_data = {}
        if cfg.enable_liquidity_analysis:
            try:
                liquidity_data = analyze_liquidity(df, pivot)
            except Exception:
                pass

        # Pattern detection
        patterns = []
        if cfg.enable_pattern_detection:
            try:
                patterns = detect_all_patterns(df)
            except Exception:
                pass

        # News sentiment (already fetched in batch)
        ticker_news = news_data.get(
            sym.replace(".NS", ""), {})

        # Compute composite score
        conf = confluence_score(latest)
        scores = compute_composite_score(
            df=df,
            latest=latest,
            base_info=base_info,
            patterns=patterns,
            strength_data=strength_data,
            news_data=ticker_news,
            risk_data=risk_data,
            market_context=market_context,
            liquidity_data=liquidity_data,
        )

        # Support level (lowest low in base)
        base_start = len(df) - 1 - end_offset - base_info["base_len"] + 1
        support = float(df["Low"].iloc[
            max(0, base_start):len(df) - end_offset].min())

        # Build candidate
        candidate = {
            "symbol": sym,
            "status": status,
            "price": price,
            "pivot": round(pivot, 2),
            "support": round(support, 2),
            "entry_low": entry_zone["entry_low"],
            "entry_high": entry_zone["entry_high"],
            "scores": scores,
            "risk": risk_data,
            "base_info": base_info,
            "strength": strength_data,
            "structure": structure_data,
            "liquidity": liquidity_data,
            "patterns": patterns,
            "news": ticker_news,
            "market_context": market_context,
            "confluence": conf,
            # Quick-access fields
            "rsi": float(latest.get("rsi", 0)) if not pd.isna(
                latest.get("rsi", 0)) else 0,
            "adx": float(latest.get("adx", 0)) if not pd.isna(
                latest.get("adx", 0)) else 0,
            "rvol": float(latest.get("rvol", 0)) if not pd.isna(
                latest.get("rvol", 0)) else 0,
            "macd_hist": float(latest.get("macd_hist", 0)) if not pd.isna(
                latest.get("macd_hist", 0)) else 0,
        }

        return candidate

    except Exception as e:
        print(f"  [WARN] {sym}: {e}", file=sys.stderr)
        return None


def run_scan():
    """
    Main scan execution — the complete pipeline.
    """
    cfg = Settings.get()
    now = cfg.get_now_ist()
    today = now.strftime("%Y-%m-%d")

    print(f"{'=' * 50}")
    print(f"NSE BREAKOUT DETECTION ENGINE")
    print(f"{'=' * 50}")
    print(f"IST: {now.strftime('%Y-%m-%d %H:%M')}")

    # Weekend check
    if now.weekday() >= 5:
        print("Weekend. Exiting.")
        return

    # Before market
    if now.hour * 100 + now.minute < 910:
        print("Before market hours. Exiting.")
        return

    # Load state
    state_mgr = StateManager()
    if state_mgr.eod_date == today:
        print(f"Already completed EOD for {today}. Exiting.")
        return

    state_mgr.reset_day(today)
    alerted = state_mgr.alerted_today
    is_eod = now.hour > 15 or (now.hour == 15 and now.minute >= 28)

    print(f"Mode: {'EOD' if is_eod else 'INTRADAY'} | "
          f"Watchlist: {len(state_mgr.watchlist)}")

    # Initialize providers
    data_provider = DataProvider()

    # Step 1: Market context
    print("\n📊 Analyzing market context...")
    market_context = analyze_market_context(data_provider)
    print(f"   Regime: {market_context['regime']} | "
          f"VIX: {market_context.get('vix_value', 0):.1f} | "
          f"Score: {market_context.get('market_score', 50):.0f}/100")

    if not market_context.get("regime_ok", True):
        send_telegram(
            f"*Market Regime: {market_context['regime'].upper()}*\n"
            f"VIX: {market_context.get('vix_value', 0):.1f}\n"
            f"No long setups in bear market.")

    # Step 2: Build universe
    print("\n📋 Loading universe...")
    universe = Universe()
    nifty500 = universe.get_nifty500()
    scan_symbols = list(set(nifty500 + state_mgr.watchlist))
    print(f"   Universe: {len(scan_symbols)} symbols")

    # Step 3: Download data
    print("\n📥 Downloading data...")
    stock_data = data_provider.get_bulk_data(scan_symbols)
    nifty_df = data_provider.get_nifty50()
    print(f"   Downloaded: {len(stock_data)} stocks")

    # Step 4: News sentiment (batch, parallel-safe)
    print("\n📰 Fetching news...")
    if cfg.enable_news_engine:
        # Only fetch news for a subset to stay within rate limits
        news_tickers = list(stock_data.keys())[:50]
        news_data = analyze_sentiment_batch(news_tickers)
        print(f"   News analyzed for {len(news_data)} tickers")
    else:
        news_data = {}

    # Step 5: Analyze all stocks
    print(f"\n🔍 Analyzing {len(stock_data)} stocks...")
    candidates: List[Dict] = []

    for sym in tqdm(stock_data, desc="  Scanning"):
        df = stock_data[sym]
        sector = universe.get_sector(sym)
        result = analyze_single_ticker(
            sym, df, nifty_df, market_context, news_data, sector)
        if result is not None:
            candidates.append(result)

    print(f"   Found {len(candidates)} raw candidates")

    # Step 6: Risk filtering
    print("\n🛡️ Applying risk filters...")
    filtered = []
    for c in candidates:
        df = stock_data.get(c["symbol"])
        if df is not None:
            passes, reasons = apply_risk_filters(c, df, market_context)
            c["rejection_reasons"] = reasons
            if passes:
                filtered.append(c)
            else:
                print(f"   ❌ {c['symbol']}: {'; '.join(reasons[:2])}")

    # Step 7: Score threshold filter
    scored = filter_by_threshold(filtered)
    print(f"   After threshold: {len(scored)} candidates")

    # Step 8: Rank
    ranked = rank_candidates(scored)

    # Step 9: Separate ACTIONABLE vs WATCHLIST
    actionable, watchlist = separate_actionable_watchlist(ranked)

    print(f"\n{'=' * 50}")
    print(f"Results: {len(actionable)} ACTIONABLE, "
          f"{len(watchlist)} WATCHLIST")
    print(f"{'=' * 50}")

    # Step 10: Send alerts
    # Market context (always send)
    if is_eod or not alerted:
        send_telegram(format_market_context_telegram(market_context))

    # Breakout alerts
    new_breakouts = 0
    for c in actionable:
        if c["symbol"] not in alerted:
            send_telegram(format_breakout_telegram(c))
            state_mgr.add_alerted(c["symbol"])
            new_breakouts += 1
            print(f"  🚀 {c['symbol']} BREAKOUT ₹{c['price']:.0f} "
                  f"[Score: {c['scores']['composite']:.0f}]")
        else:
            sym = c["symbol"].replace(".NS", "")
            print(f"  ⏩ {sym} already alerted, skipping.")

    # Watchlist (EOD only)
    if is_eod and watchlist:
        wl_msg = format_watchlist_telegram(watchlist, is_eod=True)
        verdict = get_groq_verdict(watchlist, is_eod=True)
        send_telegram(wl_msg + verdict)
        print(f"  📋 EOD watchlist sent ({len(watchlist)} stocks)")

    # Update watchlist
    state_mgr.watchlist = [c["symbol"] for c in watchlist]

    # No results message (EOD only)
    if not actionable and not watchlist and is_eod:
        send_telegram("*EOD Scan Complete* — No clean setups today.")

    # Step 11: Track predictions
    if cfg.enable_learning:
        try:
            tracker = PerformanceTracker()

            # Record new alerts
            for c in actionable + watchlist:
                tracker.record_alert(c)

            # Check outcomes for old alerts
            def _price_lookup(ticker):
                df = stock_data.get(ticker)
                if df is not None and not df.empty:
                    return float(df["Close"].iloc[-1])
                return None

            resolved = tracker.check_outcomes(_price_lookup)
            if resolved:
                print(f"  📊 Resolved {resolved} past alerts")

            # Weekly performance report (only on Fridays at EOD)
            if is_eod and now.weekday() == 4:
                report = tracker.get_performance_report()
                send_telegram(report)

                # Try weight optimization
                optimizer = WeightOptimizer(tracker)
                if optimizer.should_optimize():
                    new_weights = optimizer.optimize()
                    if new_weights:
                        print(f"  🧠 Weights optimized: {new_weights}")

            tracker.close()
        except Exception as e:
            print(f"  [LEARN] Error: {e}", file=sys.stderr)

    # Step 12: Save state
    if is_eod:
        state_mgr.eod_date = today

    state_mgr.save()
    print(f"\nSaved: {len(state_mgr.watchlist)} watchlist, "
          f"{len(state_mgr.alerted_today)} alerted")

    # Print detailed results for top candidates
    if ranked:
        print(f"\n{'─' * 50}")
        print("TOP CANDIDATES:")
        print(f"{'─' * 50}")
        for c in ranked[:5]:
            sym = c["symbol"].replace(".NS", "")
            scores = c["scores"]
            print(f"  #{c['rank']} {sym:10s} "
                  f"[{c['status']:10s}] "
                  f"₹{c['price']:>7.0f} "
                  f"Score: {scores['composite']:>5.1f} "
                  f"Prob: {scores['breakout_probability']:>5.1f}% "
                  f"Conf: {scores['confidence']:>5.1f}")
