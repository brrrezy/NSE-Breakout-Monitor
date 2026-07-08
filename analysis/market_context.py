"""
Market Context Analysis.

Evaluates:
  - Nifty 50 trend & regime (uptrend / selective / bear)
  - India VIX level (complacency / elevated / extreme)
  - Market breadth (advance/decline proxy)
  - Nifty Bank trend
  - Overall market score
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import ta

from config.settings import Settings
from core.data_provider import DataProvider


# ============================================================
# MARKET REGIME
# ============================================================

def check_market_regime(nifty_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Determine market regime from Nifty 50 data.

    Returns:
        regime: "uptrend", "selective", "bear", "unknown"
        regime_ok: True if longs are safe
        details: dict with EMAs, slopes, etc.
    """
    if nifty_df is None or len(nifty_df) < 60:
        return {"regime": "unknown", "regime_ok": True, "details": {}}

    c = nifty_df["Close"]
    e21 = ta.trend.ema_indicator(c, 21)
    e55 = ta.trend.ema_indicator(c, 55)
    e200 = ta.trend.ema_indicator(c, 200)

    latest = c.iloc[-1]
    ema21_now, ema55_now = e21.iloc[-1], e55.iloc[-1]
    ema21_5d, ema55_5d = e21.iloc[-5], e55.iloc[-5]
    ema200_now = e200.iloc[-1] if len(e200) >= 200 else ema55_now

    both_up = ema21_now > ema21_5d and ema55_now > ema55_5d

    if latest > ema21_now and latest > ema55_now and both_up:
        regime = "uptrend"
        regime_ok = True
    elif latest < ema21_now and latest < ema55_now:
        regime = "bear"
        regime_ok = False
    else:
        regime = "selective"
        regime_ok = True

    # Nifty performance
    perf_20d = float(c.iloc[-1] / c.iloc[-20] - 1) if len(c) >= 20 else 0
    perf_50d = float(c.iloc[-1] / c.iloc[-50] - 1) if len(c) >= 50 else 0

    return {
        "regime": regime,
        "regime_ok": regime_ok,
        "nifty_close": float(latest),
        "nifty_ema21": float(ema21_now),
        "nifty_ema55": float(ema55_now),
        "nifty_ema200": float(ema200_now),
        "nifty_above_ema21": bool(latest > ema21_now),
        "nifty_above_ema55": bool(latest > ema55_now),
        "nifty_above_ema200": bool(latest > ema200_now),
        "nifty_perf_20d": round(perf_20d, 4),
        "nifty_perf_50d": round(perf_50d, 4),
        "emas_rising": both_up,
    }


# ============================================================
# VIX ANALYSIS
# ============================================================

def analyze_vix(vix_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Analyze India VIX for fear/greed levels.

    Low VIX (< 13): Complacency — squeeze setups work well
    Normal (13-20): Neutral
    High (20-30): Elevated — be selective, wider stops
    Extreme (> 30): Fear — avoid new longs, wait for VIX to drop
    """
    cfg = Settings.get()

    if vix_df is None or vix_df.empty:
        return {
            "vix_level": "unknown",
            "vix_value": 0,
            "vix_regime": "neutral",
            "favorable_for_longs": True,
        }

    vix = float(vix_df["Close"].iloc[-1])
    vix_5d_ago = float(vix_df["Close"].iloc[-5]) if len(vix_df) >= 5 else vix
    vix_trend = "falling" if vix < vix_5d_ago * 0.95 else \
        ("rising" if vix > vix_5d_ago * 1.05 else "flat")

    # Percentile of current VIX vs 1-year range
    if len(vix_df) >= 50:
        vix_pctile = float(
            (vix_df["Close"].iloc[-252:] < vix).sum() /
            min(len(vix_df), 252) * 100
        )
    else:
        vix_pctile = 50.0

    if vix < 13:
        level = "low"
        regime = "complacent"
    elif vix < cfg.vix_high_threshold:
        level = "normal"
        regime = "neutral"
    elif vix < cfg.vix_extreme_threshold:
        level = "high"
        regime = "elevated"
    else:
        level = "extreme"
        regime = "fear"

    favorable = vix < cfg.vix_extreme_threshold

    return {
        "vix_value": round(vix, 2),
        "vix_level": level,
        "vix_regime": regime,
        "vix_trend": vix_trend,
        "vix_percentile": round(vix_pctile, 1),
        "favorable_for_longs": favorable,
    }


# ============================================================
# MARKET BREADTH (PROXY)
# ============================================================

def estimate_breadth(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Estimate market breadth from the stock universe data.
    Since we don't have advance/decline data directly,
    we proxy it from the downloaded stock data.

    Metrics:
    - % stocks above 20 EMA
    - % stocks above 50 EMA
    - % stocks with positive 20-day returns
    """
    if not stock_data:
        return {
            "above_ema20_pct": 50.0,
            "above_ema50_pct": 50.0,
            "positive_20d_pct": 50.0,
            "breadth_score": 50.0,
        }

    above_20 = 0
    above_50 = 0
    positive_20d = 0
    total = 0

    for sym, df in stock_data.items():
        if len(df) < 50:
            continue
        total += 1
        c = df["Close"].iloc[-1]

        ema20 = ta.trend.ema_indicator(df["Close"], 20).iloc[-1]
        ema50 = ta.trend.ema_indicator(df["Close"], 50).iloc[-1]

        if c > ema20:
            above_20 += 1
        if c > ema50:
            above_50 += 1
        if c > df["Close"].iloc[-20]:
            positive_20d += 1

    if total == 0:
        return {
            "above_ema20_pct": 50.0,
            "above_ema50_pct": 50.0,
            "positive_20d_pct": 50.0,
            "breadth_score": 50.0,
        }

    a20 = round(above_20 / total * 100, 1)
    a50 = round(above_50 / total * 100, 1)
    p20 = round(positive_20d / total * 100, 1)
    breadth = round((a20 * 0.3 + a50 * 0.4 + p20 * 0.3), 1)

    return {
        "above_ema20_pct": a20,
        "above_ema50_pct": a50,
        "positive_20d_pct": p20,
        "breadth_score": breadth,
        "stocks_counted": total,
    }


# ============================================================
# FULL MARKET CONTEXT
# ============================================================

def analyze_market_context(data_provider: DataProvider,
                           stock_data: Optional[Dict] = None
                           ) -> Dict[str, Any]:
    """
    Complete market context analysis.
    """
    nifty_df = data_provider.get_nifty50()
    vix_df = data_provider.get_india_vix()
    bank_df = data_provider.get_nifty_bank()

    regime = check_market_regime(nifty_df)
    vix = analyze_vix(vix_df)

    # Bank Nifty trend (quick check)
    bank_trend = "unknown"
    if bank_df is not None and len(bank_df) >= 21:
        bank_close = bank_df["Close"].iloc[-1]
        bank_ema21 = ta.trend.ema_indicator(bank_df["Close"], 21).iloc[-1]
        bank_trend = "bullish" if bank_close > bank_ema21 else "bearish"

    # Breadth (optional — only if stock data provided)
    breadth = estimate_breadth(stock_data) if stock_data else {
        "breadth_score": 50.0
    }

    # Overall market score (0-100)
    market_score = 50.0
    if regime["regime"] == "uptrend":
        market_score += 20
    elif regime["regime"] == "bear":
        market_score -= 30
    elif regime["regime"] == "selective":
        market_score += 5

    if vix["favorable_for_longs"]:
        market_score += 10
    else:
        market_score -= 20

    if vix.get("vix_trend") == "falling":
        market_score += 5
    elif vix.get("vix_trend") == "rising":
        market_score -= 5

    if bank_trend == "bullish":
        market_score += 5

    breadth_sc = breadth.get("breadth_score", 50)
    market_score += (breadth_sc - 50) * 0.2

    market_score = max(0, min(100, market_score))

    return {
        **regime,
        **vix,
        "bank_nifty_trend": bank_trend,
        "breadth": breadth,
        "market_score": round(market_score, 1),
    }
