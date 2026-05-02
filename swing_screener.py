import argparse
import datetime as dt
import json
import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm

# ============================================================
# 1) CONFIG & CACHE PATHS
# ============================================================

_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "nse_screener_cache"
CACHE_DIR = Path(os.environ.get("NSE_SCREENER_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"
DB_PATH = Path("nse_screener_cache.db")

# ============================================================
# TELEGRAM CONFIG (Set these via environment variables or replace here)
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Track alerted symbols to avoid spam
alerted_today = set()

# ============================================================
# 2) DATABASE & CACHE ENGINE
# ============================================================


class FundamentalCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at TIMESTAMP
                )
            """)

    def get(self, symbol: str, ttl_days: int = 7) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, updated_at FROM fundamentals WHERE symbol = ?",
                    (symbol,),
                )
                row = cursor.fetchone()
                if row:
                    data_json, updated_at = row
                    updated_at = dt.datetime.fromisoformat(updated_at)
                    if (dt.datetime.now() - updated_at).days < ttl_days:
                        return json.loads(data_json)
        except Exception:
            pass
        return None

    def set(self, symbol: str, data: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO fundamentals (symbol, data, updated_at) VALUES (?, ?, ?)",
                    (symbol, json.dumps(data), dt.datetime.now().isoformat()),
                )
        except Exception:
            pass


db_cache = FundamentalCache(DB_PATH)

# ============================================================
# 3) DATA FETCHING
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    if NSE_EQUITY_LIST_CACHE.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(
            NSE_EQUITY_LIST_CACHE.stat().st_mtime
        )
        if age.total_seconds() < cache_ttl_hours * 3600:
            text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            return [
                f"{l.split(',')[0].strip().strip('"')}.NS"
                for l in lines[1:]
                if l.strip()
            ]

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")
        lines = resp.text.splitlines()
        return [
            f"{l.split(',')[0].strip().strip('"')}.NS" for l in lines[1:] if l.strip()
        ]
    except Exception:
        return []


def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Alert Failed: {e}")


def is_market_open() -> bool:
    # NSE Hours: 09:15 to 15:30 IST
    now = dt.datetime.now()
    # Check if it's a weekday
    if now.weekday() >= 5:
        return False
    # Check time (IST is approx UTC+5.5)
    # This check assumes the system time is in IST. 
    # If system is UTC, adjust accordingly.
    current_time = now.time()
    start_time = dt.time(9, 15)
    end_time = dt.time(15, 30)
    return start_time <= current_time <= end_time


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    return df


# ============================================================
# 4) TECHNICAL ENGINE
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMAs
    for p in [8, 21, 50, 150, 200]:
        df[f"ema{p}"] = ta.trend.ema_indicator(df["Close"], p)

    # Momentum
    df["rsi"] = ta.momentum.rsi(df["Close"], 14)
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    # Volume
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ma50"] = df["Volume"].rolling(50).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]
    df["rvol50"] = df["Volume"] / df["vol_ma50"]

    # Volatility
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    # ADR% over 20 days
    df["adrp20"] = ((df["High"] - df["Low"]).rolling(20).mean() / df["Close"]) * 100.0
    # Current Day Range %
    df["drp"] = ((df["High"] - df["Low"]) / df["Close"]) * 100.0

    # CLV / Delta
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["delta_proxy"] = (
        ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    ) * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, bool]]:
    close, ema50, ema200 = (
        latest.get("Close"),
        latest.get("ema50"),
        latest.get("ema200"),
    )
    trend_ok = (
        bool(close > ema50 > ema200) if not pd.isna(ema200) else bool(close > ema50)
    )

    signals = {
        "MACD": bool(latest.get("macd") > latest.get("macd_signal")),
        "RSI": bool(latest.get("rsi") >= 55),
        "STOCH": bool(latest.get("stoch") >= 60),
        "VOLUME": bool(latest.get("vol_mult") >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0) > 0),
        "TREND": trend_ok,
        "ADX": bool(latest.get("adx") >= 20),
        "ATR": bool(latest.get("atr") > 0),
    }
    return int(sum(signals.values())), signals


def detect_minervini_trend(df: pd.DataFrame) -> bool:
    if len(df) < 200:
        return False
    c, e50, e150, e200 = (
        df["Close"].iloc[-1],
        df["ema50"].iloc[-1],
        df["ema150"].iloc[-1],
        df["ema200"].iloc[-1],
    )
    h52, l52 = df["High"].tail(252).max(), df["Low"].tail(252).min()
    e200_1m = df["ema200"].iloc[-20] if len(df) >= 20 else e200

    conds = [
        c > e150 > e200,
        e200 > e200_1m,
        e50 > e150,
        e50 > e200,
        c > e50,
        c >= l52 * 1.3,
        c >= h52 * 0.75,
    ]
    return all(conds)


def detect_vcp_setup(df: pd.DataFrame, lookback: int = 60) -> Tuple[bool, Dict[str, Any]]:
    """
    Detects VCP Setup (Watchlist phase):
    - Price tightening (volatility contraction)
    - Volume drying up
    - Holding near recent highs
    """
    if len(df) < lookback:
        return False, {}

    w = df.tail(lookback)
    latest = w.iloc[-1]

    # 1. Near Highs Filter (Already in trend, but double check)
    h52 = w["High"].max()
    near_highs = latest["Close"] >= h52 * 0.90  # Within 10% of consolidation high

    # 2. Tightness (Calmness)
    # Current day range vs average range
    is_tight = latest["drp"] < latest["adrp20"] * 0.8
    # Standard deviation of last 5 days close prices
    tightness_5d = w["Close"].tail(5).std() / latest["Close"] < 0.015

    # 3. Volume Drying
    # Last 3 days avg volume vs 50-day average
    vol_dry = w["Volume"].tail(3).mean() < latest["vol_ma50"] * 0.8

    # 4. Volatility Contraction (Behavioral)
    # Divide window into 3 parts and check if ADR is decreasing
    r1 = w["drp"].iloc[:20].mean()
    r2 = w["drp"].iloc[20:40].mean()
    r3 = w["drp"].iloc[40:60].mean()
    contraction = (r1 > r2 >= r3) or (r1 > r3)

    is_setup = near_highs and (is_tight or tightness_5d) and vol_dry and contraction

    details = {
        "near_highs": near_highs,
        "is_tight": is_tight,
        "vol_dry": vol_dry,
        "contraction": contraction,
        "pivot_high": float(h52),
    }
    return bool(is_setup), details


def detect_breakout_trigger(df: pd.DataFrame, pivot_high: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Detects Breakout Trigger (Entry phase):
    - Price breaking above pivot/resistance
    - Volume expansion
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Breakout condition
    price_break = latest["Close"] > pivot_high
    # Conviction: Volume > 1.5x 50-day average OR 1.5x 20-day average
    vol_conviction = (latest["Volume"] > latest["vol_ma50"] * 1.5) or (latest["vol_mult"] > 1.5)

    is_trigger = price_break and vol_conviction

    return bool(is_trigger), {"price_break": price_break, "vol_conviction": vol_conviction}


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2:
        return False
    swing_low = float(df.iloc[-(lookback + 1) : -1]["Low"].min())
    last = df.iloc[-1]
    return bool(
        (last["Low"] < swing_low)
        and (last["Close"] > swing_low)
        and (last["Close"] > last["Open"])
    )


def detect_ipo_base(symbol: str) -> bool:
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return False
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        return (dt.date.today() - ipo_date).days <= 365
    except:
        return False


def calculate_rs_raw(df: pd.DataFrame) -> float:
    def get_ret(p):
        return (
            (df["Close"].iloc[-1] / df["Close"].iloc[-p - 1] - 1) if len(df) > p else 0
        )

    return float(0.4 * get_ret(63) + 0.2 * get_ret(126) + 0.4 * get_ret(252))


# ============================================================
# 5) FUNDAMENTALS (CACHED & PARALLEL)
# ============================================================


def fetch_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    cached = db_cache.get(symbol)
    if cached:
        return cached

    import random
    import time

    # Add a tiny random delay to avoid "bot-like" behavior patterns
    time.sleep(random.uniform(0.1, 0.5))

    for attempt in range(3):  # 3 retries with backoff
        try:
            s = yf.Ticker(symbol)
            # Try to fetch only necessary info to minimize request footprint
            info = s.info or {}

            res = {
                "PE": info.get("trailingPE"),
                "ROE": info.get("returnOnEquity"),
                "DebtToEquity": info.get("debtToEquity"),
                "OperatingMargin": info.get("operatingMargins"),
                "RevenueGrowth": info.get("revenueGrowth"),
                "FundamentalQualityScore": 0,
            }

            q_score = 0
            if (res["ROE"] or 0) >= 0.15:
                q_score += 1
            if (res["DebtToEquity"] or 99) <= 1.0:
                q_score += 1
            if (res["OperatingMargin"] or 0) >= 0.12:
                q_score += 1
            if (res["RevenueGrowth"] or 0) >= 0.10:
                q_score += 1
            res["FundamentalQualityScore"] = q_score

            db_cache.set(symbol, res)
            return res
        except Exception as e:
            if "Rate Limit" in str(e) or "401" in str(e):
                time.sleep(attempt * 2 + 1)  # Exponential backoff
                continue
            break

    return {"FundamentalQualityScore": 0}


# ============================================================
# 6) CORE ENGINE
# ============================================================


@dataclass
class Candidate:
    symbol: str
    score: int
    price: float
    rs_raw: float
    details: Dict[str, Any]


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "1y",
    interval: str = "1d",
    top_n: int = 10,
    start_index: int = 0,
    manual_symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if manual_symbols:
        stocks = [s.upper().strip() for s in manual_symbols if s.strip()]
    else:
        all_stocks = get_nse_stocks()
        stocks = all_stocks[
            start_index : start_index + (universe_limit or len(all_stocks))
        ]

    counters = {
        "total_universe": len(stocks),
        "scanned": 0,
        "passed_filter": 0,
        "errors": 0,
    }

    print(f"Hyper-Scanning {len(stocks)} symbols in chunks...")
    candidates = []
    chunk_size = 50
    import gc

    for i in range(0, len(stocks), chunk_size):
        chunk = stocks[i : i + chunk_size]
        try:
            bulk_df = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval=interval,
                group_by="ticker",
                threads=10,
                progress=False,
            )
        except Exception as e:
            print(f"Chunk download failed: {e}")
            bulk_df = pd.DataFrame()

        for sym in tqdm(chunk, desc=f"Analyzing Chunk {i // chunk_size + 1}"):
            counters["scanned"] += 1
            try:
                if sym not in bulk_df.columns.levels[0]:
                    continue
                df = bulk_df[sym].dropna(how="all").copy()
                if len(df) < 90:
                    continue

                df = add_technical_indicators(normalize_ohlcv_df(df))
                latest = df.iloc[-1]

                # Step 1: Filter (Minervini Trend)
                is_trend = detect_minervini_trend(df)
                if not is_trend and not manual_symbols:
                    continue

                counters["passed_filter"] += 1

                # Step 2: Setup (Watchlist)
                is_vcp, vcp_details = detect_vcp_setup(df)
                is_sfp = detect_swing_failure(df)
                is_ipo = detect_ipo_base(sym)

                is_setup = is_vcp or is_sfp or is_ipo
                
                # Step 3: Trigger (Entry)
                is_trigger = False
                trigger_details = {}
                pivot_high = vcp_details.get("pivot_high", float(df["High"].tail(20).max()))
                
                if is_setup:
                    is_trigger, trigger_details = detect_breakout_trigger(df, pivot_high)

                # Confluence score as extra quality check
                conf_score, conf_sigs = predicta_v4_confluence(latest)

                # Build notes for clarity
                notes = []
                if is_vcp: notes.append("VCP")
                if is_sfp: notes.append("SFP")
                if is_ipo: notes.append("IPO Base")
                
                if is_vcp:
                    if vcp_details.get("is_tight"): notes.append("Tight")
                    if vcp_details.get("vol_dry"): notes.append("Vol Dry")
                
                if is_trigger:
                    notes.append("BREAKOUT")

                status = "ACTIONABLE" if (is_setup and is_trigger) else ("WATCHLIST" if is_setup else "TRENDING")

                row = {
                    "Symbol": sym,
                    "Status": status,
                    "Notes": ", ".join(notes) if notes else "Trending",
                    "Price": float(latest["Close"]),
                    "Pivot": float(pivot_high),
                    "Score": conf_score + (5 if is_trigger else (2 if is_setup else 0)),
                    "Minervini": is_trend,
                    "VCP_Setup": is_vcp,
                    "SFP": is_sfp,
                    "IPO_Base": is_ipo,
                    "Breakout": is_trigger,
                    "Vol_Dry": vcp_details.get("vol_dry", False),
                    "Tightness": vcp_details.get("is_tight", False),
                    "VolMult": float(latest["vol_mult"]),
                    "RSI": float(latest["rsi"]),
                    "ADR%": float(latest["adrp20"]),
                    **{f"C_{k}": v for k, v in conf_sigs.items()},
                }
                candidates.append(
                    Candidate(sym, row["Score"], row["Price"], calculate_rs_raw(df), row)
                )
            except Exception as e:
                counters["errors"] += 1

        del bulk_df
        gc.collect()

    if not candidates:
        return pd.DataFrame()

    candidates.sort(key=lambda x: x.score, reverse=True)
    finalists = candidates[:50]
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_results = list(
            executor.map(lambda c: fetch_stock_fundamentals(c.symbol), finalists)
        )

    for c, f in zip(finalists, f_results):
        c.details.update(f)
        c.score += f.get("FundamentalQualityScore", 0)

    df_results = pd.DataFrame([c.details for c in finalists])
    df_results["RSRating"] = (
        pd.Series([c.rs_raw for c in finalists]).rank(pct=True) * 99
    ).astype(int)

    top = (
        df_results.sort_values(by=["Score", "RSRating"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top.attrs["summary"] = counters

    print("\n" + "="*50)
    print(f" TOP {top_n} CANDIDATES ")
    print("="*50)
    if not top.empty:
        # Filter columns to show for brevity
        show_cols = ["Symbol", "Status", "Notes", "Price", "Score", "VCP_Setup", "Breakout", "VolMult"]
        print(top[show_cols].to_string(index=False))
    else:
        print("No candidates found.")
    print("="*50 + "\n")

    with pd.ExcelWriter("NSE_Swing_Screener_Report.xlsx", engine="openpyxl") as writer:
        top.to_excel(writer, sheet_name="Top10", index=False)

    return top


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe-limit", type=int, default=500)
    p.add_argument("--min-score", type=int, default=6)
    p.add_argument("--period", type=str, default="2y")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--live", action="store_true", help="Run in live monitoring mode during market hours")
    p.add_argument("--interval-min", type=int, default=15, help="Scan interval in minutes for live mode")
    p.add_argument("--test-alert", action="store_true", help="Send a test Telegram message and exit")
    return p.parse_args()


if __name__ == "__main__":
    import traceback
    import sys

    try:
        args = parse_args()

        if args.test_alert:
            print("Sending test alert to Telegram...")
            send_telegram_message("🔔 *Test Alert*: Your NSE Screener is successfully connected to Telegram! 🚀")
            print("Done. Check your phone!")
            sys.exit(0)

        if args.live:
            print(f"🚀 LIVE MONITORING MODE STARTED (Interval: {args.interval_min}m)")
            print(f"Alerts will be sent to Telegram for ACTIONABLE breakouts.")
            
            while True:
                if is_market_open():
                    print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] Starting live scan...")
                    results = run_full_system(
                        args.universe_limit,
                        args.min_score,
                        args.period,
                        "15m",  # Use shorter interval for live triggers if possible, or stick to 1d
                        args.top_n,
                        args.start_index,
                    )
                    
                    # Check for new Actionable alerts
                    if not results.empty:
                        actionable = results[results["Status"] == "ACTIONABLE"]
                        for _, row in actionable.iterrows():
                            if row["Symbol"] not in alerted_today:
                                msg = (
                                    f"🔥 *ENTRY CONFIRMED: {row['Symbol']}*\n"
                                    f"💰 Price: {row['Price']:.2f}\n"
                                    f"🎯 Pivot: {row['Pivot']:.2f}\n"
                                    f"📝 Notes: {row['Notes']}\n"
                                    f"📊 Score: {row['Score']}"
                                )
                                send_telegram_message(msg)
                                alerted_today.add(row["Symbol"])
                    
                    print(f"Scan complete. Sleeping for {args.interval_min} minutes...")
                    time.sleep(args.interval_min * 60)
                else:
                    # If market is closed, check again in 5 minutes
                    # Also reset alerts at start of a new day
                    if dt.datetime.now().hour == 0:
                        alerted_today.clear()
                    
                    print(f"\r[{dt.datetime.now().strftime('%H:%M:%S')}] Market is closed. Waiting...", end="")
                    time.sleep(300)
        else:
            results = run_full_system(
                args.universe_limit,
                args.min_score,
                args.period,
                args.interval,
                args.top_n,
                args.start_index,
            )
            
            # In non-live mode (like GitHub Actions), send alerts for ACTIONABLE stocks
            if not results.empty and (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
                actionable = results[results["Status"] == "ACTIONABLE"]
                for _, row in actionable.iterrows():
                    msg = (
                        f"🔥 *ENTRY CONFIRMED: {row['Symbol']}*\n"
                        f"💰 Price: {row['Price']:.2f}\n"
                        f"🎯 Pivot: {row['Pivot']:.2f}\n"
                        f"📝 Notes: {row['Notes']}\n"
                        f"📊 Score: {row['Score']}"
                    )
                    send_telegram_message(msg)
                    print(f"Alert sent for {row['Symbol']}")
    except Exception as e:
        print("❌ CRITICAL ERROR IN SCREENER:")
        traceback.print_exc()
        sys.exit(1)
