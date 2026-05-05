"""
NSE Breakout Scanner Pro — Institutional-Grade Swing Trading Engine
===================================================================
Nifty 500 daily + Full universe weekly discovery.
ATR-based risk management. Market regime awareness.
Threaded analysis. Telegram alerts.
"""

import argparse
import concurrent.futures
import datetime as dt
import gc
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
_CACHE_DIR = Path(os.environ.get("NSE_SCREENER_CACHE_DIR", str(Path(tempfile.gettempdir()) / "nse_screener_cache")))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = Path("watchlist_persistent.json")
NIFTY500_CACHE = _CACHE_DIR / "nifty500.csv"
FULL_EQUITY_CACHE = _CACHE_DIR / "EQUITY_L.csv"
SECTOR_CACHE = _CACHE_DIR / "sectors.json"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Strategy constants
MAX_PRICE = 5000           # Skip stocks above this price for swing sizing
MIN_DATA_DAYS = 90         # Minimum trading days required
VOLUME_BREAKOUT_MULT = 1.5 # Volume must be 1.5x average for breakout
STRONG_CLOSE_PCT = 0.40    # Close must be in top 60% of daily range
WIN_RATE_MIN_SAMPLES = 3   # Minimum breakout samples for WinProb display
TIGHTNESS_STAR = 8         # Tightness score threshold for star rating
WINPROB_STAR = 65          # WinProb threshold for star rating
DISCOVERY_MIN_SCORE = 8    # Non-Nifty500 stocks must score this high to be promoted
CHUNK_SIZE = 200           # Stocks per Yahoo download batch
MAX_WORKERS = 8            # Threads for parallel analysis


def get_now_ist():
    return dt.datetime.now(dt.timezone.utc) + IST_OFFSET


# ============================================================
# PERSISTENT STATE
# ============================================================

def load_state() -> dict:
    default = {"watchlist": [], "alerted_today": [], "alerted_date": "", "discovery": [], "discovery_date": "", "eod_date": ""}
    if not STATE_FILE.exists():
        return default
    try:
        data = json.loads(STATE_FILE.read_text().strip() or "{}")
        if isinstance(data, list):
            state_dict = {**default, "watchlist": data}
        else:
            state_dict = {**default, **data}

        return state_dict
    except Exception:
        return default


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Skipped — no credentials.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if r.status_code == 200:
            print(f"[TG] ✅ Sent ({len(msg)} chars)")
        else:
            print(f"[TG] {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[TG] Error: {e}")


# ============================================================
# STOCK UNIVERSE
# ============================================================

def _clean_symbol(raw: str) -> str:
    return raw.split(",")[0].strip().strip('"').replace("$", "") + ".NS"


def _fetch_cached_csv(url: str, cache_path: Path, ttl_hours: int = 24) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if cache_path.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age.total_seconds() < ttl_hours * 3600:
            lines = cache_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return lines
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        return resp.text.splitlines()
    except Exception:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return []


def get_nifty500() -> List[str]:
    """Fetch the official Nifty 500 constituent list from NSE."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    lines = _fetch_cached_csv(url, NIFTY500_CACHE)
    if not lines:
        return []
    try:
        df = pd.read_csv(StringIO("\n".join(lines)))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        syms = [s.strip() + ".NS" for s in df[col].dropna().tolist()]
        # NSE includes dummy test entries (DUMMYVEDL1-4) in official CSV — filter them
        syms = [s for s in syms if not s.startswith("DUMMY")]
        return syms
    except Exception:
        return [_clean_symbol(l) for l in lines[1:] if l.strip() and not l.startswith("Dummy")]


def get_full_universe() -> List[str]:
    """Fetch the complete NSE equity list (2600+ stocks)."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    lines = _fetch_cached_csv(url, FULL_EQUITY_CACHE)
    return [_clean_symbol(l) for l in lines[1:] if l.strip()]


def get_sector(sym: str) -> str:
    cache = {}
    if SECTOR_CACHE.exists():
        try:
            cache = json.loads(SECTOR_CACHE.read_text())
        except Exception:
            pass
    if sym in cache:
        return cache[sym]
    try:
        sector = yf.Ticker(sym).info.get("sector", "Unknown")
        cache[sym] = sector
        SECTOR_CACHE.write_text(json.dumps(cache))
        return sector
    except Exception:
        return "Unknown"


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # EMAs
    for p in [8, 21, 50, 150, 200]:
        df[f"ema{p}"] = ta.trend.ema_indicator(c, p)

    # Momentum
    df["rsi"] = ta.momentum.rsi(c, 14)
    macd_obj = ta.trend.MACD(c)
    df["macd"] = macd_obj.macd()
    df["macd_sig"] = macd_obj.macd_signal()
    df["adx"] = ta.trend.adx(h, l, c, 14)
    df["stoch"] = ta.momentum.stoch(h, l, c, 14)

    # Volume
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ma50"] = v.rolling(50).mean()
    df["vol_mult"] = v / df["vol_ma20"]

    # Volatility
    df["atr"] = ta.volatility.average_true_range(h, l, c, 14)
    df["adrp"] = ((h - l).rolling(20).mean() / c) * 100
    df["drp"] = ((h - l) / c) * 100

    # Delta proxy (buying vs selling pressure)
    hl = (h - l).replace(0, np.nan)
    df["delta"] = ((c - l) - (h - c)) / hl * v

    # Institutional absorption: narrow candle + high volume + bullish close
    df["absorption"] = ((h - l) < df["atr"] * 0.8) & (v > df["vol_ma50"] * 1.2) & (c >= df["Open"])
    df["accum_days"] = df["absorption"].rolling(20).sum()

    return df


# ============================================================
# MARKET REGIME FILTER
# ============================================================

def check_market_regime() -> Tuple[bool, str]:
    """Check if Nifty 50 is in a healthy uptrend. Avoids breakout trading in bear markets."""
    try:
        nifty = yf.download("^NSEI", period="2y", interval="1d", progress=False)
        if nifty.empty or len(nifty) < 200:
            return True, "Unknown"  # If we can't check, assume OK
        
        # Flatten MultiIndex if present
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        
        c = nifty["Close"].iloc[-1]
        ema21 = ta.trend.ema_indicator(nifty["Close"], 21).iloc[-1]
        ema50 = ta.trend.ema_indicator(nifty["Close"], 50).iloc[-1]
        ema200 = ta.trend.ema_indicator(nifty["Close"], 200).iloc[-1]

        if c > ema50 and ema50 > ema200:
            return True, "Confirmed Bull"
        elif c > ema21 and c > ema50:
            return True, "Short-Term Bull"
        elif c > ema200:
            return True, "Neutral"
        elif c > ema21:
            return True, "Recovery"
        else:
            return False, "Bear"
    except Exception:
        return True, "Unknown"


# ============================================================
# SETUP DETECTION
# ============================================================

def confluence_score(latest: pd.Series) -> int:
    c = latest.get("Close", 0)
    e50 = latest.get("ema50", 0)
    e200 = latest.get("ema200", 0)
    trend = bool(c > e50 > e200) if not pd.isna(e200) else bool(c > e50)

    checks = [
        latest.get("macd", 0) > latest.get("macd_sig", 0),
        latest.get("rsi", 0) >= 55,
        latest.get("stoch", 0) >= 60,
        latest.get("vol_mult", 0) >= 1.2,
        latest.get("delta", 0) > 0,
        trend,
        latest.get("adx", 0) >= 20,
    ]
    return sum(bool(x) for x in checks)


def detect_minervini(df: pd.DataFrame) -> bool:
    if len(df) < 200:
        return False
    c = df["Close"].iloc[-1]
    e50, e150, e200 = df["ema50"].iloc[-1], df["ema150"].iloc[-1], df["ema200"].iloc[-1]
    e200_prev = df["ema200"].iloc[-20]
    h52, l52 = df["High"].tail(252).max(), df["Low"].tail(252).min()

    return all([
        c > e150 > e200,
        e200 > e200_prev,
        e50 > e150 > e200,
        c > e50,
        c >= l52 * 1.3,
        c >= h52 * 0.75,
    ])


def detect_vcp(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """VCP: Volatility Contraction Pattern with tightness scoring and footprint detection."""
    if len(df) < 60:
        return False, {}

    w = df.tail(60)
    latest = w.iloc[-1]
    pivot = float(w["High"].max())

    near_highs = latest["Close"] >= pivot * 0.90
    vol_dry = w["Volume"].tail(3).mean() < latest["vol_ma50"] * 0.8

    # Tightness score (1-10) from coefficient of variation
    std20 = w["Close"].tail(20).std()
    mean20 = w["Close"].tail(20).mean()
    cv = (std20 / mean20) * 100 if mean20 > 0 else 100
    tightness = max(1, min(10, int(12 - cv)))

    # Institutional footprints
    footprints = int(latest.get("accum_days", 0)) >= 2

    is_setup = near_highs and vol_dry and (tightness >= 6 or footprints)

    return bool(is_setup), {
        "pivot_high": pivot,
        "tightness_score": tightness,
        "vol_dry": vol_dry,
        "has_footprints": footprints,
    }


def detect_sfp(df: pd.DataFrame) -> bool:
    """Swing Failure Pattern: false breakdown followed by bullish reversal."""
    if len(df) < 22:
        return False
    swing_low = float(df.iloc[-21:-1]["Low"].min())
    last = df.iloc[-1]
    return bool(last["Low"] < swing_low and last["Close"] > swing_low and last["Close"] > last["Open"])


def detect_breakout(df: pd.DataFrame, pivot: float) -> bool:
    """Price breaks pivot on volume with strong close (fakeout protection)."""
    latest = df.iloc[-1]
    price_break = latest["Close"] > pivot
    vol_ok = latest["Volume"] > latest["vol_ma50"] * VOLUME_BREAKOUT_MULT or latest["vol_mult"] > VOLUME_BREAKOUT_MULT

    rng = latest["High"] - latest["Low"]
    strong_close = latest["Close"] >= (latest["Low"] + rng * STRONG_CLOSE_PCT) if rng > 0 else True

    return bool(price_break and vol_ok and strong_close)


def calc_rs(df: pd.DataFrame) -> float:
    def ret(p):
        return (df["Close"].iloc[-1] / df["Close"].iloc[-p - 1] - 1) if len(df) > p else 0
    return float(0.4 * ret(63) + 0.2 * ret(126) + 0.4 * ret(252))


def calc_win_rate(df: pd.DataFrame) -> Tuple[int, float]:
    """Mini-backtest: how often did similar breakouts hit +10% before -7% stop."""
    if len(df) < 50:
        return 0, 0.0
    prev_h20 = df["High"].shift(1).rolling(20).max()
    bo_mask = (df["Close"] > prev_h20) & (df["Volume"] > df["vol_ma50"] * VOLUME_BREAKOUT_MULT)
    bo_idx = np.where(bo_mask)[0]

    wins, total = 0, 0
    for idx in bo_idx:
        if idx > len(df) - 15:
            continue
        entry = df["Close"].iloc[idx]
        ahead = df.iloc[idx + 1: min(idx + 21, len(df))]
        for _, row in ahead.iterrows():
            if row["Low"] <= entry * 0.93:
                total += 1
                break
            if row["High"] >= entry * 1.10:
                wins += 1
                total += 1
                break
    return total, (wins / total * 100) if total > 0 else 0.0


def calc_risk_reward(price: float, pivot: float, atr: float) -> Tuple[float, float, float]:
    """ATR-based stop loss and target. Returns (stop, target, rr_ratio)."""
    stop = price - (atr * 2)       # 2 ATR stop
    target = price + (atr * 4)     # 4 ATR target = 2:1 R:R minimum
    rr = (target - price) / (price - stop) if price > stop else 0
    return round(stop, 2), round(target, 2), round(rr, 1)


# ============================================================
# CORE ENGINE
# ============================================================

@dataclass
class Candidate:
    symbol: str
    score: int
    price: float
    rs_raw: float
    details: Dict[str, Any]


def analyze_ticker(sym: str, df: pd.DataFrame, is_manual: bool) -> Optional[Candidate]:
    """Analyze a single ticker. Returns Candidate or None."""
    try:
        df = add_indicators(normalize_df(df))
        latest = df.iloc[-1]
        price = float(latest["Close"])

        if price > MAX_PRICE and not is_manual:
            return None

        # Detect IPO (less than 1 year of trading data)
        is_ipo = len(df) < 252

        # Strict Minervini trend requires 200 days. True IPOs are exempt from this strict check.
        is_trend = detect_minervini(df)
        if not is_trend and not is_ipo and not is_manual:
            return None

        is_vcp, vcp = detect_vcp(df)
        is_sfp = detect_sfp(df)
        is_setup = is_vcp or is_sfp or is_ipo

        pivot = vcp.get("pivot_high", float(df["High"].tail(20).max()))
        is_breakout = detect_breakout(df, pivot) if is_setup else False

        score = confluence_score(latest) + (5 if is_breakout else (2 if is_setup else 0))
        sector = get_sector(sym) if is_setup else "Unknown"

        # Win probability
        total_bo, win_pct = calc_win_rate(df) if is_setup else (0, 0.0)
        win_display = f"{win_pct:.0f}%" if total_bo >= WIN_RATE_MIN_SAMPLES else "NA"

        # Risk/Reward
        atr = float(latest["atr"])
        stop, target, rr = calc_risk_reward(price, pivot, atr)

        # Notes (ultra-compact for Telegram)
        notes = []
        if is_vcp:
            notes.append("VCP")
            notes.append(f"T:{vcp.get('tightness_score', 1)}")
            if vcp.get("has_footprints"):
                notes.append("Accum")
        if is_sfp:
            notes.append("SFP")
        if is_ipo:
            notes.append("IPO")
        notes.append(f"W:{win_display}")

        status = "ACTIONABLE" if (is_setup and is_breakout) else ("WATCHLIST" if is_setup else "TRENDING")

        return Candidate(sym, score, price, calc_rs(df), {
            "Symbol": sym, "Sector": sector, "Status": status,
            "Notes": ", ".join(notes),
            "Price": price, "Pivot": round(pivot, 2),
            "Stop": stop, "Target": target, "RR": rr,
            "Score": score,
            "VCP": is_vcp, "SFP": is_sfp, "IPO": is_ipo,
            "Breakout": is_breakout,
            "WinProb": win_pct if total_bo >= WIN_RATE_MIN_SAMPLES else 0.0,
            "TightnessScore": vcp.get("tightness_score", 0),
            "VolMult": float(latest["vol_mult"]),
            "RSI": float(latest["rsi"]),
        })
    except Exception as e:
        print(f"  [WARN] {sym}: {e}", file=sys.stderr)
        return None


def scan_universe(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
    is_manual: bool = False,
) -> pd.DataFrame:
    """Download and analyze a list of symbols. Returns sorted top 10 DataFrame."""
    print(f"  Scanning {len(symbols)} symbols...")
    candidates = []
    errors = 0

    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i: i + CHUNK_SIZE]
        try:
            bulk = yf.download(" ".join(chunk), period=period, interval=interval,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print(f"  Download failed: {e}")
            continue

        # Extract valid DataFrames
        items = []
        is_multi = isinstance(bulk.columns, pd.MultiIndex)
        for sym in chunk:
            try:
                if is_multi:
                    if sym not in bulk.columns.levels[0]:
                        continue
                    df = bulk[sym].dropna(how="all").copy()
                else:
                    if bulk.empty:
                        continue
                    df = bulk.dropna(how="all").copy()
                if len(df) >= MIN_DATA_DAYS:
                    items.append((sym, df))
            except Exception:
                continue

        # Parallel analysis
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(analyze_ticker, s, d, is_manual): s for s, d in items}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                          desc=f"  Chunk {i // CHUNK_SIZE + 1}"):
                try:
                    result = f.result()
                    if result:
                        candidates.append(result)
                except Exception:
                    errors += 1

        del bulk
        gc.collect()

    if not candidates:
        return pd.DataFrame()

    df_out = pd.DataFrame([c.details for c in candidates])
    df_out["RSRating"] = pd.Series([c.rs_raw for c in candidates]).rank(pct=True).mul(99).astype(int)

    return (
        df_out
        .sort_values(["Breakout", "WinProb", "TightnessScore", "Score", "RSRating"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


# ============================================================
# ALERT FORMATTING
# ============================================================

def _short(sym: str) -> str:
    """Strip .NS suffix for cleaner display."""
    return sym.replace(".NS", "")


def fmt_breakout(row) -> str:
    s = _short(row['Symbol'])
    # Build a human-readable "why" line
    why_parts = []
    notes = row.get("Notes", "")
    if "VCP" in notes:
        t = row.get("TightnessScore", 0)
        why_parts.append(f"VCP base, coiled tight ({t}/10)")
    if "Accum" in notes:
        why_parts.append("institutions accumulating")
    if "SFP" in notes:
        why_parts.append("swing failure reversal")
    if "IPO" in notes:
        why_parts.append("IPO base breakout")
    why = ", ".join(why_parts) if why_parts else "clean setup"

    win = notes.split("W:")[1].split(",")[0].split(")")[0].strip() if "W:" in notes else "NA"
    win_line = f"{win} similar setups worked" if win != "NA" else "new pattern, no history"

    return (
        f"*{s}*\n"
        f"BREAKOUT CONFIRMED\n\n"
        f"Entry: {row['Price']:.0f}\n"
        f"Stop: {row['Stop']:.0f} | Target: {row['Target']:.0f}\n"
        f"R:R {row['RR']}:1 | Vol {row['VolMult']:.1f}x\n\n"
        f"Why: {why}\n"
        f"Win rate: {win_line}\n"
        f"Sector: {row['Sector']}"
    )


def fmt_watchlist(df: pd.DataFrame, title: str = "WATCHLIST") -> str:
    lines = []
    for _, r in df.head(10).iterrows():
        s = _short(r['Symbol'])
        star = ">" if r.get("WinProb", 0) >= WINPROB_STAR or r.get("TightnessScore", 0) >= TIGHTNESS_STAR else "-"
        lines.append(f"{star} {s} {r['Price']:.0f}\n  {r['Notes']}")
    return f"*{title}*\n\n" + "\n\n".join(lines)


def fmt_discovery(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.head(5).iterrows():
        s = _short(r['Symbol'])
        notes = r.get("Notes", "")
        # Human-readable description
        desc_parts = []
        if "VCP" in notes:
            t = r.get("TightnessScore", 0)
            desc_parts.append(f"tight base ({t}/10)")
        if "Accum" in notes:
            desc_parts.append("smart money entering")
        if "SFP" in notes:
            desc_parts.append("reversal pattern")
        if "IPO" in notes:
            desc_parts.append("new listing, building base")
        desc = ", ".join(desc_parts) if desc_parts else "forming setup"
        lines.append(f"{s} {r['Price']:.0f}\n  {desc}")
    return "*HIDDEN GEMS THIS WEEK*\n\n" + "\n\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="NSE Breakout Scanner Pro")
    p.add_argument("--period", default="2y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--test-alert", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        if args.test_alert:
            send_telegram("*Test Alert*: Scanner connected.")
            sys.exit(0)

        state = load_state()
        now = get_now_ist()
        today = now.strftime("%Y-%m-%d")

        # --- Early Exits for 24/7 Cron schedules ---
        if now.weekday() >= 5:
            print("Weekend. Market is closed. Exiting.")
            sys.exit(0)

        current_time = now.hour * 100 + now.minute
        if current_time < 910:
            print("Before market hours. Exiting.")
            sys.exit(0)

        if state.get("eod_date") == today:
            print(f"EOD already completed for {today}. Exiting.")
            sys.exit(0)

        # Day reset
        if state.get("alerted_date") != today:
            state["alerted_today"] = []
            state["alerted_date"] = today

        # Weekly discovery reset (Friday)
        if now.weekday() == 4 and state.get("discovery_date") != today:
            state["discovery"] = []
            state["discovery_date"] = today

        alerted = set(state["alerted_today"])
        discovery_syms = state.get("discovery", [])
        # --- Time Windows ---
        is_morning = now.hour == 9 and 10 <= now.minute <= 14
        is_eod = now.hour > 15 or (now.hour == 15 and now.minute >= 28)

        print(f"IST: {now.strftime('%Y-%m-%d %H:%M')} | Watchlist: {len(state['watchlist'])} | Discovery: {len(discovery_syms)}")

        # --- Market Regime Check ---
        regime_ok, regime = check_market_regime()
        print(f"Market Regime: {regime}")
        if not regime_ok:
            send_telegram(f"*Market Regime: {regime}*\nBreakout quality is low in bear markets. Scanning with caution.")

        # --- Morning startup notification ---
        if is_morning:
            send_telegram(
                f"\U0001f50d *Morning Scan Started*\n"
                f"IST: {now.strftime('%H:%M')} | Market: {regime}\n"
                f"Watchlist: {len(state['watchlist'])} symbols"
            )

        # --- Build scan universe ---
        nifty500 = get_nifty500()
        scan_symbols = list(set(nifty500 + discovery_syms + state.get("watchlist", [])))
        print(f"Daily Universe: {len(scan_symbols)} (N500: {len(nifty500)}, Discovery: {len(discovery_syms)})")

        # --- Morning Priority (09:10-09:14 IST) ---
        if is_morning:
            print("\nMORNING PRIORITY SCAN (Pre-open)...")
            if state.get("watchlist"):
                priority = scan_universe(state["watchlist"], args.period, args.interval, is_manual=True)
                if not priority.empty:
                    for _, row in priority.iterrows():
                        if row["Status"] == "ACTIONABLE" and row["Symbol"] not in alerted:
                            send_telegram(fmt_breakout(row))
                            alerted.add(row["Symbol"])
                    wl = priority[priority["Status"].isin(["WATCHLIST", "ACTIONABLE"])]
                    if not wl.empty:
                        send_telegram(fmt_watchlist(wl, "MORNING WATCHLIST"))
            
            # Skip main scan during pre-open
            scan_symbols = []

        # --- Main Scan ---
        results = pd.DataFrame()
        actionable = pd.DataFrame()
        watchlist = pd.DataFrame()
        
        if scan_symbols:
            print("\nMAIN SCAN...")
            results = scan_universe(scan_symbols, args.period, args.interval)

            if not results.empty:
                actionable = results[results["Status"] == "ACTIONABLE"]
                watchlist = results[results["Status"] == "WATCHLIST"]

                for _, row in actionable.iterrows():
                    if row["Symbol"] not in alerted:
                        send_telegram(fmt_breakout(row))
                        alerted.add(row["Symbol"])

                # Intraday: merge new finds into existing watchlist
                if not is_eod:
                    existing = set(state.get("watchlist", []))
                    new_wl = set(watchlist["Symbol"].tolist()) if not watchlist.empty else set()
                    fresh = new_wl - existing
                    state["watchlist"] = list(existing | new_wl)
                    # Notify only about newly discovered watchlist stocks
                    if fresh and not watchlist.empty:
                        new_entries = watchlist[watchlist["Symbol"].isin(fresh)]
                        if not new_entries.empty:
                            send_telegram(fmt_watchlist(new_entries, f"\U0001f195 NEW SETUPS ({len(fresh)})"))

        # --- EOD (15:30 IST) ---
        now = get_now_ist()
        is_eod = now.hour > 15 or (now.hour == 15 and now.minute >= 28)
        if is_eod:
            print("\nEND OF DAY...")
            if not results.empty:
                combined = pd.concat([watchlist, actionable]).drop_duplicates(subset=["Symbol"])
                # Merge today's scan with accumulated watchlist
                existing = set(state.get("watchlist", []))
                state["watchlist"] = list(existing | set(combined["Symbol"].tolist()))

                if not combined.empty:
                    # Sector heatmap
                    sectors = combined[combined["Sector"] != "Unknown"]["Sector"].value_counts()
                    hot = sectors[sectors >= 2]
                    sector_msg = ""
                    if not hot.empty:
                        sector_msg = "\n\nHot Sectors:\n" + "\n".join(f"- {s} ({c})" for s, c in hot.items())
                    send_telegram(fmt_watchlist(combined) + sector_msg)
                else:
                    send_telegram("*Market Closed* \u2014 No clean setups for tomorrow.")
            else:
                send_telegram("*Market Closed* \u2014 No clean setups for tomorrow.")

            # EOD summary
            send_telegram(
                f"\U0001f4ca *Day Complete*\n"
                f"Watchlist: {len(state['watchlist'])} | Alerted: {len(alerted)}\n"
                f"Market: {regime}"
            )

        # --- Weekly Discovery (Friday EOD) ---
        if now.weekday() == 4 and is_eod:
            print("\nWEEKLY DEEP DIVE — Full Universe...")
            full_universe = get_full_universe()
            non_nifty = [s for s in full_universe if s not in set(nifty500)]
            print(f"  Non-Nifty500 stocks to scan: {len(non_nifty)}")

            discovery_results = scan_universe(non_nifty, args.period, args.interval)
            if not discovery_results.empty:
                # Only promote high-quality stocks
                elite = discovery_results[discovery_results["Score"] >= DISCOVERY_MIN_SCORE]
                if not elite.empty:
                    promoted = elite["Symbol"].tolist()
                    state["discovery"] = promoted
                    send_telegram(fmt_discovery(elite))
                    print(f"  Promoted {len(promoted)} stocks to discovery list.")

        # --- Save ---
        if is_eod:
            state["eod_date"] = today
            
        state["alerted_today"] = list(alerted)
        save_state(state)
        print(f"\nState saved: {len(state['watchlist'])} watchlist, {len(alerted)} alerted, {len(state.get('discovery', []))} discovery")

    except Exception as e:
        print("CRITICAL ERROR:")
        traceback.print_exc()
        try:
            send_telegram(f"*Scanner Crashed*: {str(e)[:200]}")
        except Exception:
            pass
        sys.exit(1)
