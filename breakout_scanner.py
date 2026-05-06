"""
NSE Momentum Burst Scanner
===========================
Strict implementation of Swing Trading Playbook.
Phase 1: Prior uptrend (EMA stack + U/D ratio)
Phase 2: Base formation (5-15 days, quiet volume, SFP/Doji)
Phase 3: Breakout confirmation (volume, candle, confluence)
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
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
_CACHE_DIR = Path(os.environ.get("NSE_SCREENER_CACHE_DIR",
                  str(Path(tempfile.gettempdir()) / "nse_screener_cache")))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = Path("watchlist_persistent.json")
NIFTY500_CACHE = _CACHE_DIR / "nifty500.csv"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Playbook constants
MIN_PRICE = 50
MAX_PRICE = 800
MIN_DATA_DAYS = 150
BASE_MIN_DAYS = 5
BASE_MAX_DAYS = 15
RVOL_MULT = 1.5
UD_RATIO_MIN = 1.4
CONFLUENCE_MIN = 4
RISK_MAX_PCT = 0.03
CHUNK_SIZE = 200
MAX_WORKERS = 8


def get_now_ist():
    return dt.datetime.now(dt.timezone.utc) + IST_OFFSET


# ============================================================
# STATE & TELEGRAM
# ============================================================

def load_state() -> dict:
    default = {"watchlist": [], "alerted_today": [],
               "alerted_date": "", "eod_date": ""}
    if not STATE_FILE.exists():
        return default
    try:
        data = json.loads(STATE_FILE.read_text().strip() or "{}")
        if isinstance(data, list):
            return {**default, "watchlist": data}
        return {**default, **data}
    except Exception:
        return default


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Skipped — no credentials.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "Markdown"},
            timeout=10)
        print(f"[TG] {'✅' if r.status_code == 200 else r.status_code}")
    except Exception as e:
        print(f"[TG] Error: {e}")


# ============================================================
# STOCK UNIVERSE
# ============================================================

def _fetch_cached_csv(url: str, cache_path: Path,
                      ttl_hours: int = 24) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if cache_path.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(
            cache_path.stat().st_mtime)
        if age.total_seconds() < ttl_hours * 3600:
            return cache_path.read_text(
                encoding="utf-8", errors="ignore").splitlines()
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        return resp.text.splitlines()
    except Exception:
        if cache_path.exists():
            return cache_path.read_text(
                encoding="utf-8", errors="ignore").splitlines()
        return []


def get_nifty500() -> List[str]:
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    lines = _fetch_cached_csv(url, NIFTY500_CACHE)
    if not lines:
        return []
    try:
        df = pd.read_csv(StringIO("\n".join(lines)))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        syms = [s.strip() + ".NS" for s in df[col].dropna().tolist()]
        return [s for s in syms if not s.startswith("DUMMY")]
    except Exception:
        return []


# ============================================================
# INDICATORS
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

    # EMAs per playbook
    for p in [8, 21, 55, 144]:
        df[f"ema{p}"] = ta.trend.ema_indicator(c, p)

    # Momentum
    df["rsi"] = ta.momentum.rsi(c, 14)
    macd = ta.trend.MACD(c)
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["adx"] = ta.trend.adx(h, l, c, 14)
    df["stoch"] = ta.momentum.stoch(h, l, c, 14)

    # Volume
    df["vol_ma50"] = v.rolling(50).mean()
    df["rvol"] = v / df["vol_ma50"]

    # Volatility
    df["atr"] = ta.volatility.average_true_range(h, l, c, 14)
    df["adr"] = (h - l).rolling(20).mean()

    # U/D ratio (count of up days vs down days over 20 sessions)
    up = (c > c.shift(1)).astype(int)
    dn = (c < c.shift(1)).astype(int)
    df["ud_ratio"] = up.rolling(20).sum() / dn.rolling(20).sum().clip(lower=1)

    # Daily change
    df["pct_chg"] = c.pct_change()

    return df


# ============================================================
# MARKET REGIME — Playbook: Nifty vs EMA-21 and EMA-55
# ============================================================

def check_market_regime() -> Tuple[bool, str]:
    try:
        nifty = yf.download("^NSEI", period="1y", interval="1d",
                            progress=False)
        if nifty.empty or len(nifty) < 60:
            return True, "Unknown"

        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        c = nifty["Close"].iloc[-1]
        e21 = ta.trend.ema_indicator(nifty["Close"], 21)
        e55 = ta.trend.ema_indicator(nifty["Close"], 55)

        ema21_now, ema55_now = e21.iloc[-1], e55.iloc[-1]
        ema21_5d, ema55_5d = e21.iloc[-5], e55.iloc[-5]
        both_up = ema21_now > ema21_5d and ema55_now > ema55_5d

        if c > ema21_now and c > ema55_now and both_up:
            return True, "Uptrend"
        elif c < ema21_now and c < ema55_now:
            return False, "Bear"
        else:
            return True, "Selective"
    except Exception:
        return True, "Unknown"


# ============================================================
# PHASE 1 — Prior Uptrend
# ============================================================

def detect_uptrend(df: pd.DataFrame) -> bool:
    """EMA-8 > EMA-21 > EMA-55, EMA-144 rising, U/D >= 1.4"""
    if len(df) < 144:
        return False
    latest = df.iloc[-1]
    e8 = latest["ema8"]
    e21 = latest["ema21"]
    e55 = latest["ema55"]

    # EMA stack
    if not (e8 > e21 > e55):
        return False

    # EMA-144 pointing up or flat (not falling)
    e144_now = df["ema144"].iloc[-1]
    e144_prev = df["ema144"].iloc[-20]
    if e144_now < e144_prev * 0.99:
        return False

    # U/D ratio
    ud = latest["ud_ratio"]
    if pd.isna(ud) or ud < UD_RATIO_MIN:
        return False

    return True


# ============================================================
# PHASE 2 — Base Detection
# ============================================================

def find_base(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Find a valid 5-15 day consolidation base.
    For WATCHLIST: base includes today (stock still resting).
    For BREAKOUT: base is in prior days, today is breakout candle.

    Returns (found, info_dict).
    info_dict keys: pivot, base_len, has_sfp, doji_count, base_end_idx
    """
    if len(df) < 20:
        return False, {}

    # Try to find a base ending at different points
    # Option A: base ends at today (stock in base = WATCHLIST candidate)
    # Option B: base ends at yesterday (today could be breakout)
    for end_offset in [0, 1]:
        end_idx = len(df) - 1 - end_offset
        if end_idx < BASE_MAX_DAYS:
            continue

        for blen in range(BASE_MAX_DAYS, BASE_MIN_DAYS - 1, -1):
            start_idx = end_idx - blen + 1
            if start_idx < 1:
                continue

            base = df.iloc[start_idx:end_idx + 1]
            pivot = float(base["High"].max())

            # --- BASE VALIDITY CHECKS ---

            # 1. Daily candles are small (range < ADR)
            adr_at_start = df["adr"].iloc[start_idx]
            if pd.isna(adr_at_start) or adr_at_start <= 0:
                continue
            ranges = base["High"] - base["Low"]
            small_pct = (ranges < adr_at_start * 1.2).sum() / len(base)
            if small_pct < 0.5:
                continue

            # 2. Volume drying up during base
            vol_ma = df["vol_ma50"].iloc[end_idx]
            if pd.isna(vol_ma) or vol_ma <= 0:
                continue
            base_vol_avg = base["Volume"].mean()
            if base_vol_avg >= vol_ma:
                continue

            # 3. No close below EMA-21
            if (base["Close"] < base["ema21"]).any():
                continue

            # 4. No 3 consecutive down days
            down = (base["Close"] < base["Close"].shift(1)).astype(int)
            if len(base) >= 3 and down.rolling(3).sum().max() >= 3:
                continue

            # 5. No volume spike on a down day (distribution)
            down_mask = base["Close"] < base["Open"]
            vol_spike = base["Volume"] > vol_ma * 1.5
            if (down_mask & vol_spike).any():
                continue

            # 6. No close below EMA-55
            if (base["Close"] < base["ema55"]).any():
                continue

            # 7. Price holding near resistance
            last_close = base["Close"].iloc[-1]
            if last_close < pivot * 0.92:
                continue

            # --- BONUS SIGNALS ---

            # SFP: price dips below prior low, closes back above
            has_sfp = False
            for i in range(2, len(base)):
                prior_low = base["Low"].iloc[:i].min()
                candle = base.iloc[i]
                if (candle["Low"] < prior_low
                        and candle["Close"] > prior_low
                        and candle["Close"] > candle["Open"]):
                    has_sfp = True
                    break

            # Doji candles
            body = abs(base["Close"] - base["Open"])
            wick = (base["High"] - base["Low"]).replace(0, np.nan)
            doji_count = int((body / wick < 0.15).sum())

            return True, {
                "pivot": pivot,
                "base_len": blen,
                "has_sfp": has_sfp,
                "doji_count": doji_count,
                "base_end_offset": end_offset,
            }

    return False, {}


# ============================================================
# PHASE 3 — Breakout Confirmation
# ============================================================

def is_breakout_candle(df: pd.DataFrame, pivot: float) -> bool:
    """Check if the LATEST candle is a valid breakout."""
    if len(df) < 5:
        return False
    latest = df.iloc[-1]
    prev_close = float(df["Close"].iloc[-2])

    # 1. Close above resistance
    if latest["Close"] <= pivot:
        return False

    # 2. Green candle with body > 60% of range
    rng = latest["High"] - latest["Low"]
    if rng <= 0:
        return False
    body = latest["Close"] - latest["Open"]
    if body <= 0 or body / rng < 0.60:
        return False

    # 3. Close in top 10% of day's range
    if latest["Close"] < latest["High"] - rng * 0.10:
        return False

    # 4. RVol >= 1.5x
    if pd.isna(latest["rvol"]) or latest["rvol"] < RVOL_MULT:
        return False

    # 5. Move >= 4% from previous close
    if (latest["Close"] / prev_close - 1) < 0.04:
        return False

    # 6. NOT up 3 consecutive days before today
    if len(df) >= 5:
        d1 = df["Close"].iloc[-2] > df["Close"].iloc[-3]
        d2 = df["Close"].iloc[-3] > df["Close"].iloc[-4]
        d3 = df["Close"].iloc[-4] > df["Close"].iloc[-5]
        if d1 and d2 and d3:
            return False

    return True


def confluence_score(latest: pd.Series) -> int:
    """Playbook: at least 4 of 6 must be true."""
    checks = [
        # MACD bullish crossover or positive histogram
        latest.get("macd_hist", 0) > 0,
        # RSI above 55
        latest.get("rsi", 0) >= 55,
        # Stochastic above 50
        latest.get("stoch", 0) >= 50,
        # Volume expanding
        latest.get("rvol", 0) >= 1.2,
        # Trend: EMA-8 > EMA-21 > EMA-55
        bool(latest.get("ema8", 0) > latest.get("ema21", 0)
             > latest.get("ema55", 0)),
        # ADX above 20
        latest.get("adx", 0) >= 20,
    ]
    return sum(bool(x) for x in checks)


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calc_risk(price: float, breakout_low: float
              ) -> Tuple[float, float, float, bool]:
    """
    Stop = low of breakout candle.
    Returns (stop, target, rr, risk_ok).
    risk_ok = False if stop is > 3% from entry.
    """
    stop = round(breakout_low, 2)
    risk_pct = (price - stop) / price if price > stop else 1.0
    risk_ok = risk_pct <= RISK_MAX_PCT
    reward = price - stop
    target = round(price + reward * 2, 2) if reward > 0 else price
    rr = round(reward * 2 / reward, 1) if reward > 0 else 0
    return stop, target, rr, risk_ok


# ============================================================
# CORE ENGINE
# ============================================================

@dataclass
class Candidate:
    symbol: str
    status: str        # ACTIONABLE or WATCHLIST
    price: float
    pivot: float
    stop: float
    target: float
    rr: float
    score: int
    notes: str
    details: Dict[str, Any]


def analyze_ticker(sym: str, df: pd.DataFrame) -> Optional[Candidate]:
    """Analyze one stock against the playbook."""
    try:
        if len(df) < MIN_DATA_DAYS:
            return None

        df = add_indicators(normalize_df(df))
        latest = df.iloc[-1]
        price = float(latest["Close"])

        # Price filter
        if price < MIN_PRICE or price > MAX_PRICE:
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

        # Determine status
        if end_offset == 1 and is_breakout_candle(df, pivot):
            # Base ended yesterday, today is breakout
            conf = confluence_score(latest)
            if conf < CONFLUENCE_MIN:
                # Not enough confluence — keep on watchlist
                status = "WATCHLIST"
            else:
                status = "ACTIONABLE"
        elif end_offset == 0 and price <= pivot:
            # Base includes today, stock still resting
            status = "WATCHLIST"
        else:
            return None

        # Risk
        bo_low = float(latest["Low"]) if status == "ACTIONABLE" else 0
        stop, target, rr, risk_ok = calc_risk(price, bo_low)
        conf = confluence_score(latest)

        # Build notes
        parts = []
        parts.append(f"Base:{base_info['base_len']}d")
        if base_info["has_sfp"]:
            parts.append("SFP")
        if base_info["doji_count"] > 0:
            parts.append(f"Doji:{base_info['doji_count']}")
        parts.append(f"Conf:{conf}/6")
        if status == "ACTIONABLE":
            parts.append(f"RVol:{latest['rvol']:.1f}x")
            if not risk_ok:
                parts.append("Risk>3%")

        return Candidate(
            symbol=sym, status=status, price=price,
            pivot=round(pivot, 2), stop=stop, target=target,
            rr=rr, score=conf, notes=", ".join(parts),
            details={
                "Symbol": sym, "Status": status, "Price": price,
                "Pivot": round(pivot, 2), "Stop": stop,
                "Target": target, "RR": rr, "Score": conf,
                "Notes": ", ".join(parts),
                "RSI": float(latest["rsi"]),
                "RVol": float(latest["rvol"]) if not pd.isna(
                    latest["rvol"]) else 0,
                "ADX": float(latest["adx"]) if not pd.isna(
                    latest["adx"]) else 0,
                "UD": float(latest["ud_ratio"]) if not pd.isna(
                    latest["ud_ratio"]) else 0,
                "Breakout": status == "ACTIONABLE",
                "BaseDays": base_info["base_len"],
                "SFP": base_info["has_sfp"],
                "Doji": base_info["doji_count"],
            })
    except Exception as e:
        print(f"  [WARN] {sym}: {e}", file=sys.stderr)
        return None


def scan_universe(symbols: List[str], period: str = "2y",
                  interval: str = "1d") -> pd.DataFrame:
    print(f"  Scanning {len(symbols)} symbols...")
    candidates = []

    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i + CHUNK_SIZE]
        try:
            bulk = yf.download(" ".join(chunk), period=period,
                               interval=interval, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            print(f"  Download failed: {e}")
            continue

        items = []
        is_multi = isinstance(bulk.columns, pd.MultiIndex)
        for sym in chunk:
            try:
                if is_multi:
                    if sym not in bulk.columns.levels[0]:
                        continue
                    sdf = bulk[sym].dropna(how="all").copy()
                else:
                    if bulk.empty:
                        continue
                    sdf = bulk.dropna(how="all").copy()
                if len(sdf) >= MIN_DATA_DAYS:
                    items.append((sym, sdf))
            except Exception:
                continue

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(analyze_ticker, s, d): s
                       for s, d in items}
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc=f"  Chunk {i // CHUNK_SIZE + 1}"):
                try:
                    result = f.result()
                    if result:
                        candidates.append(result)
                except Exception:
                    pass

        del bulk
        gc.collect()

    if not candidates:
        return pd.DataFrame()

    df_out = pd.DataFrame([c.details for c in candidates])
    return (df_out
            .sort_values(["Breakout", "Score", "RVol"],
                         ascending=False)
            .head(15)
            .reset_index(drop=True))


# ============================================================
# ALERT FORMATTING
# ============================================================

def _short(sym: str) -> str:
    return sym.replace(".NS", "")


def fmt_breakout(row) -> str:
    s = _short(row["Symbol"])
    return (
        f"🚀 *{s}* — BREAKOUT ALERT\n"
        f"{'=' * 24}\n"
        f"📍 Entry  : ₹{row['Price']:.0f}\n"
        f"🛡️ Stop   : ₹{row['Stop']:.0f}\n"
        f"🎯 Target : ₹{row['Target']:.0f}\n"
        f"⚖️ R:R    : {row['RR']}:1\n"
        f"📈 RVol   : {row['RVol']:.1f}x\n"
        f"📝 Setup  : {row['Notes']}\n"
        f"{'-' * 24}"
    )


def get_groq_verdict(df: pd.DataFrame, is_eod: bool) -> str:
    if not GROQ_API_KEY or df.empty:
        return ""
        
    wl_text = ""
    for _, r in df.head(10).iterrows():
        s = _short(r["Symbol"])
        dist_to_pivot = ((r['Pivot'] - r['Price']) / r['Price']) * 100 if r['Price'] > 0 else 0
        wl_text += f"{s}: Gap {dist_to_pivot:.1f}%, Setup: {r['Notes']}\n"
        
    time_ctx = "End of Day (building tomorrow's watchlist)" if is_eod else "Morning (looking for immediate breakouts today)"
    prompt = f"""
You are an expert swing trader. I have a watchlist of stocks near their breakout pivot points.
Time of day: {time_ctx}

Here is the watchlist data:
{wl_text}

Provide a very short, punchy 'Final Verdict' (max 3-4 sentences). 
Tell me which 1 or 2 stocks to prioritize based on the tightness of the base (high doji count), perfect confluence (6/6), proximity to pivot (Gap %), and SFP (Swing Failure Pattern).
Point out any stock that can be ignored for now (e.g. gap is too large).
Keep it highly actionable and conversational. Do not use markdown headers, just plain text with maybe some emojis.
"""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.3}
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip()
        return f"\n\n🤖 *Groq Verdict:*\n{verdict}"
    except Exception as e:
        print(f"[GROQ] Error: {e}")
        return ""


def fmt_watchlist(df: pd.DataFrame, title: str = "WATCHLIST", is_eod: bool = False) -> str:
    lines = [f"📋 *{title}*", "=" * 20]
    for _, r in df.head(10).iterrows():
        s = _short(r["Symbol"])
        dist_to_pivot = 0.0
        if r['Price'] > 0:
            dist_to_pivot = ((r['Pivot'] - r['Price']) / r['Price']) * 100
        
        lines.append(f"👀 *{s}*")
        lines.append(f"  • Price : ₹{r['Price']:.0f}")
        lines.append(f"  • Pivot : ₹{r['Pivot']:.0f} (Gap: {dist_to_pivot:.1f}%)")
        lines.append(f"  • Setup : {r['Notes']}")
        lines.append("-" * 20)
        
    base_text = "\n".join(lines)
    verdict = get_groq_verdict(df, is_eod)
    return base_text + verdict


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Momentum Burst Scanner")
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

        # Weekend check
        if now.weekday() >= 5:
            print("Weekend. Exiting.")
            sys.exit(0)

        # Before market
        if now.hour * 100 + now.minute < 910:
            print("Before market hours. Exiting.")
            sys.exit(0)

        # Already ran today
        if state.get("eod_date") == today:
            print(f"Already completed for {today}. Exiting.")
            sys.exit(0)

        # Day reset
        if state.get("alerted_date") != today:
            state["alerted_today"] = []
            state["alerted_date"] = today

        alerted = set(state["alerted_today"])

        print(f"IST: {now.strftime('%Y-%m-%d %H:%M')} | "
              f"Watchlist: {len(state['watchlist'])}")

        # Market regime
        regime_ok, regime = check_market_regime()
        print(f"Market Regime: {regime}")
        if not regime_ok:
            send_telegram(
                f"*Market Regime: {regime}*\n"
                f"No long setups in bear market.")

        # Build universe
        nifty500 = get_nifty500()
        scan_symbols = list(set(
            nifty500 + state.get("watchlist", [])))
        print(f"Universe: {len(scan_symbols)} symbols")

        # Scan
        print("\nSCANNING...")
        results = scan_universe(scan_symbols, args.period,
                                args.interval)

        if not results.empty:
            actionable = results[results["Status"] == "ACTIONABLE"]
            watchlist = results[results["Status"] == "WATCHLIST"]

            # Alert breakouts
            for _, row in actionable.iterrows():
                if row["Symbol"] not in alerted:
                    send_telegram(fmt_breakout(row))
                    alerted.add(row["Symbol"])
                    print(f"  🚀 {row['Symbol']} BREAKOUT "
                          f"₹{row['Price']:.0f}")

            # Update watchlist — REPLACE, don't accumulate
            new_wl = watchlist["Symbol"].tolist() if not watchlist.empty \
                else []
            state["watchlist"] = new_wl

            # Calculate is_eod before formatting
            is_eod = now.hour > 15 or (now.hour == 15 and now.minute >= 28)
            
            # Send watchlist
            if not watchlist.empty:
                send_telegram(fmt_watchlist(watchlist, is_eod=is_eod))

            # Summary
            print(f"\nResults: {len(actionable)} actionable, "
                  f"{len(watchlist)} watchlist")
        else:
            state["watchlist"] = []
            print("\nNo setups found.")
            send_telegram("*Scan Complete* — No clean setups today.")
            is_eod = now.hour > 15 or (now.hour == 15 and now.minute >= 28)

        # Save
        if is_eod:
            state["eod_date"] = today

        state["alerted_today"] = list(alerted)
        save_state(state)
        print(f"\nSaved: {len(state['watchlist'])} watchlist, "
              f"{len(alerted)} alerted")

    except Exception as e:
        print("CRITICAL ERROR:")
        traceback.print_exc()
        try:
            send_telegram(f"*Scanner Crashed*: {str(e)[:200]}")
        except Exception:
            pass
        sys.exit(1)
