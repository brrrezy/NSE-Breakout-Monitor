import argparse
import datetime as dt
import gc
import json
import os
import sys
import tempfile
import traceback
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
# IST TIMEZONE HANDLING
# ============================================================

IST_OFFSET = dt.timedelta(hours=5, minutes=30)


def get_now_ist():
    """Get current time in IST (works on any server timezone)."""
    return dt.datetime.now(dt.timezone.utc) + IST_OFFSET


# ============================================================
# 1) CONFIG & PATHS
# ============================================================

_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "nse_screener_cache"
CACHE_DIR = Path(os.environ.get("NSE_SCREENER_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = Path("watchlist_persistent.json")
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ============================================================
# 2) PERSISTENT STATE (survives between GitHub Action runs)
# ============================================================

def load_state() -> dict:
    """Load watchlist + alerted-today from persistent JSON."""
    default = {"watchlist": [], "alerted_today": [], "alerted_date": ""}
    if not STATE_FILE.exists():
        return default
    try:
        content = STATE_FILE.read_text().strip()
        if not content:
            return default
        data = json.loads(content)
        # Handle old format (plain list of symbols)
        if isinstance(data, list):
            return {**default, "watchlist": data}
        return {**default, **data}
    except Exception:
        return default


def save_state(state: dict):
    """Save state to persistent JSON."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ============================================================
# 3) TELEGRAM
# ============================================================

def send_telegram_message(message: str):
    """Send a message via Telegram bot. Silently fails if keys are missing."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped — no token/chat_id configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[Telegram] Failed: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        print(f"[Telegram] Error: {e}")


# ============================================================
# 4) NSE STOCK UNIVERSE
# ============================================================

def _clean_symbol(raw: str) -> str:
    """Normalize a raw symbol from NSE CSV to Yahoo format."""
    return raw.split(',')[0].strip().strip('"').replace('$', '') + ".NS"


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
            return [_clean_symbol(l) for l in lines[1:] if l.strip()]

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")
        lines = resp.text.splitlines()
        return [_clean_symbol(l) for l in lines[1:] if l.strip()]
    except Exception:
        return []


# ============================================================
# 5) TECHNICAL ENGINE
# ============================================================

def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    return df


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
    df["adrp20"] = ((df["High"] - df["Low"]).rolling(20).mean() / df["Close"]) * 100.0
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
    is_tight = latest["drp"] < latest["adrp20"] * 0.8
    tightness_5d = w["Close"].tail(5).std() / latest["Close"] < 0.015

    # 3. Volume Drying
    vol_dry = w["Volume"].tail(3).mean() < latest["vol_ma50"] * 0.8

    # 4. Volatility Contraction (Behavioral)
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


def detect_ipo_base(df: pd.DataFrame) -> bool:
    """Proxy: stock has < 1 year of trading data. No API call needed."""
    return len(df) < 252


def calculate_rs_raw(df: pd.DataFrame) -> float:
    def get_ret(p):
        return (
            (df["Close"].iloc[-1] / df["Close"].iloc[-p - 1] - 1) if len(df) > p else 0
        )

    return float(0.4 * get_ret(63) + 0.2 * get_ret(126) + 0.4 * get_ret(252))


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
    universe_limit: int = 0,
    min_confluence_score: int = 6,
    period: str = "1y",
    interval: str = "1d",
    top_n: int = 25,
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

    print(f"Scanning {len(stocks)} symbols in chunks...")
    candidates = []
    chunk_size = 50

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

        for sym in tqdm(chunk, desc=f"Chunk {i // chunk_size + 1}"):
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
                is_ipo = detect_ipo_base(df)  # Uses df length — no API call

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
            except Exception:
                counters["errors"] += 1

        del bulk_df
        gc.collect()

    if not candidates:
        return pd.DataFrame()

    df_results = pd.DataFrame([c.details for c in candidates])
    df_results["RSRating"] = (
        pd.Series([c.rs_raw for c in candidates]).rank(pct=True) * 99
    ).astype(int)

    top = (
        df_results.sort_values(by=["Breakout", "VCP_Setup", "Score", "RSRating"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top.attrs["summary"] = counters

    print("\n" + "=" * 50)
    print(f" TOP {top_n} CANDIDATES ")
    print("=" * 50)
    if not top.empty:
        show_cols = ["Symbol", "Status", "Notes", "Price", "Score", "VCP_Setup", "Breakout", "VolMult"]
        print(top[show_cols].to_string(index=False))
    else:
        print("No candidates found.")
    print("=" * 50 + "\n")

    return top


# ============================================================
# 7) MAIN — GitHub Actions Entry Point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="NSE Swing Screener — GitHub Actions Workflow Engine")
    p.add_argument("--universe-limit", type=int, default=0, help="0 = scan all NSE stocks")
    p.add_argument("--min-score", type=int, default=6)
    p.add_argument("--period", type=str, default="1y")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--test-alert", action="store_true", help="Send a test Telegram message and exit")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        if args.test_alert:
            print("Sending test alert to Telegram...")
            send_telegram_message("🔔 *Test Alert*: Your NSE Screener is connected! 🚀")
            print("Done. Check your phone!")
            sys.exit(0)

        # --- 1. Load persistent state ---
        state = load_state()
        today_str = get_now_ist().strftime("%Y-%m-%d")

        # Reset alerted set if it's a new day
        if state.get("alerted_date") != today_str:
            state["alerted_today"] = []
            state["alerted_date"] = today_str

        alerted_set = set(state["alerted_today"])
        priority_symbols = state.get("watchlist", [])
        now_ist = get_now_ist()

        print(f"⏰ Current IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 Watchlist from previous session: {len(priority_symbols)} stocks")
        print(f"🚫 Already alerted today: {len(alerted_set)} stocks")

        send_telegram_message(f"🔄 *Scan Started* — {now_ist.strftime('%H:%M IST')}")

        # --- 2. Morning priority scan (fast, watchlist only) ---
        is_morning = now_ist.hour == 9 and 10 <= now_ist.minute <= 59

        if is_morning and priority_symbols:
            print(f"\n⚡ MORNING PRIORITY: Scanning {len(priority_symbols)} watchlist stocks...")
            send_telegram_message(f"🌅 *Morning Priority*: Checking {len(priority_symbols)} watchlist stocks...")

            priority_results = run_full_system(
                min_confluence_score=args.min_score,
                period=args.period,
                interval=args.interval,
                top_n=len(priority_symbols),
                manual_symbols=priority_symbols,
            )

            if not priority_results.empty:
                for _, row in priority_results.iterrows():
                    if row["Status"] == "ACTIONABLE" and row["Symbol"] not in alerted_set:
                        msg = (
                            f"🔥 *BREAKOUT: {row['Symbol']}*\n"
                            f"💰 Price: ₹{row['Price']:.2f}\n"
                            f"🎯 Pivot: ₹{row['Pivot']:.2f}\n"
                            f"📝 {row['Notes']}"
                        )
                        send_telegram_message(msg)
                        alerted_set.add(row["Symbol"])

                # Send morning watchlist update
                wl_rows = priority_results[priority_results["Status"].isin(["WATCHLIST", "ACTIONABLE"])]
                if not wl_rows.empty:
                    lines = []
                    for _, r in wl_rows.head(15).iterrows():
                        emoji = "🔥" if r["Status"] == "ACTIONABLE" else "👀"
                        lines.append(f"{emoji} {r['Symbol']} — ₹{r['Price']:.0f} | {r['Notes']}")
                    send_telegram_message("📋 *Morning Watchlist Update*\n\n" + "\n".join(lines))

        # --- 3. Full market scan (always runs) ---
        print("\n🔎 FULL MARKET SCAN...")
        results = run_full_system(
            universe_limit=args.universe_limit,
            min_confluence_score=args.min_score,
            period=args.period,
            interval=args.interval,
            top_n=args.top_n,
            start_index=args.start_index,
        )

        if not results.empty:
            actionable = results[results["Status"] == "ACTIONABLE"]
            watchlist = results[results["Status"] == "WATCHLIST"]

            # Send breakout alerts (skip already-alerted)
            for _, row in actionable.iterrows():
                if row["Symbol"] not in alerted_set:
                    msg = (
                        f"🔥 *BREAKOUT: {row['Symbol']}*\n"
                        f"💰 Price: ₹{row['Price']:.2f}\n"
                        f"🎯 Pivot: ₹{row['Pivot']:.2f}\n"
                        f"📝 {row['Notes']}"
                    )
                    send_telegram_message(msg)
                    alerted_set.add(row["Symbol"])

            # --- 4. EOD check (evaluated AFTER scan) ---
            now_ist = get_now_ist()
            is_eod = now_ist.hour >= 15 and now_ist.minute >= 20

            if is_eod:
                print("🏁 END OF DAY — Generating tomorrow's watchlist...")
                eod_symbols = list(dict.fromkeys(
                    watchlist["Symbol"].tolist() + actionable["Symbol"].tolist()
                ))

                state["watchlist"] = eod_symbols

                if eod_symbols:
                    summary = "\n".join([f"• {s}" for s in eod_symbols[:20]])
                    send_telegram_message(
                        f"🏁 *Market Closed — Tomorrow's Watchlist*\n"
                        f"({len(eod_symbols)} stocks)\n\n{summary}"
                    )
                else:
                    send_telegram_message("🏁 *Market Closed* — No clean setups for tomorrow.")
            else:
                # During market hours: keep existing watchlist, just update with new finds
                existing = set(state.get("watchlist", []))
                new_wl = set(watchlist["Symbol"].tolist()) if not watchlist.empty else set()
                state["watchlist"] = list(existing | new_wl)

        # --- 5. Save persistent state ---
        state["alerted_today"] = list(alerted_set)
        save_state(state)
        print(f"\n✅ State saved: {len(state['watchlist'])} watchlist, {len(alerted_set)} alerted")

    except Exception as e:
        print("❌ CRITICAL ERROR IN SCREENER:")
        traceback.print_exc()
        # Try to notify via Telegram
        try:
            send_telegram_message(f"❌ *Screener Crashed*: {str(e)[:200]}")
        except Exception:
            pass
        sys.exit(1)
