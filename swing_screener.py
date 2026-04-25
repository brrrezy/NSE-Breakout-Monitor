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
                    (symbol,)
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
                    (symbol, json.dumps(data), dt.datetime.now().isoformat())
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
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        if age.total_seconds() < cache_ttl_hours * 3600:
            text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            return [f"{l.split(',')[0].strip().strip('\"')}.NS" for l in lines[1:] if l.strip()]

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")
        lines = resp.text.splitlines()
        return [f"{l.split(',')[0].strip().strip('\"')}.NS" for l in lines[1:] if l.strip()]
    except Exception:
        return []

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
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()

    # Volatility
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["adrp20"] = ((df["High"] - df["Low"]).rolling(20).mean() / df["Close"]) * 100.0

    # CLV / Delta
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["delta_proxy"] = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range) * df["Volume"]

    return df

def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, bool]]:
    close, ema50, ema200 = latest.get("Close"), latest.get("ema50"), latest.get("ema200")
    trend_ok = bool(close > ema50 > ema200) if not pd.isna(ema200) else bool(close > ema50)

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
    if len(df) < 200: return False
    c, e50, e150, e200 = df["Close"].iloc[-1], df["ema50"].iloc[-1], df["ema150"].iloc[-1], df["ema200"].iloc[-1]
    h52, l52 = df["High"].tail(252).max(), df["Low"].tail(252).min()
    e200_1m = df["ema200"].iloc[-20] if len(df) >= 20 else e200

    conds = [
        c > e150 > e200, e200 > e200_1m, e50 > e150, e50 > e200,
        c > e50, c >= l52 * 1.3, c >= h52 * 0.75
    ]
    return all(conds)

def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> bool:
    if len(df) < lookback + 5: return False
    w = df.tail(lookback).copy()
    w["range%"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1, r2, r3 = w["range%"].iloc[:20].mean(), w["range%"].iloc[20:40].mean(), w["range%"].iloc[40:60].mean()
    v1, v2, v3 = w["Volume"].iloc[:20].mean(), w["Volume"].iloc[20:40].mean(), w["Volume"].iloc[40:60].mean()
    return bool((r1 > r2 > r3) and (r3 < 6.0) and (v1 > v2 > v3))

def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 2: return False
    swing_low = float(df.iloc[-(lookback + 1) : -1]["Low"].min())
    last = df.iloc[-1]
    return bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))

def detect_ipo_base(symbol: str) -> bool:
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade: return False
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        return (dt.date.today() - ipo_date).days <= 365
    except: return False

def calculate_rs_raw(df: pd.DataFrame) -> float:
    def get_ret(p):
        return (df["Close"].iloc[-1] / df["Close"].iloc[-p-1] - 1) if len(df) > p else 0
    return float(0.4*get_ret(63) + 0.2*get_ret(126) + 0.4*get_ret(252))

# ============================================================
# 5) FUNDAMENTALS (CACHED & PARALLEL)
# ============================================================

def fetch_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    cached = db_cache.get(symbol)
    if cached: return cached

    import time
    import random

    # Add a tiny random delay to avoid "bot-like" behavior patterns
    time.sleep(random.uniform(0.1, 0.5))

    for attempt in range(3): # 3 retries with backoff
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
                "FundamentalQualityScore": 0
            }
            
            q_score = 0
            if (res["ROE"] or 0) >= 0.15: q_score += 1
            if (res["DebtToEquity"] or 99) <= 1.0: q_score += 1
            if (res["OperatingMargin"] or 0) >= 0.12: q_score += 1
            if (res["RevenueGrowth"] or 0) >= 0.10: q_score += 1
            res["FundamentalQualityScore"] = q_score
            
            db_cache.set(symbol, res)
            return res
        except Exception as e:
            if "Rate Limit" in str(e) or "401" in str(e):
                time.sleep(attempt * 2 + 1) # Exponential backoff
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
    period: str = "2y",
    interval: str = "1d",
    top_n: int = 10,
    start_index: int = 0,
    manual_symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    if manual_symbols:
        stocks = [s.upper().strip() for s in manual_symbols if s.strip()]
    else:
        all_stocks = get_nse_stocks()
        stocks = all_stocks[start_index : start_index + (universe_limit or len(all_stocks))]
    
    counters = {"total_universe": len(stocks), "scanned": 0, "passed_filter": 0, "errors": 0}

    print(f"Hyper-Scanning {len(stocks)} symbols...")
    try:
        # Use a higher thread count for the bulk download (it is generally safer)
        bulk_df = yf.download(tickers=" ".join(stocks), period=period, interval=interval, group_by="ticker", threads=True, progress=False)
    except:
        bulk_df = pd.DataFrame()

    candidates = []
    for sym in tqdm(stocks, desc="Analyzing"):
        counters["scanned"] += 1
        try:
            if sym not in bulk_df.columns.levels[0]: continue
            df = bulk_df[sym].dropna(how="all").copy()
            if len(df) < 90: continue

            df = add_technical_indicators(normalize_ohlcv_df(df))
            latest = df.iloc[-1]
            conf_score, conf_sigs = predicta_v4_confluence(latest)
            
            # If manual search, bypass the score filter so user can see the data
            if not manual_symbols and conf_score < min_confluence_score:
                continue

            counters["passed_filter"] += 1
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)
            ipo = detect_ipo_base(sym)
            minervini = detect_minervini_trend(df)
            rs_raw = calculate_rs_raw(df)
            
            setup_score = (2 if vcp else 0) + (2 if sfp else 0) + (1 if ipo else 0) + (1 if minervini else 0)
            row = {
                "Symbol": sym, "Score": conf_score + setup_score,
                "ConfluenceScore": conf_score, "SetupScore": setup_score,
                "Minervini": minervini, "VCP": vcp, "SFP": sfp, "IPO_BASE": ipo,
                "Price": float(latest["Close"]),
                "RSI": float(latest["rsi"]), "VolMult": float(latest["vol_mult"]),
                "ADR%": float(latest["adrp20"]), "RVol": float(latest["rvol50"]),
                **{f"C_{k}": v for k, v in conf_sigs.items()}
            }
            candidates.append(Candidate(sym, row["Score"], row["Price"], rs_raw, row))
        except:
            counters["errors"] += 1

    if not candidates: return pd.DataFrame()

    # Parallel Fundamentals for finalists (Reduced workers to 5 for stability)
    candidates.sort(key=lambda x: x.score, reverse=True)
    finalists = candidates[:50]
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_results = list(executor.map(lambda c: fetch_stock_fundamentals(c.symbol), finalists))
    
    for c, f in zip(finalists, f_results):
        c.details.update(f)
        c.score += f.get("FundamentalQualityScore", 0)

    df_results = pd.DataFrame([c.details for c in finalists])
    df_results["RSRating"] = (pd.Series([c.rs_raw for c in finalists]).rank(pct=True) * 99).astype(int)
    
    top = df_results.sort_values(by=["Score", "RSRating"], ascending=False).head(top_n).reset_index(drop=True)
    top.attrs["summary"] = counters
    
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
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_full_system(args.universe_limit, args.min_score, args.period, args.interval, args.top_n, args.start_index)
