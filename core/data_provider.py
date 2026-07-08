"""
Data Provider — Download, cache, normalize OHLCV data.

Handles:
  - Bulk yfinance downloads with retry logic
  - Local parquet caching with configurable TTL
  - Data validation and gap detection
  - MultiIndex normalization
  - Index data (Nifty 50, Nifty Bank, India VIX)
"""

import datetime as dt
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import Settings


# ============================================================
# CACHING
# ============================================================

class DataCache:
    """Simple file-based parquet cache for OHLCV data."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self._dir = cache_dir / "ohlcv"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = dt.timedelta(hours=ttl_hours)

    def _key_path(self, symbol: str, period: str, interval: str) -> Path:
        safe = symbol.replace("^", "_idx_").replace(".", "_")
        return self._dir / f"{safe}_{period}_{interval}.parquet"

    def get(self, symbol: str, period: str, interval: str
            ) -> Optional[pd.DataFrame]:
        p = self._key_path(symbol, period, interval)
        if not p.exists():
            return None
        age = dt.datetime.now() - dt.datetime.fromtimestamp(p.stat().st_mtime)
        if age > self._ttl:
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def put(self, symbol: str, period: str, interval: str,
            df: pd.DataFrame):
        p = self._key_path(symbol, period, interval)
        try:
            df.to_parquet(p, index=True)
        except Exception:
            pass

    def clear(self):
        for f in self._dir.glob("*.parquet"):
            f.unlink(missing_ok=True)


# ============================================================
# NORMALIZER
# ============================================================

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and handle MultiIndex from yfinance."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    # Ensure required columns exist
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns after normalization: {missing}")
    return df


def validate_df(df: pd.DataFrame, min_rows: int = 50) -> bool:
    """Basic data quality checks."""
    if df is None or df.empty:
        return False
    if len(df) < min_rows:
        return False
    # Check for excessive NaN in Close
    close_nan_pct = df["Close"].isna().sum() / len(df)
    if close_nan_pct > 0.1:
        return False
    # Check for zero volume (common yfinance issue)
    zero_vol_pct = (df["Volume"] == 0).sum() / len(df)
    if zero_vol_pct > 0.3:
        return False
    return True


# ============================================================
# DATA PROVIDER
# ============================================================

class DataProvider:
    """
    Central data provider with caching and retry logic.
    Downloads OHLCV data for individual stocks and indices.
    """

    def __init__(self):
        cfg = Settings.get()
        self._cache = DataCache(cfg.cache_dir, cfg.cache_ttl_hours)
        self._period = cfg.data_period
        self._interval = cfg.data_interval
        self._chunk_size = cfg.chunk_size
        self._max_retries = 3
        self._retry_delay = 2.0

    # ── Single ticker ──

    def get_ticker_data(self, symbol: str,
                        period: str = None,
                        interval: str = None) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a single ticker, with caching."""
        period = period or self._period
        interval = interval or self._interval

        # Check cache
        cached = self._cache.get(symbol, period, interval)
        if cached is not None and validate_df(cached):
            return cached

        # Download with retry
        for attempt in range(self._max_retries):
            try:
                df = yf.download(
                    symbol, period=period, interval=interval,
                    progress=False, threads=False)
                if df.empty:
                    return None
                df = normalize_df(df)
                if validate_df(df):
                    self._cache.put(symbol, period, interval, df)
                    return df
                return None
            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    print(f"[DATA] Failed to download {symbol}: {e}",
                          file=sys.stderr)
                    return None
        return None

    # ── Bulk download ──

    def get_bulk_data(self, symbols: List[str],
                      period: str = None,
                      interval: str = None
                      ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV for many tickers in chunks.
        Returns dict of {symbol: DataFrame}.
        Uses yfinance bulk download for speed, falls back to
        individual download for failures.
        """
        period = period or self._period
        interval = interval or self._interval
        result: Dict[str, pd.DataFrame] = {}

        # First check cache for all
        uncached = []
        for sym in symbols:
            cached = self._cache.get(sym, period, interval)
            if cached is not None and validate_df(cached):
                result[sym] = cached
            else:
                uncached.append(sym)

        if not uncached:
            return result

        # Bulk download uncached in chunks
        for i in range(0, len(uncached), self._chunk_size):
            chunk = uncached[i:i + self._chunk_size]
            chunk_result = self._download_chunk(
                chunk, period, interval)
            for sym, df in chunk_result.items():
                result[sym] = df
                self._cache.put(sym, period, interval, df)
            gc.collect()

        return result

    def _download_chunk(self, symbols: List[str],
                        period: str, interval: str
                        ) -> Dict[str, pd.DataFrame]:
        """Download a chunk of symbols via yfinance bulk API."""
        result = {}

        for attempt in range(self._max_retries):
            try:
                bulk = yf.download(
                    " ".join(symbols),
                    period=period, interval=interval,
                    group_by="ticker", threads=True,
                    progress=False)
                break
            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    print(f"[DATA] Chunk download failed: {e}",
                          file=sys.stderr)
                    return result
        else:
            return result

        is_multi = isinstance(bulk.columns, pd.MultiIndex)

        for sym in symbols:
            try:
                if is_multi:
                    if sym not in bulk.columns.get_level_values(0):
                        continue
                    sdf = bulk[sym].dropna(how="all").copy()
                else:
                    if bulk.empty:
                        continue
                    sdf = bulk.dropna(how="all").copy()

                sdf = normalize_df(sdf)
                if validate_df(sdf, min_rows=Settings.get().min_data_days):
                    result[sym] = sdf
            except Exception:
                continue

        return result

    # ── Index data ──

    _index_cache: Dict[str, pd.DataFrame] = {}

    def get_index_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get index data (cached in memory for the session)."""
        if symbol in self._index_cache:
            return self._index_cache[symbol]

        df = self.get_ticker_data(symbol, period="1y")
        if df is not None:
            self._index_cache[symbol] = df
        return df

    def get_nifty50(self) -> Optional[pd.DataFrame]:
        return self.get_index_data(Settings.get().nifty_symbol)

    def get_india_vix(self) -> Optional[pd.DataFrame]:
        return self.get_index_data(Settings.get().india_vix_symbol)

    def get_nifty_bank(self) -> Optional[pd.DataFrame]:
        return self.get_index_data(Settings.get().nifty_bank_symbol)

    def clear_cache(self):
        self._cache.clear()
        self._index_cache.clear()
