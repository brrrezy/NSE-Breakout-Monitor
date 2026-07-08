"""
Stock Universe Management.

Handles:
  - Nifty 500 list fetching from NSE
  - Sector/Industry mapping
  - Watchlist merging and persistence
"""

import datetime as dt
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config.settings import Settings


# ============================================================
# NSE UNIVERSE
# ============================================================

_NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
)


def _fetch_cached_csv(url: str, cache_path: Path,
                      ttl_hours: int = 24) -> List[str]:
    """Fetch a CSV from URL with file-based caching."""
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
    except Exception as e:
        print(f"[UNIVERSE] Fetch failed ({url}): {e}", file=sys.stderr)
        if cache_path.exists():
            return cache_path.read_text(
                encoding="utf-8", errors="ignore").splitlines()
        return []


class Universe:
    """Manages the stock universe, sector mappings, and watchlist."""

    def __init__(self):
        cfg = Settings.get()
        self._cache_dir = cfg.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._nifty500_cache = self._cache_dir / "nifty500.csv"
        self._sector_map: Dict[str, str] = {}
        self._industry_map: Dict[str, str] = {}

    def get_nifty500(self) -> List[str]:
        """
        Fetch Nifty 500 symbols with .NS suffix.
        Also populates sector and industry maps.
        """
        lines = _fetch_cached_csv(
            _NIFTY500_URL, self._nifty500_cache,
            ttl_hours=Settings.get().cache_ttl_hours)
        if not lines:
            return []
        try:
            df = pd.read_csv(StringIO("\n".join(lines)))
            sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
            symbols = []
            for _, row in df.iterrows():
                raw = str(row[sym_col]).strip()
                if raw and not raw.startswith("DUMMY"):
                    sym = raw + ".NS"
                    symbols.append(sym)
                    # Map sectors
                    if "Industry" in df.columns:
                        self._industry_map[sym] = str(
                            row.get("Industry", "")).strip()
                    if "Sector" in df.columns:
                        self._sector_map[sym] = str(
                            row.get("Sector", "")).strip()
                    # Fallback: use Industry as sector if Sector col missing
                    elif "Industry" in df.columns:
                        self._sector_map[sym] = str(
                            row.get("Industry", "")).strip()
            return symbols
        except Exception as e:
            print(f"[UNIVERSE] Parse failed: {e}", file=sys.stderr)
            return []

    def get_sector(self, symbol: str) -> str:
        """Get the sector for a symbol. Returns '' if unknown."""
        return self._sector_map.get(symbol, "")

    def get_industry(self, symbol: str) -> str:
        """Get the industry for a symbol. Returns '' if unknown."""
        return self._industry_map.get(symbol, "")

    def get_sector_peers(self, symbol: str) -> List[str]:
        """Get all symbols in the same sector."""
        sector = self.get_sector(symbol)
        if not sector:
            return []
        return [s for s, sec in self._sector_map.items()
                if sec == sector and s != symbol]


# ============================================================
# STATE PERSISTENCE
# ============================================================

class StateManager:
    """Manages persistent state (watchlist, alerts, EOD tracking)."""

    _DEFAULT = {
        "watchlist": [],
        "alerted_today": [],
        "alerted_date": "",
        "eod_date": "",
    }

    def __init__(self):
        self._path = Settings.get().state_file
        self._state = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return dict(self._DEFAULT)
        try:
            raw = self._path.read_text().strip() or "{}"
            data = json.loads(raw)
            if isinstance(data, list):
                return {**self._DEFAULT, "watchlist": data}
            return {**self._DEFAULT, **data}
        except Exception:
            return dict(self._DEFAULT)

    def save(self):
        self._path.write_text(json.dumps(self._state, indent=2))

    @property
    def watchlist(self) -> List[str]:
        return self._state.get("watchlist", [])

    @watchlist.setter
    def watchlist(self, symbols: List[str]):
        self._state["watchlist"] = symbols

    @property
    def alerted_today(self) -> set:
        return set(self._state.get("alerted_today", []))

    def add_alerted(self, symbol: str):
        alerted = list(self.alerted_today)
        if symbol not in alerted:
            alerted.append(symbol)
        self._state["alerted_today"] = alerted

    def reset_day(self, today: str):
        if self._state.get("alerted_date") != today:
            self._state["alerted_today"] = []
            self._state["alerted_date"] = today

    @property
    def eod_date(self) -> str:
        return self._state.get("eod_date", "")

    @eod_date.setter
    def eod_date(self, date_str: str):
        self._state["eod_date"] = date_str

    def to_dict(self) -> dict:
        return dict(self._state)
