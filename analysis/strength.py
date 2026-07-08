"""
Relative Strength & Sector Analysis.

Computes:
  - RS vs Nifty 50 (multi-timeframe)
  - RS vs Sector peers
  - Beta calculation
  - Sector rotation stage
  - Leadership ranking
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from core.data_provider import DataProvider


# ============================================================
# RELATIVE STRENGTH
# ============================================================

def compute_rs_vs_index(stock_df: pd.DataFrame,
                        index_df: Optional[pd.DataFrame],
                        periods: List[int] = None
                        ) -> Dict[str, float]:
    """
    Compute relative strength of stock vs an index over multiple
    timeframes. Returns RS for each period.
    """
    if periods is None:
        periods = [20, 50, 100, 200]

    result = {}
    if index_df is None or stock_df.empty:
        for p in periods:
            result[f"rs_{p}d"] = 0.0
        result["rs_composite"] = 0.0
        return result

    stock_close = stock_df["Close"]
    # Align index to stock dates
    idx_close = index_df["Close"].reindex(stock_close.index, method="ffill")

    for p in periods:
        if len(stock_close) >= p and len(idx_close) >= p:
            stock_perf = float(
                stock_close.iloc[-1] / stock_close.iloc[-p] - 1)
            idx_perf = float(
                idx_close.iloc[-1] / idx_close.iloc[-p] - 1)
            result[f"rs_{p}d"] = round(stock_perf - idx_perf, 4)
        else:
            result[f"rs_{p}d"] = 0.0

    # Composite RS: weighted average (more weight to recent)
    weights = {20: 0.4, 50: 0.3, 100: 0.2, 200: 0.1}
    composite = sum(
        result.get(f"rs_{p}d", 0) * weights.get(p, 0.25)
        for p in periods if p in weights
    )
    result["rs_composite"] = round(composite, 4)
    return result


# ============================================================
# BETA CALCULATION
# ============================================================

def compute_beta(stock_df: pd.DataFrame,
                 index_df: Optional[pd.DataFrame],
                 period: int = 60) -> float:
    """
    Compute beta of the stock vs the index.
    Beta > 1 = more volatile than index.
    Beta < 1 = less volatile.
    """
    if index_df is None or len(stock_df) < period:
        return 1.0

    stock_ret = stock_df["Close"].pct_change().iloc[-period:]
    idx_close = index_df["Close"].reindex(
        stock_df.index, method="ffill")
    idx_ret = idx_close.pct_change().iloc[-period:]

    # Align
    aligned = pd.DataFrame({
        "stock": stock_ret, "index": idx_ret
    }).dropna()

    if len(aligned) < 20:
        return 1.0

    cov = aligned["stock"].cov(aligned["index"])
    var = aligned["index"].var()

    if var <= 0:
        return 1.0

    return round(cov / var, 3)


# ============================================================
# SECTOR ROTATION
# ============================================================

def analyze_sector_rotation(sector_perfs: Dict[str, float]
                            ) -> Dict[str, str]:
    """
    Classify sector stage based on relative performance.

    Leading: top quartile performance
    Weakening: was leading, now declining
    Lagging: bottom quartile
    Improving: was lagging, now rising

    Returns dict of {sector: stage}.
    """
    if not sector_perfs:
        return {}

    sorted_sectors = sorted(
        sector_perfs.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_sectors)
    q1 = n // 4

    result = {}
    for i, (sector, perf) in enumerate(sorted_sectors):
        if i < q1:
            result[sector] = "leading"
        elif i < n // 2:
            result[sector] = "improving" if perf > 0 else "weakening"
        elif i < 3 * q1:
            result[sector] = "weakening" if perf > 0 else "lagging"
        else:
            result[sector] = "lagging"

    return result


# ============================================================
# LEADERSHIP RANKING
# ============================================================

def rank_by_strength(stocks_rs: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Rank stocks by their composite relative strength.
    Returns sorted list of (symbol, rs_score).
    """
    return sorted(stocks_rs.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# FULL STRENGTH ANALYSIS
# ============================================================

def analyze_strength(stock_df: pd.DataFrame,
                     index_df: Optional[pd.DataFrame],
                     sector: str = "",
                     ) -> Dict:
    """
    Complete strength analysis for one stock.
    """
    rs = compute_rs_vs_index(stock_df, index_df)
    beta = compute_beta(stock_df, index_df)

    return {
        "rs_20d": rs.get("rs_20d", 0),
        "rs_50d": rs.get("rs_50d", 0),
        "rs_100d": rs.get("rs_100d", 0),
        "rs_200d": rs.get("rs_200d", 0),
        "rs_composite": rs.get("rs_composite", 0),
        "beta": beta,
        "sector": sector,
        "outperforming": rs.get("rs_composite", 0) > 0,
    }
