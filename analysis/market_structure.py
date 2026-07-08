"""
Market Structure Analysis.

Detects:
  - Swing highs and swing lows (fractal-based)
  - Higher Highs / Higher Lows / Lower Highs / Lower Lows
  - Trend classification (uptrend, downtrend, range)
  - Trend transitions
  - Trendline fitting and scoring
  - Breakout pressure measurement
  - Compression / range expansion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# SWING POINT DETECTION
# ============================================================

@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    index: int              # Index into the DataFrame
    price: float
    type: str               # "high" or "low"
    strength: int = 1       # How many bars on each side confirmed this swing


def detect_swing_points(df: pd.DataFrame,
                        lookback: int = 5) -> List[SwingPoint]:
    """
    Detect swing highs and lows using fractal method.
    A swing high is the highest high with 'lookback' lower highs
    on each side. Same logic for lows.
    """
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    points: List[SwingPoint] = []

    for i in range(lookback, n - lookback):
        # Swing High
        is_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            points.append(SwingPoint(
                index=i, price=float(highs[i]),
                type="high", strength=lookback))

        # Swing Low
        is_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            points.append(SwingPoint(
                index=i, price=float(lows[i]),
                type="low", strength=lookback))

    return sorted(points, key=lambda p: p.index)


# ============================================================
# HH/HL/LH/LL CLASSIFICATION
# ============================================================

@dataclass
class StructurePoint:
    """A classified swing point with HH/HL/LH/LL label."""
    index: int
    price: float
    type: str           # "high" or "low"
    label: str          # "HH", "HL", "LH", "LL"


def classify_structure(swings: List[SwingPoint]) -> List[StructurePoint]:
    """
    Classify swing points as Higher High (HH), Higher Low (HL),
    Lower High (LH), or Lower Low (LL).
    """
    if len(swings) < 2:
        return []

    result: List[StructurePoint] = []
    prev_highs = []
    prev_lows = []

    for sp in swings:
        if sp.type == "high":
            if prev_highs:
                label = "HH" if sp.price > prev_highs[-1] else "LH"
            else:
                label = "HH"
            prev_highs.append(sp.price)
            result.append(StructurePoint(
                sp.index, sp.price, sp.type, label))
        else:  # low
            if prev_lows:
                label = "HL" if sp.price > prev_lows[-1] else "LL"
            else:
                label = "HL"
            prev_lows.append(sp.price)
            result.append(StructurePoint(
                sp.index, sp.price, sp.type, label))

    return result


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(structure_points: List[StructurePoint],
                   lookback: int = 6) -> str:
    """
    Classify the current trend based on recent structure points.

    Returns: "uptrend", "downtrend", "range", "transition_up",
             "transition_down", or "unknown"
    """
    if len(structure_points) < 4:
        return "unknown"

    recent = structure_points[-lookback:]
    high_labels = [p.label for p in recent if p.type == "high"]
    low_labels = [p.label for p in recent if p.type == "low"]

    hh_count = high_labels.count("HH")
    lh_count = high_labels.count("LH")
    hl_count = low_labels.count("HL")
    ll_count = low_labels.count("LL")

    total = len(high_labels) + len(low_labels)
    if total == 0:
        return "unknown"

    bullish = (hh_count + hl_count) / total
    bearish = (lh_count + ll_count) / total

    if bullish >= 0.7:
        return "uptrend"
    elif bearish >= 0.7:
        return "downtrend"
    elif bullish >= 0.5 and bearish >= 0.3:
        return "transition_down"
    elif bearish >= 0.5 and bullish >= 0.3:
        return "transition_up"
    else:
        return "range"


# ============================================================
# TRENDLINE FITTING
# ============================================================

@dataclass
class Trendline:
    """A fitted trendline connecting swing points."""
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    slope: float                # Price change per bar
    type: str                   # "support" or "resistance"
    touches: int = 0           # Number of times price touched this line
    strength: float = 0.0      # Composite strength score (0-1)

    def price_at(self, index: int) -> float:
        """Get the trendline price at a given bar index."""
        return self.start_price + self.slope * (index - self.start_index)


def fit_trendlines(df: pd.DataFrame,
                   swings: List[SwingPoint],
                   min_touches: int = 2,
                   tolerance_pct: float = 0.01
                   ) -> List[Trendline]:
    """
    Fit trendlines by connecting swing points.
    Support: connect swing lows.
    Resistance: connect swing highs.
    """
    result: List[Trendline] = []
    n = len(df)

    # Separate highs and lows
    swing_highs = [s for s in swings if s.type == "high"]
    swing_lows = [s for s in swings if s.type == "low"]

    # Fit resistance trendlines (connect highs)
    result.extend(_fit_lines(
        df, swing_highs, "resistance", n, min_touches, tolerance_pct))

    # Fit support trendlines (connect lows)
    result.extend(_fit_lines(
        df, swing_lows, "support", n, min_touches, tolerance_pct))

    return sorted(result, key=lambda t: t.strength, reverse=True)


def _fit_lines(df: pd.DataFrame,
               points: List[SwingPoint],
               line_type: str,
               n: int,
               min_touches: int,
               tolerance_pct: float) -> List[Trendline]:
    """Fit trendlines through a set of swing points."""
    lines: List[Trendline] = []
    prices = df["High"].values if line_type == "resistance" else \
        df["Low"].values

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            if p2.index == p1.index:
                continue

            slope = (p2.price - p1.price) / (p2.index - p1.index)

            # Count touches (bars that come within tolerance of the line)
            touches = 0
            for k in range(p1.index, min(p2.index + 1, n)):
                line_price = p1.price + slope * (k - p1.index)
                tol = line_price * tolerance_pct
                if abs(prices[k] - line_price) <= tol:
                    touches += 1

            if touches >= min_touches:
                # Strength: touches, time span, recent relevance
                span = p2.index - p1.index
                recency = 1.0 - (n - p2.index) / max(n, 1)
                strength = (
                    0.4 * min(touches / 5, 1.0) +
                    0.3 * min(span / 60, 1.0) +
                    0.3 * recency
                )
                lines.append(Trendline(
                    start_index=p1.index,
                    end_index=p2.index,
                    start_price=p1.price,
                    end_price=p2.price,
                    slope=slope,
                    type=line_type,
                    touches=touches,
                    strength=strength,
                ))

    # Deduplicate: keep strongest per approximate slope bucket
    if not lines:
        return lines

    # Sort by strength, keep top 5 per type
    lines.sort(key=lambda t: t.strength, reverse=True)
    return lines[:5]


# ============================================================
# BREAKOUT PRESSURE
# ============================================================

def measure_breakout_pressure(df: pd.DataFrame,
                              resistance_lines: List[Trendline]
                              ) -> dict:
    """
    Measure how close price is to the nearest resistance trendline.
    Returns metrics about breakout pressure.
    """
    if not resistance_lines or df.empty:
        return {"pressure": 0.0, "nearest_resistance": None,
                "gap_pct": 1.0}

    current_price = float(df["Close"].iloc[-1])
    current_idx = len(df) - 1

    best_line = None
    min_gap = float("inf")

    for tl in resistance_lines:
        res_price = tl.price_at(current_idx)
        if res_price > 0:
            gap = (res_price - current_price) / current_price
            if 0 <= gap < min_gap:
                min_gap = gap
                best_line = tl

    if best_line is None:
        return {"pressure": 0.0, "nearest_resistance": None,
                "gap_pct": 1.0}

    # Pressure: closer to resistance = higher pressure
    pressure = max(0, 1.0 - min_gap * 20)  # 5% gap → 0 pressure

    return {
        "pressure": round(pressure, 3),
        "nearest_resistance": best_line.price_at(current_idx),
        "gap_pct": round(min_gap, 4),
        "trendline_strength": best_line.strength,
        "trendline_touches": best_line.touches,
    }


# ============================================================
# COMPRESSION DETECTION
# ============================================================

def detect_compression(df: pd.DataFrame,
                       lookback: int = 20) -> dict:
    """
    Detect converging trendlines (price compression).
    Measures how much the trading range is shrinking.
    """
    if len(df) < lookback:
        return {"compressed": False, "compression_ratio": 1.0}

    recent = df.iloc[-lookback:]
    ranges = (recent["High"] - recent["Low"]).values

    first_half = ranges[:lookback // 2].mean()
    second_half = ranges[lookback // 2:].mean()

    if first_half <= 0:
        return {"compressed": False, "compression_ratio": 1.0}

    ratio = second_half / first_half
    compressed = ratio < 0.65

    return {
        "compressed": compressed,
        "compression_ratio": round(ratio, 3),
        "first_half_range": round(first_half, 2),
        "second_half_range": round(second_half, 2),
    }


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_market_structure(df: pd.DataFrame,
                             swing_lookback: int = 5
                             ) -> dict:
    """
    Complete market structure analysis for one stock.
    Returns a dict with all structure metrics.
    """
    swings = detect_swing_points(df, lookback=swing_lookback)
    structure = classify_structure(swings)
    trend = classify_trend(structure)
    trendlines = fit_trendlines(df, swings)

    resistance_lines = [t for t in trendlines if t.type == "resistance"]
    support_lines = [t for t in trendlines if t.type == "support"]

    pressure = measure_breakout_pressure(df, resistance_lines)
    compression = detect_compression(df)

    # Recent structure summary
    recent_highs = [p for p in structure[-6:] if p.type == "high"]
    recent_lows = [p for p in structure[-6:] if p.type == "low"]

    return {
        "trend": trend,
        "swing_count": len(swings),
        "structure_points": len(structure),
        "recent_high_labels": [p.label for p in recent_highs],
        "recent_low_labels": [p.label for p in recent_lows],
        "resistance_trendlines": len(resistance_lines),
        "support_trendlines": len(support_lines),
        "strongest_resistance": (
            resistance_lines[0].strength if resistance_lines else 0),
        "strongest_support": (
            support_lines[0].strength if support_lines else 0),
        "breakout_pressure": pressure["pressure"],
        "nearest_resistance": pressure.get("nearest_resistance"),
        "gap_to_resistance_pct": pressure.get("gap_pct", 1.0),
        "compressed": compression["compressed"],
        "compression_ratio": compression["compression_ratio"],
        # Pass raw data for downstream use
        "_swings": swings,
        "_structure": structure,
        "_trendlines": trendlines,
    }
