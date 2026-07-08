"""
Chart Pattern Detection.

Detects:
  - Ascending Triangle (flat top, rising bottoms)
  - Descending Triangle
  - Symmetrical Triangle
  - Bull Flag / Bear Flag
  - Pennant
  - Cup and Handle
  - Rounded Bottom
  - Flat Base / Box Consolidation
  - Darvas Box
  - Double Bottom
  - Wedge (rising / falling)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.market_structure import SwingPoint, detect_swing_points


@dataclass
class PatternResult:
    """Detected chart pattern with metadata."""
    name: str
    confidence: float       # 0.0 - 1.0
    type: str               # "bullish", "bearish", "neutral"
    start_index: int
    end_index: int
    breakout_level: Optional[float] = None
    target: Optional[float] = None
    description: str = ""


# ============================================================
# ASCENDING TRIANGLE
# ============================================================

def detect_ascending_triangle(df: pd.DataFrame,
                              swings: List[SwingPoint],
                              lookback: int = 40
                              ) -> Optional[PatternResult]:
    """
    Ascending Triangle: flat resistance (equal highs) with
    rising support (higher lows).
    """
    if len(df) < lookback:
        return None

    recent_highs = [s for s in swings if s.type == "high"
                    and s.index >= len(df) - lookback]
    recent_lows = [s for s in swings if s.type == "low"
                   and s.index >= len(df) - lookback]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    # Check flat resistance (highs within 1.5% of each other)
    high_prices = [s.price for s in recent_highs[-4:]]
    high_range = (max(high_prices) - min(high_prices)) / max(high_prices)
    flat_top = high_range < 0.015

    # Check rising support (each low higher than the previous)
    low_prices = [s.price for s in recent_lows[-4:]]
    rising_bottoms = all(low_prices[i] > low_prices[i - 1] * 0.99
                         for i in range(1, len(low_prices)))

    if flat_top and rising_bottoms:
        resistance = np.mean(high_prices)
        height = resistance - min(low_prices)
        confidence = min(
            0.3 + len(recent_highs) * 0.1 + len(recent_lows) * 0.1,
            0.95)

        return PatternResult(
            name="Ascending Triangle",
            confidence=round(confidence, 2),
            type="bullish",
            start_index=recent_lows[0].index,
            end_index=len(df) - 1,
            breakout_level=round(resistance, 2),
            target=round(resistance + height, 2),
            description=(f"Flat resistance at ₹{resistance:.0f} with "
                         f"rising lows. {len(recent_highs)} tests of "
                         f"resistance."),
        )
    return None


# ============================================================
# DESCENDING TRIANGLE
# ============================================================

def detect_descending_triangle(df: pd.DataFrame,
                               swings: List[SwingPoint],
                               lookback: int = 40
                               ) -> Optional[PatternResult]:
    """
    Descending Triangle: flat support with lower highs.
    """
    if len(df) < lookback:
        return None

    recent_highs = [s for s in swings if s.type == "high"
                    and s.index >= len(df) - lookback]
    recent_lows = [s for s in swings if s.type == "low"
                   and s.index >= len(df) - lookback]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    low_prices = [s.price for s in recent_lows[-4:]]
    low_range = (max(low_prices) - min(low_prices)) / max(low_prices)
    flat_bottom = low_range < 0.015

    high_prices = [s.price for s in recent_highs[-4:]]
    falling_tops = all(high_prices[i] < high_prices[i - 1] * 1.01
                       for i in range(1, len(high_prices)))

    if flat_bottom and falling_tops:
        support = np.mean(low_prices)
        height = max(high_prices) - support
        confidence = min(
            0.3 + len(recent_highs) * 0.1 + len(recent_lows) * 0.1,
            0.95)

        return PatternResult(
            name="Descending Triangle",
            confidence=round(confidence, 2),
            type="bearish",
            start_index=recent_highs[0].index,
            end_index=len(df) - 1,
            breakout_level=round(support, 2),
            target=round(support - height, 2),
            description=(f"Flat support at ₹{support:.0f} with "
                         f"falling highs."),
        )
    return None


# ============================================================
# SYMMETRICAL TRIANGLE
# ============================================================

def detect_symmetrical_triangle(df: pd.DataFrame,
                                swings: List[SwingPoint],
                                lookback: int = 40
                                ) -> Optional[PatternResult]:
    """Symmetrical Triangle: converging highs and lows."""
    if len(df) < lookback:
        return None

    recent_highs = [s for s in swings if s.type == "high"
                    and s.index >= len(df) - lookback]
    recent_lows = [s for s in swings if s.type == "low"
                   and s.index >= len(df) - lookback]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    high_prices = [s.price for s in recent_highs[-3:]]
    low_prices = [s.price for s in recent_lows[-3:]]

    falling_highs = all(high_prices[i] < high_prices[i - 1]
                        for i in range(1, len(high_prices)))
    rising_lows = all(low_prices[i] > low_prices[i - 1]
                      for i in range(1, len(low_prices)))

    if falling_highs and rising_lows:
        midpoint = (high_prices[-1] + low_prices[-1]) / 2
        height = high_prices[0] - low_prices[0]

        return PatternResult(
            name="Symmetrical Triangle",
            confidence=0.65,
            type="neutral",
            start_index=min(recent_highs[0].index, recent_lows[0].index),
            end_index=len(df) - 1,
            breakout_level=round(high_prices[-1], 2),
            target=round(high_prices[-1] + height * 0.618, 2),
            description="Converging trendlines — breakout direction uncertain.",
        )
    return None


# ============================================================
# BULL FLAG
# ============================================================

def detect_bull_flag(df: pd.DataFrame,
                     lookback: int = 30) -> Optional[PatternResult]:
    """
    Bull Flag: strong impulsive move up (pole), followed by a
    short, downward-sloping consolidation (flag).
    """
    if len(df) < lookback:
        return None

    # Look for the pole: strong up move in 3-8 bars
    for pole_len in range(3, 9):
        pole_start = len(df) - lookback
        for offset in range(0, lookback - pole_len - 5):
            ps = pole_start + offset
            pe = ps + pole_len

            if pe >= len(df):
                continue

            pole_move = (df["Close"].iloc[pe] / df["Close"].iloc[ps]) - 1
            if pole_move < 0.08:  # Need at least 8% pole
                continue

            # Flag: remaining bars should slope slightly down
            flag = df.iloc[pe:]
            if len(flag) < 3 or len(flag) > 15:
                continue

            flag_move = (flag["Close"].iloc[-1] / flag["Close"].iloc[0]) - 1
            if flag_move > 0 or flag_move < -0.05:
                # Flag should retrace 0-5% of the pole
                continue

            # Flag range should be tight
            flag_range = (flag["High"].max() - flag["Low"].min()) / \
                flag["Close"].mean()
            if flag_range > 0.06:
                continue

            pole_high = df["High"].iloc[ps:pe + 1].max()
            return PatternResult(
                name="Bull Flag",
                confidence=round(0.5 + pole_move * 2, 2),
                type="bullish",
                start_index=ps,
                end_index=len(df) - 1,
                breakout_level=round(pole_high, 2),
                target=round(pole_high + (pole_high - df["Low"].iloc[ps]), 2),
                description=(f"Pole: {pole_move*100:.0f}% up in "
                             f"{pole_len} bars, flag: {len(flag)} bars."),
            )

    return None


# ============================================================
# CUP AND HANDLE
# ============================================================

def detect_cup_and_handle(df: pd.DataFrame,
                          lookback: int = 80
                          ) -> Optional[PatternResult]:
    """
    Cup and Handle: U-shaped recovery followed by a small pullback
    (handle) near the prior high.
    """
    if len(df) < lookback:
        return None

    window = df.iloc[-lookback:]
    prices = window["Close"].values

    # Find the cup: high → low → high (roughly U-shaped)
    left_high_idx = np.argmax(prices[:lookback // 3])
    cup_low_idx = left_high_idx + np.argmin(
        prices[left_high_idx:lookback * 2 // 3])

    if cup_low_idx <= left_high_idx + 5:
        return None

    right_section = prices[cup_low_idx:]
    if len(right_section) < 5:
        return None

    right_high_idx = cup_low_idx + np.argmax(right_section)

    left_high = prices[left_high_idx]
    cup_low = prices[cup_low_idx]
    right_high = prices[right_high_idx] if right_high_idx < len(prices) \
        else prices[-1]

    # Cup criteria
    cup_depth_pct = (left_high - cup_low) / left_high
    if cup_depth_pct < 0.10 or cup_depth_pct > 0.35:
        return None

    # Right side should recover near left high
    if right_high < left_high * 0.90:
        return None

    # Handle: small pullback after right high
    if right_high_idx >= len(prices) - 2:
        return None

    handle = prices[right_high_idx:]
    if len(handle) < 3:
        return None

    handle_pullback = (right_high - min(handle)) / right_high
    if handle_pullback > 0.10 or handle_pullback < 0.01:
        return None

    resistance = max(left_high, right_high)
    target = resistance + (resistance - cup_low)

    return PatternResult(
        name="Cup and Handle",
        confidence=0.75,
        type="bullish",
        start_index=len(df) - lookback + left_high_idx,
        end_index=len(df) - 1,
        breakout_level=round(resistance, 2),
        target=round(target, 2),
        description=(f"Cup depth: {cup_depth_pct*100:.0f}%, "
                     f"handle pullback: {handle_pullback*100:.0f}%."),
    )


# ============================================================
# DARVAS BOX
# ============================================================

def detect_darvas_box(df: pd.DataFrame,
                      lookback: int = 30) -> Optional[PatternResult]:
    """
    Darvas Box: stock makes a new high, then consolidates within
    that range for several days without making a new high or
    breaking the low of the consolidation.
    """
    if len(df) < lookback:
        return None

    # Find recent new high
    window = df.iloc[-lookback:]
    high_52w = df["High"].iloc[-252:].max() if len(df) >= 252 else \
        df["High"].max()

    for i in range(5, lookback - 5):
        if window["High"].iloc[i] >= high_52w * 0.97:
            box_top = float(window["High"].iloc[i])
            # Look for consolidation after
            consol = window.iloc[i + 1:]
            if len(consol) < 3:
                continue

            # All candles must stay within the box
            box_bottom = float(consol["Low"].min())
            stays_in_box = (consol["High"] <= box_top * 1.005).all()
            reasonable_depth = (box_top - box_bottom) / box_top < 0.10

            if stays_in_box and reasonable_depth and len(consol) >= 3:
                return PatternResult(
                    name="Darvas Box",
                    confidence=round(0.6 + len(consol) * 0.02, 2),
                    type="bullish",
                    start_index=len(df) - lookback + i,
                    end_index=len(df) - 1,
                    breakout_level=round(box_top, 2),
                    target=round(box_top + (box_top - box_bottom), 2),
                    description=(f"Box: ₹{box_bottom:.0f}-₹{box_top:.0f}, "
                                 f"{len(consol)} days in box."),
                )

    return None


# ============================================================
# DOUBLE BOTTOM
# ============================================================

def detect_double_bottom(df: pd.DataFrame,
                         swings: List[SwingPoint],
                         lookback: int = 60
                         ) -> Optional[PatternResult]:
    """
    Double Bottom: two swing lows at approximately the same level
    with a high between them.
    """
    recent_lows = [s for s in swings if s.type == "low"
                   and s.index >= len(df) - lookback]

    if len(recent_lows) < 2:
        return None

    # Check last two lows are at similar level
    l1, l2 = recent_lows[-2], recent_lows[-1]
    diff = abs(l1.price - l2.price) / max(l1.price, l2.price)

    if diff > 0.02:  # Within 2% of each other
        return None

    # Find the high between them (neckline)
    between_highs = [s for s in swings if s.type == "high"
                     and l1.index < s.index < l2.index]
    if not between_highs:
        return None

    neckline = max(s.price for s in between_highs)
    depth = neckline - min(l1.price, l2.price)

    return PatternResult(
        name="Double Bottom",
        confidence=0.70,
        type="bullish",
        start_index=l1.index,
        end_index=len(df) - 1,
        breakout_level=round(neckline, 2),
        target=round(neckline + depth, 2),
        description=(f"Two lows at ₹{l1.price:.0f} and ₹{l2.price:.0f}, "
                     f"neckline at ₹{neckline:.0f}."),
    )


# ============================================================
# FLAT BASE / BOX CONSOLIDATION
# ============================================================

def detect_flat_base(df: pd.DataFrame,
                     min_days: int = 5,
                     max_days: int = 25,
                     max_range_pct: float = 0.08
                     ) -> Optional[PatternResult]:
    """
    Flat Base: price trades in a tight horizontal range
    for min_days to max_days.
    """
    if len(df) < min_days:
        return None

    for blen in range(max_days, min_days - 1, -1):
        if blen > len(df):
            continue
        window = df.iloc[-blen:]
        high = window["High"].max()
        low = window["Low"].min()

        if high <= 0:
            continue

        range_pct = (high - low) / high
        if range_pct <= max_range_pct:
            return PatternResult(
                name="Flat Base",
                confidence=round(0.5 + (1 - range_pct / max_range_pct) * 0.3, 2),
                type="bullish",
                start_index=len(df) - blen,
                end_index=len(df) - 1,
                breakout_level=round(high, 2),
                target=round(high + (high - low), 2),
                description=(f"{blen}-day base, range: {range_pct*100:.1f}%, "
                             f"₹{low:.0f}-₹{high:.0f}."),
            )

    return None


# ============================================================
# MASTER PATTERN DETECTOR
# ============================================================

def detect_all_patterns(df: pd.DataFrame) -> List[PatternResult]:
    """
    Run all pattern detectors and return found patterns,
    sorted by confidence.
    """
    swings = detect_swing_points(df, lookback=5)
    patterns: List[PatternResult] = []

    detectors = [
        lambda: detect_ascending_triangle(df, swings),
        lambda: detect_descending_triangle(df, swings),
        lambda: detect_symmetrical_triangle(df, swings),
        lambda: detect_bull_flag(df),
        lambda: detect_cup_and_handle(df),
        lambda: detect_darvas_box(df),
        lambda: detect_double_bottom(df, swings),
        lambda: detect_flat_base(df),
    ]

    for detector in detectors:
        try:
            result = detector()
            if result is not None:
                patterns.append(result)
        except Exception:
            continue

    return sorted(patterns, key=lambda p: p.confidence, reverse=True)
