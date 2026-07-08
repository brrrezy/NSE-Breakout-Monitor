"""
Smart Money / Liquidity Analysis.

Detects:
  - Equal Highs / Equal Lows (liquidity pools)
  - Stop Hunt zones
  - Liquidity sweeps (wick through key level + reversal)
  - Order Blocks (OB)
  - Fair Value Gaps (FVG)
  - Breaker Blocks
  - Premium / Discount zones
  - Rejection candles (pin bars at key levels)
  - Order absorption (high volume + small body at resistance)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ============================================================
# EQUAL HIGHS / EQUAL LOWS (LIQUIDITY POOLS)
# ============================================================

def detect_equal_levels(df: pd.DataFrame,
                        tolerance_pct: float = 0.003,
                        min_touches: int = 2,
                        lookback: int = 60) -> dict:
    """
    Detect equal highs and equal lows — areas where price has
    tested the same level multiple times, creating a liquidity pool.
    """
    if len(df) < lookback:
        return {"equal_highs": [], "equal_lows": []}

    recent = df.iloc[-lookback:]
    highs = recent["High"].values
    lows = recent["Low"].values

    equal_highs = _find_equal_levels(highs, tolerance_pct, min_touches)
    equal_lows = _find_equal_levels(lows, tolerance_pct, min_touches)

    return {
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "eq_high_count": len(equal_highs),
        "eq_low_count": len(equal_lows),
    }


def _find_equal_levels(prices: np.ndarray,
                       tolerance_pct: float,
                       min_touches: int) -> List[dict]:
    """Find price levels touched multiple times."""
    levels = []
    used = set()

    for i in range(len(prices)):
        if i in used:
            continue
        level = prices[i]
        tol = level * tolerance_pct
        touches = []
        for j in range(i, len(prices)):
            if abs(prices[j] - level) <= tol:
                touches.append(j)
                used.add(j)
        if len(touches) >= min_touches:
            levels.append({
                "price": round(float(np.mean(
                    [prices[t] for t in touches])), 2),
                "touches": len(touches),
                "first_bar": int(touches[0]),
                "last_bar": int(touches[-1]),
            })

    return sorted(levels, key=lambda x: x["touches"], reverse=True)


# ============================================================
# LIQUIDITY SWEEP DETECTION
# ============================================================

def detect_liquidity_sweeps(df: pd.DataFrame,
                            equal_levels: dict,
                            lookback: int = 10) -> List[dict]:
    """
    Detect liquidity sweeps: price briefly pierces through an
    equal high/low level, then closes back inside — a stop hunt.
    """
    sweeps = []
    recent = df.iloc[-lookback:]

    for level_info in equal_levels.get("equal_highs", []):
        level = level_info["price"]
        for idx in range(len(recent)):
            row = recent.iloc[idx]
            # Wick above level but close below
            if (row["High"] > level and row["Close"] < level
                    and row["Close"] < row["Open"]):
                sweeps.append({
                    "type": "bearish_sweep",
                    "level": level,
                    "bar_index": recent.index[idx],
                    "wick_above": round(float(row["High"] - level), 2),
                })

    for level_info in equal_levels.get("equal_lows", []):
        level = level_info["price"]
        for idx in range(len(recent)):
            row = recent.iloc[idx]
            # Wick below level but close above
            if (row["Low"] < level and row["Close"] > level
                    and row["Close"] > row["Open"]):
                sweeps.append({
                    "type": "bullish_sweep",
                    "level": level,
                    "bar_index": recent.index[idx],
                    "wick_below": round(float(level - row["Low"]), 2),
                })

    return sweeps


# ============================================================
# ORDER BLOCKS
# ============================================================

@dataclass
class OrderBlock:
    """An institutional order block."""
    index: int
    high: float
    low: float
    type: str           # "bullish" or "bearish"
    volume: float
    tested: bool = False
    mitigated: bool = False


def detect_order_blocks(df: pd.DataFrame,
                        lookback: int = 60) -> List[OrderBlock]:
    """
    Detect order blocks:
    - Bullish OB: last bearish candle before a strong bullish impulse
    - Bearish OB: last bullish candle before a strong bearish impulse

    An impulse is defined as 3+ consecutive candles in one direction
    with above-average range.
    """
    if len(df) < lookback:
        return []

    recent = df.iloc[-lookback:]
    obs: List[OrderBlock] = []
    adr = (recent["High"] - recent["Low"]).mean()

    for i in range(2, len(recent) - 3):
        # Check for bullish impulse (3 consecutive green candles with range > ADR)
        is_bullish_impulse = True
        for j in range(i + 1, min(i + 4, len(recent))):
            row = recent.iloc[j]
            if (row["Close"] <= row["Open"] or
                    (row["High"] - row["Low"]) < adr * 0.8):
                is_bullish_impulse = False
                break

        if is_bullish_impulse:
            # The candle before the impulse (bearish) is the bullish OB
            candle = recent.iloc[i]
            if candle["Close"] < candle["Open"]:  # bearish candle
                ob = OrderBlock(
                    index=i,
                    high=float(candle["High"]),
                    low=float(candle["Low"]),
                    type="bullish",
                    volume=float(candle["Volume"]),
                )
                # Check if OB has been tested
                current_price = float(recent["Close"].iloc[-1])
                if candle["Low"] <= current_price <= candle["High"]:
                    ob.tested = True
                if current_price < candle["Low"]:
                    ob.mitigated = True
                obs.append(ob)

        # Check for bearish impulse
        is_bearish_impulse = True
        for j in range(i + 1, min(i + 4, len(recent))):
            row = recent.iloc[j]
            if (row["Close"] >= row["Open"] or
                    (row["High"] - row["Low"]) < adr * 0.8):
                is_bearish_impulse = False
                break

        if is_bearish_impulse:
            candle = recent.iloc[i]
            if candle["Close"] > candle["Open"]:  # bullish candle
                ob = OrderBlock(
                    index=i,
                    high=float(candle["High"]),
                    low=float(candle["Low"]),
                    type="bearish",
                    volume=float(candle["Volume"]),
                )
                current_price = float(recent["Close"].iloc[-1])
                if candle["Low"] <= current_price <= candle["High"]:
                    ob.tested = True
                if current_price > candle["High"]:
                    ob.mitigated = True
                obs.append(ob)

    return obs


# ============================================================
# FAIR VALUE GAPS (FVG)
# ============================================================

@dataclass
class FairValueGap:
    """A 3-candle imbalance / Fair Value Gap."""
    index: int          # Index of the middle candle
    upper: float
    lower: float
    type: str           # "bullish" or "bearish"
    filled: bool = False


def detect_fvgs(df: pd.DataFrame, lookback: int = 30) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (3-candle imbalances).

    Bullish FVG: candle 3's low > candle 1's high (gap up)
    Bearish FVG: candle 1's low > candle 3's high (gap down)
    """
    if len(df) < lookback:
        return []

    recent = df.iloc[-lookback:]
    fvgs: List[FairValueGap] = []

    for i in range(1, len(recent) - 1):
        c1 = recent.iloc[i - 1]  # First candle
        c3 = recent.iloc[i + 1]  # Third candle
        current_price = float(recent["Close"].iloc[-1])

        # Bullish FVG: gap between candle 1's high and candle 3's low
        if c3["Low"] > c1["High"]:
            fvg = FairValueGap(
                index=i,
                upper=float(c3["Low"]),
                lower=float(c1["High"]),
                type="bullish",
                filled=current_price <= c1["High"],
            )
            fvgs.append(fvg)

        # Bearish FVG: gap between candle 3's high and candle 1's low
        elif c1["Low"] > c3["High"]:
            fvg = FairValueGap(
                index=i,
                upper=float(c1["Low"]),
                lower=float(c3["High"]),
                type="bearish",
                filled=current_price >= c1["Low"],
            )
            fvgs.append(fvg)

    return fvgs


# ============================================================
# REJECTION CANDLES
# ============================================================

def detect_rejection_candles(df: pd.DataFrame,
                             lookback: int = 5) -> List[dict]:
    """
    Detect rejection candles (pin bars) at the end of the data.
    Long lower wick + small body = bullish rejection.
    Long upper wick + small body = bearish rejection.
    """
    rejections = []
    recent = df.iloc[-lookback:]

    for idx in range(len(recent)):
        row = recent.iloc[idx]
        body = abs(row["Close"] - row["Open"])
        rng = row["High"] - row["Low"]
        if rng <= 0:
            continue

        upper_wick = row["High"] - max(row["Close"], row["Open"])
        lower_wick = min(row["Close"], row["Open"]) - row["Low"]

        # Bullish rejection: long lower wick (> 2x body)
        if lower_wick > body * 2 and lower_wick > rng * 0.6:
            rejections.append({
                "type": "bullish_rejection",
                "bar_index": recent.index[idx],
                "wick_ratio": round(lower_wick / max(body, 0.01), 2),
            })

        # Bearish rejection: long upper wick (> 2x body)
        if upper_wick > body * 2 and upper_wick > rng * 0.6:
            rejections.append({
                "type": "bearish_rejection",
                "bar_index": recent.index[idx],
                "wick_ratio": round(upper_wick / max(body, 0.01), 2),
            })

    return rejections


# ============================================================
# ORDER ABSORPTION
# ============================================================

def detect_order_absorption(df: pd.DataFrame,
                            resistance: Optional[float] = None,
                            lookback: int = 10) -> dict:
    """
    Detect order absorption: high volume candle with small body
    at a resistance level. Indicates large orders being absorbed.
    """
    if len(df) < lookback:
        return {"absorption_detected": False}

    recent = df.iloc[-lookback:]
    vol_avg = df["Volume"].iloc[-50:].mean() if len(df) >= 50 else \
        recent["Volume"].mean()

    for idx in range(len(recent)):
        row = recent.iloc[idx]
        body = abs(row["Close"] - row["Open"])
        rng = row["High"] - row["Low"]
        if rng <= 0:
            continue

        body_pct = body / rng
        vol_mult = row["Volume"] / vol_avg if vol_avg > 0 else 0

        # Small body + high volume
        if body_pct < 0.3 and vol_mult > 1.5:
            # At resistance?
            if resistance and abs(row["High"] - resistance) / \
                    resistance < 0.01:
                return {
                    "absorption_detected": True,
                    "bar_index": recent.index[idx],
                    "volume_mult": round(vol_mult, 2),
                    "body_pct": round(body_pct, 3),
                }

    return {"absorption_detected": False}


# ============================================================
# PREMIUM / DISCOUNT ZONES
# ============================================================

def premium_discount_zone(df: pd.DataFrame,
                          lookback: int = 60) -> dict:
    """
    Determine if price is in premium or discount zone.
    Based on the recent range: above midpoint = premium,
    below = discount.
    """
    if len(df) < lookback:
        return {"zone": "unknown", "zone_pct": 0.5}

    recent = df.iloc[-lookback:]
    range_high = recent["High"].max()
    range_low = recent["Low"].min()

    if range_high <= range_low:
        return {"zone": "unknown", "zone_pct": 0.5}

    current = float(df["Close"].iloc[-1])
    pct = (current - range_low) / (range_high - range_low)

    if pct >= 0.5:
        zone = "premium"
    else:
        zone = "discount"

    return {
        "zone": zone,
        "zone_pct": round(pct, 3),
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "range_mid": round((range_high + range_low) / 2, 2),
    }


# ============================================================
# FULL LIQUIDITY ANALYSIS
# ============================================================

def analyze_liquidity(df: pd.DataFrame,
                      resistance: Optional[float] = None) -> dict:
    """
    Complete liquidity analysis for one stock.
    """
    equal = detect_equal_levels(df)
    sweeps = detect_liquidity_sweeps(df, equal)
    obs = detect_order_blocks(df)
    fvgs = detect_fvgs(df)
    rejections = detect_rejection_candles(df)
    absorption = detect_order_absorption(df, resistance)
    pd_zone = premium_discount_zone(df)

    # Separate OBs
    bullish_obs = [ob for ob in obs if ob.type == "bullish"]
    bearish_obs = [ob for ob in obs if ob.type == "bearish"]
    untested_bull_obs = [ob for ob in bullish_obs
                         if not ob.tested and not ob.mitigated]

    # Unfilled FVGs near current price
    current_price = float(df["Close"].iloc[-1]) if not df.empty else 0
    relevant_fvgs = []
    for fvg in fvgs:
        if not fvg.filled:
            gap_dist = abs(
                (fvg.lower + fvg.upper) / 2 - current_price
            ) / current_price
            if gap_dist < 0.05:  # Within 5%
                relevant_fvgs.append(fvg)

    return {
        "equal_highs": equal["eq_high_count"],
        "equal_lows": equal["eq_low_count"],
        "equal_high_levels": [e["price"] for e in equal["equal_highs"][:3]],
        "equal_low_levels": [e["price"] for e in equal["equal_lows"][:3]],
        "sweep_count": len(sweeps),
        "recent_sweeps": sweeps[-3:] if sweeps else [],
        "bullish_order_blocks": len(bullish_obs),
        "bearish_order_blocks": len(bearish_obs),
        "untested_bullish_obs": len(untested_bull_obs),
        "fvg_count": len(fvgs),
        "unfilled_fvg_count": sum(1 for f in fvgs if not f.filled),
        "relevant_fvgs": len(relevant_fvgs),
        "rejection_candles": len(rejections),
        "recent_rejections": rejections[-2:] if rejections else [],
        "absorption": absorption,
        "zone": pd_zone["zone"],
        "zone_pct": pd_zone["zone_pct"],
    }
