"""
Breakout Confirmation — Phase 3 of the breakout methodology.

Enhanced from the original breakout_scanner.py with
configurable thresholds and additional checks.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from config.settings import Settings


def is_breakout_candle(df: pd.DataFrame, pivot: float) -> bool:
    """
    Check if the LATEST candle is a valid breakout.

    Criteria:
    1. Close above pivot/resistance
    2. Green candle with body > 60% of range
    3. Close in top 8% of day's range
    4. RVol >= configured threshold
    5. Move >= configured minimum from previous close
    6. Not chasing (no 3+ consecutive up days before)
    """
    cfg = Settings.get()

    if len(df) < 5:
        return False

    latest = df.iloc[-1]
    prev_close = float(df["Close"].iloc[-2])

    # 1. Close above resistance
    if latest["Close"] <= pivot:
        return False

    # 2. Green candle with body > threshold of range
    rng = latest["High"] - latest["Low"]
    if rng <= 0:
        return False
    body = latest["Close"] - latest["Open"]
    if body <= 0 or body / rng < cfg.breakout_body_pct:
        return False

    # 3. Close in top portion of day's range
    if latest["Close"] < latest["High"] - rng * cfg.breakout_close_top_pct:
        return False

    # 4. Relative volume check
    rvol_col = "rvol" if "rvol" in df.columns else None
    if rvol_col:
        rvol = latest[rvol_col]
        if pd.isna(rvol) or rvol < cfg.breakout_rvol_min:
            return False

    # 5. Minimum move from previous close
    move = latest["Close"] / prev_close - 1
    if move < cfg.breakout_move_min_pct:
        return False

    # 6. No chasing: reject if N consecutive up days before
    chase_days = cfg.breakout_no_chase_days
    if len(df) >= chase_days + 2:
        all_up = True
        for d in range(2, chase_days + 2):
            if df["Close"].iloc[-d] <= df["Close"].iloc[-d - 1]:
                all_up = False
                break
        if all_up:
            return False

    return True


def confluence_score(latest: pd.Series) -> int:
    """
    Count how many confluence checks pass.
    Playbook: at least 4 of 6 must be true.
    """
    checks = [
        # MACD bullish crossover or positive histogram
        latest.get("macd_hist", 0) > 0,
        # RSI above 55
        latest.get("rsi", 0) >= 55,
        # Stochastic above 50
        latest.get("stoch_k", latest.get("stoch", 0)) >= 50,
        # Volume expanding
        latest.get("rvol", 0) >= 1.2,
        # Trend: EMA-8 > EMA-21 > EMA-55
        bool(latest.get("ema8", 0) > latest.get("ema21", 0)
             > latest.get("ema55", 0)),
        # ADX above 20
        latest.get("adx", 0) >= 20,
    ]
    return sum(bool(x) for x in checks)


def calc_risk(price: float, breakout_low: float,
              atr: float = 0) -> Dict[str, Any]:
    """
    Calculate stop loss, targets, and risk/reward.

    Stop = low of breakout candle (or ATR-based if available).
    Target 1 = 2R, Target 2 = 3R, Target 3 = 5R.
    """
    cfg = Settings.get()

    stop = round(breakout_low, 2)
    risk = price - stop
    risk_pct = risk / price if price > stop else 1.0
    risk_ok = risk_pct <= cfg.risk_max_pct

    target_1 = round(price + risk * 2, 2) if risk > 0 else price
    target_2 = round(price + risk * 3, 2) if risk > 0 else price
    target_3 = round(price + risk * 5, 2) if risk > 0 else price
    rr = round(2.0, 1) if risk > 0 else 0  # Base R:R is 2:1

    return {
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "target_3": target_3,
        "rr": rr,
        "risk_pct": round(risk_pct, 4),
        "risk_ok": risk_ok,
        "risk_amount": round(risk, 2),
    }


def calculate_entry_zone(pivot: float, atr: float = 0) -> Dict[str, float]:
    """
    Calculate ideal entry zone around the breakout level.
    Entry zone = pivot to pivot + 0.5 * ATR.
    """
    buffer = atr * 0.5 if atr > 0 else pivot * 0.005
    return {
        "entry_low": round(pivot, 2),
        "entry_high": round(pivot + buffer, 2),
    }
