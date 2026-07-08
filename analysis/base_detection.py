"""
Base Detection — Phase 2 of the breakout methodology.

Extracted and enhanced from the original breakout_scanner.py.
Finds valid 5-15 day consolidation bases with institutional
edge signals (VCP, Vol Coil, Tightening, Net Accumulation).
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings


def find_base(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Find a valid consolidation base.

    For WATCHLIST: base includes today (stock still resting).
    For BREAKOUT: base is in prior days, today is breakout candle.

    Returns (found, info_dict).
    """
    cfg = Settings.get()

    if len(df) < 20:
        return False, {}

    # Try to find a base ending at different points
    for end_offset in [0, 1]:
        end_idx = len(df) - 1 - end_offset
        if end_idx < cfg.base_max_days:
            continue

        for blen in range(cfg.base_max_days, cfg.base_min_days - 1, -1):
            start_idx = end_idx - blen + 1
            if start_idx < 1:
                continue

            base = df.iloc[start_idx:end_idx + 1]
            pivot = float(base["High"].max())

            # ── BASE VALIDITY CHECKS ──

            # 1. Daily candles are small (range < ADR)
            adr_at_start = df["adr"].iloc[start_idx] if "adr" in df.columns \
                else (df["High"] - df["Low"]).iloc[
                    max(0, start_idx - 20):start_idx].mean()
            if pd.isna(adr_at_start) or adr_at_start <= 0:
                continue
            ranges = base["High"] - base["Low"]
            small_pct = (ranges < adr_at_start *
                         cfg.base_small_candle_adr_mult).sum() / len(base)
            if small_pct < cfg.base_small_candle_pct:
                continue

            # 2. Volume drying up during base
            vol_ma = df["vol_ma50"].iloc[end_idx] if "vol_ma50" in df.columns \
                else df["Volume"].iloc[max(0, end_idx - 50):end_idx].mean()
            if pd.isna(vol_ma) or vol_ma <= 0:
                continue
            base_vol_avg = base["Volume"].mean()
            if base_vol_avg >= vol_ma * cfg.base_vol_dryup_mult:
                continue

            # 3. No close below EMA-21
            if "ema21" in df.columns and (base["Close"] < base["ema21"]).any():
                continue

            # 4. No 3+ consecutive down days
            down = (base["Close"] < base["Close"].shift(1)).astype(int)
            if len(base) >= cfg.base_max_consecutive_down and \
                    down.rolling(cfg.base_max_consecutive_down).sum().max() >= \
                    cfg.base_max_consecutive_down:
                continue

            # 5. No volume spike on a down day (distribution)
            down_mask = base["Close"] < base["Open"]
            vol_spike = base["Volume"] > vol_ma * cfg.base_distribution_vol_mult
            if (down_mask & vol_spike).any():
                continue

            # 6. No close below EMA-55
            if "ema55" in df.columns and (base["Close"] < base["ema55"]).any():
                continue

            # 7. Price holding near resistance
            last_close = base["Close"].iloc[-1]
            if last_close < pivot * cfg.base_price_near_pivot_pct:
                continue

            # 8. Supply Bar Prevention
            upper_wick = base["High"] - np.maximum(base["Close"], base["Open"])
            body = abs(base["Close"] - base["Open"])
            wick_ratio = upper_wick / (body + 1e-5)
            supply_bar = (
                (wick_ratio > cfg.base_supply_bar_wick_ratio) &
                (base["Volume"] > vol_ma * cfg.base_supply_bar_vol_mult)
            )
            if supply_bar.any():
                continue

            # ── BONUS SIGNALS ──

            # SFP: Swing Failure Pattern
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
            body_vals = abs(base["Close"] - base["Open"])
            wick_vals = (base["High"] - base["Low"]).replace(0, np.nan)
            doji_count = int((body_vals / wick_vals < 0.15).sum())

            # ── INSTITUTIONAL EDGE SIGNALS ──

            # VCP: Volatility Contraction Pattern
            half = len(base) // 2
            if half >= 2:
                first_half_range = (
                    base["High"].iloc[:half] - base["Low"].iloc[:half]
                ).mean()
                second_half_range = (
                    base["High"].iloc[half:] - base["Low"].iloc[half:]
                ).mean()
                has_vcp = second_half_range < first_half_range * \
                    cfg.base_vcp_contraction_pct
            else:
                has_vcp = False

            # Volume coiling: rising volume + flat price in last 3 days
            vol_coil = False
            if len(base) >= 4:
                last3_vol = base["Volume"].iloc[-3:]
                last3_range = (
                    base["High"].iloc[-3:] - base["Low"].iloc[-3:]
                ).mean()
                avg_base_range = ranges.mean()
                if (last3_vol.is_monotonic_increasing
                        and last3_range < avg_base_range * 0.9):
                    vol_coil = True

            # Tightening closes
            tightening = False
            if "close_range_5" in df.columns and "close_range_20" in df.columns:
                cr5 = df["close_range_5"].iloc[end_idx]
                cr20 = df["close_range_20"].iloc[end_idx]
                if (not pd.isna(cr5) and not pd.isna(cr20) and cr20 > 0
                        and cr5 < cr20 * cfg.base_tightening_ratio):
                    tightening = True

            # Net accumulation
            net_accum = 0
            if "accum_days_20" in df.columns and \
                    "distrib_days_20" in df.columns:
                accum = df["accum_days_20"].iloc[end_idx]
                distrib = df["distrib_days_20"].iloc[end_idx]
                if not pd.isna(accum) and not pd.isna(distrib):
                    net_accum = int(accum) - int(distrib)

            # NR7 in base
            has_nr7 = False
            if "nr7" in df.columns:
                has_nr7 = bool(base["nr7"].any()) if "nr7" in base.columns \
                    else False

            # Inside bars in base
            inside_bar_count = 0
            if "inside_bar" in df.columns:
                inside_bar_count = int(base["inside_bar"].sum()) if \
                    "inside_bar" in base.columns else 0

            # Squeeze in base
            squeeze_active = False
            if "squeeze_on" in df.columns:
                squeeze_active = bool(df["squeeze_on"].iloc[end_idx])

            return True, {
                "pivot": pivot,
                "base_len": blen,
                "has_sfp": has_sfp,
                "doji_count": doji_count,
                "base_end_offset": end_offset,
                "has_vcp": has_vcp,
                "vol_coil": vol_coil,
                "tightening": tightening,
                "net_accum": net_accum,
                "has_nr7": has_nr7,
                "inside_bar_count": inside_bar_count,
                "squeeze_active": squeeze_active,
                "base_vol_ratio": round(base_vol_avg / vol_ma, 3)
                    if vol_ma > 0 else 0,
                "base_range_pct": round(
                    (pivot - base["Low"].min()) / pivot, 4)
                    if pivot > 0 else 0,
            }

    return False, {}


def detect_uptrend(df: pd.DataFrame) -> bool:
    """
    Phase 1: Prior Uptrend Detection.
    EMA-8 > EMA-21 > EMA-55, EMA-144 rising, U/D >= 1.4.
    """
    cfg = Settings.get()

    if len(df) < 144:
        return False

    latest = df.iloc[-1]

    # EMA stack
    stack = cfg.uptrend_ema_stack  # [8, 21, 55]
    ema_cols = [f"ema{p}" for p in stack]
    if not all(c in df.columns for c in ema_cols):
        return False
    ema_vals = [latest[c] for c in ema_cols]
    for i in range(len(ema_vals) - 1):
        if ema_vals[i] <= ema_vals[i + 1]:
            return False

    # EMA-144 pointing up
    ema_long_col = f"ema{cfg.uptrend_ema_long}"
    if ema_long_col in df.columns:
        e_now = df[ema_long_col].iloc[-1]
        e_prev = df[ema_long_col].iloc[-cfg.uptrend_ema_long_lookback]
        if e_now < e_prev * cfg.uptrend_ema_long_tolerance:
            return False

    # U/D ratio
    if "ud_ratio" in df.columns:
        ud = latest["ud_ratio"]
        if pd.isna(ud) or ud < cfg.uptrend_ud_ratio_min:
            return False

    # Proximity to 120-day high
    if "high_120" in df.columns:
        high_120 = latest["high_120"]
        if pd.isna(high_120) or latest["Close"] < high_120 * \
                cfg.uptrend_proximity_pct:
            return False

    # Liquidity check
    if "turnover_ma50" in df.columns:
        turnover = latest["turnover_ma50"]
        if pd.isna(turnover) or turnover < Settings.get().min_turnover:
            return False

    # Anti-exhaustion filters
    if "rsi" in df.columns:
        rsi = latest.get("rsi", 50)
        if not pd.isna(rsi) and rsi >= cfg.uptrend_rsi_max:
            return False

    if "ema21" in df.columns:
        if latest["Close"] > latest["ema21"] * cfg.uptrend_extension_ema21:
            return False

    if ema_long_col in df.columns:
        if latest["Close"] > latest[ema_long_col] * \
                cfg.uptrend_extension_ema144:
            return False

    return True
