"""
Volume Indicators.

Computes:
  - On Balance Volume (OBV) with trend
  - Chaikin Money Flow (CMF)
  - Money Flow Index (MFI)
  - VWAP (session-level)
  - Volume Profile (POC, VAH, VAL)
  - Accumulation/Distribution line
  - Relative Volume (vs 20, 50 day averages)
  - Volume dry-up detection
  - Buying/Selling pressure (Close Location Value)
  - Accumulation/Distribution day counting
"""

import numpy as np
import pandas as pd
import ta


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On Balance Volume with slope."""
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    # OBV slope (5-day rate of change)
    df["obv_slope"] = df["obv"].pct_change(5)
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow."""
    df["cmf"] = ta.volume.chaikin_money_flow(
        df["High"], df["Low"], df["Close"], df["Volume"],
        window=period)
    return df


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index."""
    df["mfi"] = ta.volume.money_flow_index(
        df["High"], df["Low"], df["Close"], df["Volume"],
        window=period)
    return df


def add_ad_line(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulation/Distribution Line."""
    df["ad_line"] = ta.volume.acc_dist_index(
        df["High"], df["Low"], df["Close"], df["Volume"])
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Weighted Average Price.
    Uses rolling 20-day VWAP as a proxy since we use daily data.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (tp * df["Volume"]).rolling(20).sum()
    cum_vol = df["Volume"].rolling(20).sum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return df


def add_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Relative volume vs 20-day and 50-day averages."""
    vol = df["Volume"]
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ma50"] = vol.rolling(50).mean()
    df["rvol_20"] = vol / df["vol_ma20"].replace(0, np.nan)
    df["rvol"] = vol / df["vol_ma50"].replace(0, np.nan)
    return df


def add_volume_dryup(df: pd.DataFrame, lookback: int = 5,
                     threshold: float = 0.5) -> pd.DataFrame:
    """
    Detect volume dry-up: average volume of last N days is
    below threshold * 50-day average.
    """
    if "vol_ma50" not in df.columns:
        df = add_relative_volume(df)

    recent_avg = df["Volume"].rolling(lookback).mean()
    df["vol_dryup"] = recent_avg < (df["vol_ma50"] * threshold)
    return df


def add_buying_selling_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close Location Value (CLV): where close falls within the
    day's range. +1 = closed at high, -1 = closed at low.
    """
    rng = df["High"] - df["Low"]
    rng = rng.replace(0, np.nan)
    df["clv"] = (2 * df["Close"] - df["Low"] - df["High"]) / rng

    # Rolling buying pressure (average CLV over 10 days)
    df["buying_pressure_10"] = df["clv"].rolling(10).mean()
    return df


def add_accumulation_distribution_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count accumulation days (up close on above-avg volume) and
    distribution days (down close on above-avg volume) over 20 sessions.
    """
    if "vol_ma50" not in df.columns:
        df = add_relative_volume(df)

    c, v = df["Close"], df["Volume"]
    up_vol = ((c > c.shift(1)) & (v > df["vol_ma50"])).astype(int)
    dn_vol = ((c < c.shift(1)) & (v > df["vol_ma50"])).astype(int)
    df["accum_days_20"] = up_vol.rolling(20).sum()
    df["distrib_days_20"] = dn_vol.rolling(20).sum()
    df["net_accum_20"] = df["accum_days_20"] - df["distrib_days_20"]
    return df


def add_volume_profile(df: pd.DataFrame,
                       lookback: int = 60,
                       bins: int = 20) -> pd.DataFrame:
    """
    Simplified Volume Profile: find Point of Control (POC),
    Value Area High (VAH), and Value Area Low (VAL) over
    the lookback period.

    POC = price level with the most volume.
    Value Area = 70% of volume centered around POC.
    """
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val_ = np.full(n, np.nan)

    for i in range(lookback, n):
        window = df.iloc[i - lookback:i + 1]
        prices = window["Close"].values
        volumes = window["Volume"].values

        lo, hi = prices.min(), prices.max()
        if hi <= lo:
            continue

        # Create price bins
        bin_edges = np.linspace(lo, hi, bins + 1)
        bin_vols = np.zeros(bins)

        for p, v in zip(prices, volumes):
            idx = int((p - lo) / (hi - lo) * (bins - 1))
            idx = max(0, min(idx, bins - 1))
            bin_vols[idx] += v

        poc_idx = np.argmax(bin_vols)
        poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

        # Value area (70% of volume)
        total_vol = bin_vols.sum()
        target_vol = total_vol * 0.70
        sorted_idx = np.argsort(bin_vols)[::-1]
        cumul = 0
        va_bins = []
        for si in sorted_idx:
            cumul += bin_vols[si]
            va_bins.append(si)
            if cumul >= target_vol:
                break

        if va_bins:
            vah[i] = bin_edges[max(va_bins) + 1]
            val_[i] = bin_edges[min(va_bins)]

    df["vp_poc"] = poc
    df["vp_vah"] = vah
    df["vp_val"] = val_
    return df


def add_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """Average daily turnover (price × volume)."""
    if "vol_ma50" not in df.columns:
        df = add_relative_volume(df)
    df["turnover"] = df["Volume"] * df["Close"]
    df["turnover_ma50"] = df["turnover"].rolling(50).mean()
    return df


def add_tightening_closes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tightening closes: std of close-to-close changes.
    Used to detect coiling/spring-loading patterns.
    """
    close_range = abs(df["Close"] - df["Close"].shift(1))
    df["close_range_5"] = close_range.rolling(5).std()
    df["close_range_20"] = close_range.rolling(20).std()
    return df


def add_all_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all volume indicators in one call."""
    df = add_obv(df)
    df = add_cmf(df)
    df = add_mfi(df)
    df = add_ad_line(df)
    df = add_vwap(df)
    df = add_relative_volume(df)
    df = add_volume_dryup(df)
    df = add_buying_selling_pressure(df)
    df = add_accumulation_distribution_days(df)
    df = add_volume_profile(df)
    df = add_turnover(df)
    df = add_tightening_closes(df)
    return df
