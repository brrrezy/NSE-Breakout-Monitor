"""
Trend Indicators.

Computes:
  - EMAs (8, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200)
  - SMAs (50, 200)
  - ADX with +DI / -DI
  - Supertrend
  - Moving Average Ribbon
  - EMA slope & trend maturity
  - Trend quality scoring
"""

import numpy as np
import pandas as pd
import ta

from config.settings import Settings


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add all configured EMAs."""
    c = df["Close"]
    for p in Settings.get().ema_periods:
        col = f"ema{p}"
        if col not in df.columns:
            df[col] = ta.trend.ema_indicator(c, window=p)
    return df


def add_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Add configured SMAs."""
    c = df["Close"]
    for p in Settings.get().sma_periods:
        col = f"sma{p}"
        if col not in df.columns:
            df[col] = c.rolling(p).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with +DI and -DI."""
    h, l, c = df["High"], df["Low"], df["Close"]
    df["adx"] = ta.trend.adx(h, l, c, window=period)
    df["plus_di"] = ta.trend.adx_pos(h, l, c, window=period)
    df["minus_di"] = ta.trend.adx_neg(h, l, c, window=period)
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10,
                   multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    atr_vals = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=period).values

    n = len(df)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.ones(n)  # 1 = up, -1 = down

    for i in range(period, n):
        hl2 = (h[i] + l[i]) / 2
        atr_v = atr_vals[i] if not np.isnan(atr_vals[i]) else 0

        basic_upper = hl2 + multiplier * atr_v
        basic_lower = hl2 - multiplier * atr_v

        # Upper band
        if i == period:
            upper_band[i] = basic_upper
        else:
            upper_band[i] = (
                min(basic_upper, upper_band[i - 1])
                if c[i - 1] <= upper_band[i - 1]
                else basic_upper)

        # Lower band
        if i == period:
            lower_band[i] = basic_lower
        else:
            lower_band[i] = (
                max(basic_lower, lower_band[i - 1])
                if c[i - 1] >= lower_band[i - 1]
                else basic_lower)

        # Direction
        if i == period:
            direction[i] = 1 if c[i] > upper_band[i] else -1
        else:
            if direction[i - 1] == 1:
                direction[i] = -1 if c[i] < lower_band[i] else 1
            else:
                direction[i] = 1 if c[i] > upper_band[i] else -1

        supertrend[i] = (lower_band[i] if direction[i] == 1
                         else upper_band[i])

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction  # 1=bullish, -1=bearish
    return df


def add_ma_ribbon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moving Average Ribbon: 8/13/21/34/55/89 EMA.
    Adds ribbon_bullish (True if all stacked bullish) and
    ribbon_score (count of aligned pairs / total pairs).
    """
    ribbon_periods = [8, 13, 21, 34, 55, 89]
    ribbon_cols = []
    for p in ribbon_periods:
        col = f"ema{p}"
        if col not in df.columns:
            df[col] = ta.trend.ema_indicator(df["Close"], window=p)
        ribbon_cols.append(col)

    # Check alignment
    def _ribbon_score(row):
        vals = [row[c] for c in ribbon_cols if not pd.isna(row[c])]
        if len(vals) < 2:
            return 0
        aligned = sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1])
        return aligned / (len(vals) - 1)

    df["ribbon_score"] = df.apply(_ribbon_score, axis=1)
    df["ribbon_bullish"] = df["ribbon_score"] >= 0.8
    return df


def add_ema_slope(df: pd.DataFrame, ema_col: str = "ema21",
                  lookback: int = 5) -> pd.DataFrame:
    """
    EMA slope: rate of change of EMA over lookback periods.
    Positive = rising, negative = falling.
    """
    if ema_col in df.columns:
        df[f"{ema_col}_slope"] = (
            (df[ema_col] - df[ema_col].shift(lookback))
            / df[ema_col].shift(lookback)
        )
    return df


def add_trend_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend maturity: count consecutive days the EMA stack
    (8 > 21 > 55) has been aligned.
    """
    required = ["ema8", "ema21", "ema55"]
    if not all(c in df.columns for c in required):
        df["trend_maturity"] = 0
        return df

    aligned = (
        (df["ema8"] > df["ema21"]) & (df["ema21"] > df["ema55"])
    ).astype(int)

    # Count consecutive
    maturity = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if aligned.iloc[i]:
            maturity[i] = maturity[i - 1] + 1
        else:
            maturity[i] = 0
    df["trend_maturity"] = maturity
    return df


def add_trend_quality(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Trend quality: ratio of higher lows to total swing lows
    over the lookback period. Higher = cleaner uptrend.
    """
    lows = df["Low"].values
    n = len(lows)
    quality = np.full(n, np.nan)

    for i in range(lookback, n):
        window = lows[i - lookback:i + 1]
        # Find local minima (simplified: compare to neighbors)
        swing_lows = []
        for j in range(1, len(window) - 1):
            if window[j] < window[j - 1] and window[j] < window[j + 1]:
                swing_lows.append(window[j])
        if len(swing_lows) < 2:
            quality[i] = 1.0
            continue
        higher = sum(1 for k in range(1, len(swing_lows))
                     if swing_lows[k] > swing_lows[k - 1])
        quality[i] = higher / (len(swing_lows) - 1)

    df["trend_quality"] = quality
    return df


def add_all_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all trend indicators in one call."""
    df = add_emas(df)
    df = add_smas(df)
    df = add_adx(df)
    df = add_supertrend(df)
    df = add_ma_ribbon(df)
    df = add_ema_slope(df, "ema21", 5)
    df = add_ema_slope(df, "ema55", 10)
    df = add_ema_slope(df, "ema200", 20)
    df = add_trend_maturity(df)
    df = add_trend_quality(df)
    return df
