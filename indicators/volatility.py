"""
Volatility Indicators.

Computes:
  - ATR with expansion/contraction detection
  - Bollinger Bands with squeeze detection
  - Keltner Channels
  - Donchian Channels
  - BB-inside-Keltner squeeze indicator
  - Volatility percentile (current vs 1-year range)
  - Historical volatility (20-day)
  - NR7 (Narrowest Range of 7 days)
  - Inside Bar detection
  - ADR (Average Day Range)
"""

import numpy as np
import pandas as pd
import ta


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR with expansion and contraction detection."""
    df["atr"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=period)

    # ATR moving average for comparison
    df["atr_ma20"] = df["atr"].rolling(20).mean()

    # Expansion: ATR > 1.5x its 20-day average
    df["atr_expanding"] = df["atr"] > df["atr_ma20"] * 1.5
    # Contraction: ATR < 0.7x its 20-day average
    df["atr_contracting"] = df["atr"] < df["atr_ma20"] * 0.7
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with bandwidth and %B."""
    bb = ta.volatility.BollingerBands(
        df["Close"], window=period, window_dev=std_dev)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct_b"] = bb.bollinger_pband()
    return df


def add_keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                         atr_period: int = 14,
                         atr_mult: float = 1.5) -> pd.DataFrame:
    """Keltner Channels."""
    kc = ta.volatility.KeltnerChannel(
        df["High"], df["Low"], df["Close"],
        window=ema_period, window_atr=atr_period,
        multiplier=atr_mult)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_middle"] = kc.keltner_channel_mband()
    df["kc_lower"] = kc.keltner_channel_lband()
    return df


def add_donchian_channels(df: pd.DataFrame,
                          period: int = 20) -> pd.DataFrame:
    """Donchian Channels (highest high, lowest low)."""
    dc = ta.volatility.DonchianChannel(
        df["High"], df["Low"], df["Close"], window=period)
    df["dc_upper"] = dc.donchian_channel_hband()
    df["dc_middle"] = dc.donchian_channel_mband()
    df["dc_lower"] = dc.donchian_channel_lband()
    df["dc_width"] = dc.donchian_channel_wband()
    return df


def add_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Squeeze indicator: Bollinger Bands inside Keltner Channels.
    When BB is inside KC, volatility is compressed → squeeze is ON.
    When squeeze releases (BB expands outside KC), expect a move.
    """
    if "bb_upper" not in df.columns:
        df = add_bollinger_bands(df)
    if "kc_upper" not in df.columns:
        df = add_keltner_channels(df)

    df["squeeze_on"] = (
        (df["bb_upper"] < df["kc_upper"]) &
        (df["bb_lower"] > df["kc_lower"])
    )

    # Squeeze fire: squeeze was on, now releasing
    df["squeeze_fire"] = (
        df["squeeze_on"].shift(1).fillna(False) &
        ~df["squeeze_on"]
    )

    # Count consecutive squeeze days
    squeeze_int = df["squeeze_on"].astype(int)
    squeeze_count = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if squeeze_int.iloc[i]:
            squeeze_count[i] = squeeze_count[i - 1] + 1
    df["squeeze_count"] = squeeze_count

    return df


def add_volatility_percentile(df: pd.DataFrame,
                              lookback: int = 252) -> pd.DataFrame:
    """
    Current ATR percentile vs 1-year range.
    0 = lowest volatility, 100 = highest volatility.
    """
    if "atr" not in df.columns:
        df = add_atr(df)

    def _percentile(series):
        """Rank current value within the rolling window."""
        n = len(series)
        if n < 2 or pd.isna(series.iloc[-1]):
            return np.nan
        val = series.iloc[-1]
        rank = (series < val).sum()
        return rank / (n - 1) * 100

    df["vol_percentile"] = df["atr"].rolling(
        lookback, min_periods=50).apply(_percentile, raw=False)
    return df


def add_historical_volatility(df: pd.DataFrame,
                              period: int = 20) -> pd.DataFrame:
    """Historical (realized) volatility: annualized std of log returns."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["hist_vol"] = log_ret.rolling(period).std() * np.sqrt(252)
    return df


def add_nr7(df: pd.DataFrame) -> pd.DataFrame:
    """
    NR7: Narrowest Range of 7 days.
    Today's range is the smallest of the last 7 days.
    """
    rng = df["High"] - df["Low"]
    min_range_7 = rng.rolling(7).min()
    df["nr7"] = rng == min_range_7
    return df


def add_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inside bar: today's high < yesterday's high AND
    today's low > yesterday's low.
    """
    df["inside_bar"] = (
        (df["High"] < df["High"].shift(1)) &
        (df["Low"] > df["Low"].shift(1))
    )
    return df


def add_adr(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Average Day Range (mean of High-Low over N days)."""
    df["adr"] = (df["High"] - df["Low"]).rolling(period).mean()
    return df


def add_high_120(df: pd.DataFrame) -> pd.DataFrame:
    """120-day rolling high."""
    df["high_120"] = df["High"].rolling(120).max()
    return df


def add_all_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all volatility indicators in one call."""
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_keltner_channels(df)
    df = add_donchian_channels(df)
    df = add_squeeze(df)
    df = add_volatility_percentile(df)
    df = add_historical_volatility(df)
    df = add_nr7(df)
    df = add_inside_bar(df)
    df = add_adr(df)
    df = add_high_120(df)
    return df
