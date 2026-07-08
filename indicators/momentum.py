"""
Momentum Indicators.

Computes:
  - RSI (14) with divergence detection
  - MACD (12, 26, 9) with histogram analysis
  - Stochastic RSI (14, 14, 3, 3)
  - CCI (20)
  - ROC (Rate of Change, 10 and 20)
  - Williams %R
  - Momentum acceleration (ROC of ROC)
  - U/D ratio (up days vs down days)
  - Momentum divergence scanner
"""

import numpy as np
import pandas as pd
import ta


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI with overbought/oversold zones."""
    df["rsi"] = ta.momentum.rsi(df["Close"], window=period)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    """MACD line, signal, and histogram."""
    macd = ta.trend.MACD(df["Close"], window_fast=fast,
                         window_slow=slow, window_sign=signal)
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # MACD histogram direction (accelerating or decelerating)
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["macd_hist_accel"] = df["macd_hist"] > df["macd_hist_prev"]
    return df


def add_stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Stochastic %K and %D."""
    df["stoch_k"] = ta.momentum.stoch(
        df["High"], df["Low"], df["Close"], window=period)
    df["stoch_d"] = ta.momentum.stoch_signal(
        df["High"], df["Low"], df["Close"], window=period)
    return df


def add_stochastic_rsi(df: pd.DataFrame, rsi_period: int = 14,
                       stoch_period: int = 14,
                       k_period: int = 3,
                       d_period: int = 3) -> pd.DataFrame:
    """Stochastic RSI."""
    df["stoch_rsi"] = ta.momentum.stochrsi(
        df["Close"], window=rsi_period, smooth1=k_period,
        smooth2=d_period)
    df["stoch_rsi_k"] = ta.momentum.stochrsi_k(
        df["Close"], window=rsi_period, smooth1=k_period,
        smooth2=d_period)
    df["stoch_rsi_d"] = ta.momentum.stochrsi_d(
        df["Close"], window=rsi_period, smooth1=k_period,
        smooth2=d_period)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    df["cci"] = ta.trend.cci(
        df["High"], df["Low"], df["Close"], window=period)
    return df


def add_roc(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of Change (10 and 20 period)."""
    df["roc_10"] = ta.momentum.roc(df["Close"], window=10)
    df["roc_20"] = ta.momentum.roc(df["Close"], window=20)
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R."""
    df["williams_r"] = ta.momentum.williams_r(
        df["High"], df["Low"], df["Close"], lbp=period)
    return df


def add_momentum_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum acceleration: rate of change of the rate of change.
    Positive = momentum increasing. Negative = decelerating.
    """
    if "roc_10" not in df.columns:
        df = add_roc(df)
    df["mom_accel"] = df["roc_10"] - df["roc_10"].shift(5)
    return df


def add_ud_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Up/Down ratio: count of up days vs down days over N sessions.
    Values > 1.0 indicate bullish bias.
    """
    c = df["Close"]
    up = (c > c.shift(1)).astype(int)
    dn = (c < c.shift(1)).astype(int)
    df["ud_ratio"] = up.rolling(period).sum() / dn.rolling(
        period).sum().clip(lower=1)
    return df


def add_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage change."""
    df["pct_chg"] = df["Close"].pct_change()
    return df


def add_momentum_divergence(df: pd.DataFrame,
                            lookback: int = 20) -> pd.DataFrame:
    """
    Detect bullish and bearish divergence between price and RSI.

    Bullish: price makes lower low, RSI makes higher low.
    Bearish: price makes higher high, RSI makes lower high.
    """
    if "rsi" not in df.columns:
        df = add_rsi(df)

    n = len(df)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)

    price = df["Close"].values
    rsi = df["rsi"].values

    for i in range(lookback, n):
        window_price = price[i - lookback:i + 1]
        window_rsi = rsi[i - lookback:i + 1]

        if np.any(np.isnan(window_rsi)):
            continue

        # Find swing lows in price
        price_lows_idx = []
        for j in range(1, len(window_price) - 1):
            if (window_price[j] < window_price[j - 1] and
                    window_price[j] < window_price[j + 1]):
                price_lows_idx.append(j)

        # Bullish divergence: last two swing lows
        if len(price_lows_idx) >= 2:
            prev_idx = price_lows_idx[-2]
            curr_idx = price_lows_idx[-1]
            if (window_price[curr_idx] < window_price[prev_idx] and
                    window_rsi[curr_idx] > window_rsi[prev_idx]):
                bull_div[i] = True

        # Find swing highs in price
        price_highs_idx = []
        for j in range(1, len(window_price) - 1):
            if (window_price[j] > window_price[j - 1] and
                    window_price[j] > window_price[j + 1]):
                price_highs_idx.append(j)

        # Bearish divergence: last two swing highs
        if len(price_highs_idx) >= 2:
            prev_idx = price_highs_idx[-2]
            curr_idx = price_highs_idx[-1]
            if (window_price[curr_idx] > window_price[prev_idx] and
                    window_rsi[curr_idx] < window_rsi[prev_idx]):
                bear_div[i] = True

    df["bullish_divergence"] = bull_div
    df["bearish_divergence"] = bear_div
    return df


def add_all_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all momentum indicators in one call."""
    df = add_rsi(df)
    df = add_macd(df)
    df = add_stochastic(df)
    df = add_stochastic_rsi(df)
    df = add_cci(df)
    df = add_roc(df)
    df = add_williams_r(df)
    df = add_momentum_acceleration(df)
    df = add_ud_ratio(df)
    df = add_pct_change(df)
    df = add_momentum_divergence(df)
    return df
