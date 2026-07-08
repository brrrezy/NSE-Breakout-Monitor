"""
Tests for indicator modules.

Uses synthetic OHLCV data to verify indicator calculations.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if trend == "up":
        base = np.cumsum(np.random.randn(n) * 0.5 + 0.3) + 100
    elif trend == "down":
        base = np.cumsum(np.random.randn(n) * 0.5 - 0.3) + 200
    else:
        base = np.random.randn(n) * 2 + 150

    base = np.maximum(base, 10)  # Floor at 10
    high = base + np.abs(np.random.randn(n)) * 2
    low = base - np.abs(np.random.randn(n)) * 2
    low = np.maximum(low, 1)
    open_ = base + np.random.randn(n) * 0.5
    close = base
    volume = np.random.randint(100000, 5000000, n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


# ============================================================
# TREND INDICATORS
# ============================================================

class TestTrendIndicators:

    def test_add_emas(self):
        from indicators.trend import add_emas
        df = _make_ohlcv()
        result = add_emas(df)
        assert "ema8" in result.columns
        assert "ema21" in result.columns
        assert "ema55" in result.columns
        assert "ema200" in result.columns
        # EMA should not be all NaN
        assert result["ema21"].dropna().shape[0] > 0

    def test_add_adx(self):
        from indicators.trend import add_adx
        df = _make_ohlcv()
        result = add_adx(df)
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
        # ADX should be between 0 and 100
        valid = result["adx"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_supertrend(self):
        from indicators.trend import add_supertrend
        df = _make_ohlcv()
        result = add_supertrend(df)
        assert "supertrend" in result.columns
        assert "supertrend_dir" in result.columns
        # Direction should be 1 or -1
        dirs = result["supertrend_dir"].dropna().unique()
        assert set(dirs).issubset({-1, 1})

    def test_add_ma_ribbon(self):
        from indicators.trend import add_ma_ribbon
        df = _make_ohlcv()
        result = add_ma_ribbon(df)
        assert "ribbon_score" in result.columns
        assert "ribbon_bullish" in result.columns
        valid = result["ribbon_score"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_add_trend_maturity(self):
        from indicators.trend import add_emas, add_trend_maturity
        df = add_emas(_make_ohlcv())
        result = add_trend_maturity(df)
        assert "trend_maturity" in result.columns
        assert (result["trend_maturity"] >= 0).all()

    def test_all_trend_indicators(self):
        from indicators.trend import add_all_trend_indicators
        df = _make_ohlcv()
        result = add_all_trend_indicators(df)
        expected = ["ema8", "ema21", "adx", "supertrend",
                    "ribbon_score", "trend_maturity", "trend_quality"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


# ============================================================
# MOMENTUM INDICATORS
# ============================================================

class TestMomentumIndicators:

    def test_add_rsi(self):
        from indicators.momentum import add_rsi
        df = _make_ohlcv()
        result = add_rsi(df)
        assert "rsi" in result.columns
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_macd(self):
        from indicators.momentum import add_macd
        df = _make_ohlcv()
        result = add_macd(df)
        assert "macd" in result.columns
        assert "macd_sig" in result.columns
        assert "macd_hist" in result.columns

    def test_add_stochastic_rsi(self):
        from indicators.momentum import add_stochastic_rsi
        df = _make_ohlcv()
        result = add_stochastic_rsi(df)
        assert "stoch_rsi" in result.columns

    def test_add_divergence(self):
        from indicators.momentum import add_momentum_divergence
        df = _make_ohlcv()
        result = add_momentum_divergence(df)
        assert "bullish_divergence" in result.columns
        assert "bearish_divergence" in result.columns

    def test_all_momentum_indicators(self):
        from indicators.momentum import add_all_momentum_indicators
        df = _make_ohlcv()
        result = add_all_momentum_indicators(df)
        expected = ["rsi", "macd", "stoch_rsi", "cci", "roc_10",
                    "ud_ratio", "pct_chg"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


# ============================================================
# VOLUME INDICATORS
# ============================================================

class TestVolumeIndicators:

    def test_add_obv(self):
        from indicators.volume import add_obv
        df = _make_ohlcv()
        result = add_obv(df)
        assert "obv" in result.columns
        assert "obv_slope" in result.columns

    def test_add_relative_volume(self):
        from indicators.volume import add_relative_volume
        df = _make_ohlcv()
        result = add_relative_volume(df)
        assert "rvol" in result.columns
        valid = result["rvol"].dropna()
        assert (valid >= 0).all()

    def test_add_volume_profile(self):
        from indicators.volume import add_volume_profile
        df = _make_ohlcv()
        result = add_volume_profile(df)
        assert "vp_poc" in result.columns
        assert "vp_vah" in result.columns

    def test_all_volume_indicators(self):
        from indicators.volume import add_all_volume_indicators
        df = _make_ohlcv()
        result = add_all_volume_indicators(df)
        expected = ["obv", "cmf", "mfi", "vwap", "rvol",
                    "accum_days_20", "turnover_ma50"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"


# ============================================================
# VOLATILITY INDICATORS
# ============================================================

class TestVolatilityIndicators:

    def test_add_atr(self):
        from indicators.volatility import add_atr
        df = _make_ohlcv()
        result = add_atr(df)
        assert "atr" in result.columns
        valid = result["atr"].dropna()
        assert (valid >= 0).all()

    def test_add_squeeze(self):
        from indicators.volatility import add_squeeze
        df = _make_ohlcv()
        result = add_squeeze(df)
        assert "squeeze_on" in result.columns
        assert "squeeze_fire" in result.columns

    def test_add_nr7(self):
        from indicators.volatility import add_nr7
        df = _make_ohlcv()
        result = add_nr7(df)
        assert "nr7" in result.columns

    def test_all_volatility_indicators(self):
        from indicators.volatility import add_all_volatility_indicators
        df = _make_ohlcv()
        result = add_all_volatility_indicators(df)
        expected = ["atr", "bb_upper", "kc_upper", "dc_upper",
                    "squeeze_on", "vol_percentile", "nr7", "inside_bar"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"
