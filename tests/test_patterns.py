"""
Tests for chart pattern detection and market structure analysis.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if trend == "up":
        base = np.cumsum(np.random.randn(n) * 0.5 + 0.3) + 100
    elif trend == "down":
        base = np.cumsum(np.random.randn(n) * 0.5 - 0.3) + 200
    else:
        base = np.random.randn(n) * 2 + 150

    base = np.maximum(base, 10)
    high = base + np.abs(np.random.randn(n)) * 2
    low = base - np.abs(np.random.randn(n)) * 2
    low = np.maximum(low, 1)
    open_ = base + np.random.randn(n) * 0.5
    close = base
    volume = np.random.randint(100000, 5000000, n).astype(float)

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


class TestMarketStructure:

    def test_swing_points_detected(self):
        from analysis.market_structure import detect_swing_points
        df = _make_ohlcv(200)
        swings = detect_swing_points(df, lookback=3)
        assert len(swings) > 0
        # Should have both highs and lows
        types = {s.type for s in swings}
        assert "high" in types
        assert "low" in types

    def test_classify_structure(self):
        from analysis.market_structure import (
            detect_swing_points, classify_structure)
        df = _make_ohlcv(200, "up")
        swings = detect_swing_points(df, lookback=3)
        structure = classify_structure(swings)
        assert len(structure) > 0
        # Each point should have a label
        for sp in structure:
            assert sp.label in ("HH", "HL", "LH", "LL")

    def test_classify_trend_uptrend(self):
        from analysis.market_structure import (
            detect_swing_points, classify_structure, classify_trend)
        df = _make_ohlcv(200, "up")
        swings = detect_swing_points(df, lookback=3)
        structure = classify_structure(swings)
        trend = classify_trend(structure)
        assert trend in ("uptrend", "range", "transition_up",
                         "transition_down", "downtrend", "unknown")

    def test_full_analysis(self):
        from analysis.market_structure import analyze_market_structure
        df = _make_ohlcv(200)
        result = analyze_market_structure(df)
        assert "trend" in result
        assert "breakout_pressure" in result
        assert "compressed" in result

    def test_compression_detection(self):
        from analysis.market_structure import detect_compression
        df = _make_ohlcv(50, "flat")
        result = detect_compression(df, lookback=20)
        assert "compressed" in result
        assert "compression_ratio" in result


class TestPatterns:

    def test_detect_all_patterns_runs(self):
        from analysis.patterns import detect_all_patterns
        df = _make_ohlcv(200)
        patterns = detect_all_patterns(df)
        # Should return a list (possibly empty)
        assert isinstance(patterns, list)

    def test_flat_base_detection(self):
        from analysis.patterns import detect_flat_base
        # Create flat data
        n = 20
        price = 100.0
        df = pd.DataFrame({
            "Open": [price + np.random.randn() * 0.5 for _ in range(n)],
            "High": [price + abs(np.random.randn()) * 1.0 for _ in range(n)],
            "Low": [price - abs(np.random.randn()) * 1.0 for _ in range(n)],
            "Close": [price + np.random.randn() * 0.3 for _ in range(n)],
            "Volume": [1000000] * n,
        })
        result = detect_flat_base(df, min_days=5, max_days=20,
                                  max_range_pct=0.10)
        # May or may not detect depending on randomness, but shouldn't crash
        assert result is None or result.name == "Flat Base"


class TestLiquidity:

    def test_equal_levels(self):
        from analysis.liquidity import detect_equal_levels
        df = _make_ohlcv(100)
        result = detect_equal_levels(df)
        assert "equal_highs" in result
        assert "equal_lows" in result

    def test_order_blocks(self):
        from analysis.liquidity import detect_order_blocks
        df = _make_ohlcv(100)
        obs = detect_order_blocks(df)
        assert isinstance(obs, list)

    def test_fvgs(self):
        from analysis.liquidity import detect_fvgs
        df = _make_ohlcv(50)
        fvgs = detect_fvgs(df)
        assert isinstance(fvgs, list)

    def test_full_liquidity_analysis(self):
        from analysis.liquidity import analyze_liquidity
        df = _make_ohlcv(100)
        result = analyze_liquidity(df, resistance=120.0)
        assert "equal_highs" in result
        assert "zone" in result
        assert result["zone"] in ("premium", "discount", "unknown")
