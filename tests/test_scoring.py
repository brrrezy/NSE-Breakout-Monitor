"""
Tests for the scoring engine and risk manager.
"""

import numpy as np
import pandas as pd
import pytest


def _make_latest_series() -> pd.Series:
    """Create a mock 'latest' row with all expected indicator columns."""
    return pd.Series({
        "Close": 250.0, "Open": 245.0, "High": 255.0, "Low": 244.0,
        "Volume": 2000000,
        "ema8": 248.0, "ema21": 245.0, "ema55": 240.0, "ema144": 230.0,
        "ema200": 220.0,
        "adx": 28.0, "plus_di": 30.0, "minus_di": 15.0,
        "supertrend_dir": 1,
        "ribbon_score": 0.85, "ribbon_bullish": True,
        "ema21_slope": 0.02, "ema55_slope": 0.01,
        "trend_maturity": 15, "trend_quality": 0.8,
        "rsi": 62.0,
        "macd": 1.5, "macd_sig": 1.0, "macd_hist": 0.5,
        "macd_hist_prev": 0.3, "macd_hist_accel": True,
        "stoch_k": 65.0, "stoch_d": 60.0,
        "stoch_rsi_k": 0.7, "stoch_rsi_d": 0.65,
        "cci": 80.0, "roc_10": 3.5, "roc_20": 5.0,
        "williams_r": -30.0, "mom_accel": 1.0,
        "ud_ratio": 1.6, "pct_chg": 0.02,
        "bullish_divergence": False, "bearish_divergence": False,
        "obv": 50000000, "obv_slope": 0.03,
        "cmf": 0.15, "mfi": 65.0,
        "rvol": 1.8, "vol_ma50": 1200000,
        "accum_days_20": 8, "distrib_days_20": 3, "net_accum_20": 5,
        "buying_pressure_10": 0.3,
        "turnover_ma50": 300000000,
        "close_range_5": 1.2, "close_range_20": 3.5,
        "atr": 5.0, "atr_ma20": 6.0,
        "atr_expanding": False, "atr_contracting": True,
        "bb_upper": 260.0, "bb_lower": 240.0,
        "kc_upper": 258.0, "kc_lower": 242.0,
        "squeeze_on": True, "squeeze_fire": False, "squeeze_count": 5,
        "vol_percentile": 25.0, "nr7": False, "inside_bar": False,
        "adr": 8.0, "high_120": 260.0,
    })


def _make_ohlcv_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    base = np.cumsum(np.random.randn(n) * 0.5 + 0.3) + 100
    base = np.maximum(base, 10)
    return pd.DataFrame({
        "Open": base + np.random.randn(n) * 0.5,
        "High": base + np.abs(np.random.randn(n)) * 2,
        "Low": base - np.abs(np.random.randn(n)) * 2,
        "Close": base,
        "Volume": np.random.randint(100000, 5000000, n).astype(float),
    })


class TestScorer:

    def test_score_technical(self):
        from scoring.scorer import score_technical
        latest = _make_latest_series()
        df = _make_ohlcv_df()
        score = score_technical(latest, df)
        assert 0 <= score <= 100

    def test_score_momentum(self):
        from scoring.scorer import score_momentum
        latest = _make_latest_series()
        score = score_momentum(latest)
        assert 0 <= score <= 100

    def test_score_volume(self):
        from scoring.scorer import score_volume
        latest = _make_latest_series()
        score = score_volume(latest)
        assert 0 <= score <= 100

    def test_score_trend(self):
        from scoring.scorer import score_trend
        latest = _make_latest_series()
        df = _make_ohlcv_df()
        score = score_trend(latest, df)
        assert 0 <= score <= 100

    def test_score_volatility(self):
        from scoring.scorer import score_volatility
        latest = _make_latest_series()
        score = score_volatility(latest)
        assert 0 <= score <= 100
        # Squeeze on should give higher score
        assert score > 30  # squeeze_on = True in mock data

    def test_score_base_quality(self):
        from scoring.scorer import score_base_quality
        base_info = {
            "base_len": 10, "has_vcp": True, "vol_coil": True,
            "tightening": True, "has_sfp": True, "doji_count": 3,
            "net_accum": 5, "squeeze_active": True,
            "has_nr7": False, "inside_bar_count": 1,
            "base_vol_ratio": 0.4,
        }
        score = score_base_quality(base_info)
        assert 0 <= score <= 100
        assert score >= 50  # Good base should score high

    def test_score_news_neutral(self):
        from scoring.scorer import score_news
        score = score_news({})
        assert score == 50.0  # Neutral when no news

    def test_composite_score(self):
        from scoring.scorer import compute_composite_score
        latest = _make_latest_series()
        df = _make_ohlcv_df()
        base_info = {
            "base_len": 10, "has_vcp": True, "vol_coil": False,
            "tightening": False, "has_sfp": False, "doji_count": 1,
            "net_accum": 2, "squeeze_active": False,
            "base_vol_ratio": 0.6,
        }
        scores = compute_composite_score(
            df=df, latest=latest, base_info=base_info,
            patterns=[], strength_data={"rs_composite": 0.05, "beta": 1.1},
            news_data={}, risk_data={"risk_pct": 0.025, "rr": 2.0},
            market_context={"market_score": 70},
        )
        assert "composite" in scores
        assert 0 <= scores["composite"] <= 100
        assert "confidence" in scores


class TestRanker:

    def test_rank_candidates(self):
        from scoring.ranker import rank_candidates
        candidates = [
            {"symbol": "A.NS", "status": "WATCHLIST",
             "scores": {"composite": 72, "breakout_probability": 65}},
            {"symbol": "B.NS", "status": "ACTIONABLE",
             "scores": {"composite": 85, "breakout_probability": 78}},
            {"symbol": "C.NS", "status": "WATCHLIST",
             "scores": {"composite": 60, "breakout_probability": 55}},
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0]["symbol"] == "B.NS"  # Highest composite
        assert ranked[0]["rank"] == 1

    def test_separate_actionable(self):
        from scoring.ranker import separate_actionable_watchlist
        candidates = [
            {"status": "ACTIONABLE", "symbol": "A"},
            {"status": "WATCHLIST", "symbol": "B"},
            {"status": "ACTIONABLE", "symbol": "C"},
        ]
        act, wl = separate_actionable_watchlist(candidates)
        assert len(act) == 2
        assert len(wl) == 1


class TestRiskManager:

    def test_failure_conditions(self):
        from scoring.risk_manager import generate_failure_conditions
        candidate = {
            "scores": {"volume": 35, "momentum": 35, "market": 35,
                       "relative_strength": 35},
            "base_info": {"base_vol_ratio": 0.95, "has_vcp": False},
            "liquidity": {"equal_highs": 4},
        }
        conditions = generate_failure_conditions(candidate)
        assert len(conditions) > 0

    def test_monitoring_frequency(self):
        from scoring.risk_manager import suggest_monitoring_frequency
        c_high = {"status": "WATCHLIST",
                  "scores": {"composite": 85}}
        assert "Hourly" in suggest_monitoring_frequency(c_high)

        c_action = {"status": "ACTIONABLE",
                    "scores": {"composite": 70}}
        assert "15 min" in suggest_monitoring_frequency(c_action)
