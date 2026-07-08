"""
Tests for the core engine components.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSettings:

    def test_settings_singleton(self):
        from config.settings import Settings
        s1 = Settings.get()
        s2 = Settings.get()
        assert s1 is s2

    def test_defaults_loaded(self):
        from config.settings import Settings
        cfg = Settings.reload()
        assert cfg.min_price == 50
        assert cfg.data_period == "2y"
        assert isinstance(cfg.scoring_weights, dict)
        assert len(cfg.scoring_weights) > 0

    def test_scoring_weights_sum_to_one(self):
        from config.settings import Settings
        cfg = Settings.get()
        total = sum(cfg.scoring_weights.values())
        assert abs(total - 1.0) < 0.01


class TestStateManager:

    def test_default_state(self):
        from core.universe import StateManager
        # Patch state file to non-existent path
        with patch("config.settings.Settings.get") as mock:
            from pathlib import Path
            mock_cfg = MagicMock()
            mock_cfg.state_file = Path("__test_nonexistent__.json")
            mock.return_value = mock_cfg
            sm = StateManager()
            assert sm.watchlist == []
            assert sm.eod_date == ""


class TestBreakoutModule:

    def test_confluence_score(self):
        import pandas as pd
        from analysis.breakout import confluence_score
        latest = pd.Series({
            "macd_hist": 0.5,
            "rsi": 60,
            "stoch_k": 55,
            "rvol": 1.5,
            "ema8": 100, "ema21": 98, "ema55": 95,
            "adx": 25,
        })
        score = confluence_score(latest)
        assert score == 6  # All 6 pass

    def test_confluence_score_weak(self):
        import pandas as pd
        from analysis.breakout import confluence_score
        latest = pd.Series({
            "macd_hist": -0.5,
            "rsi": 40,
            "stoch_k": 30,
            "rvol": 0.5,
            "ema8": 95, "ema21": 98, "ema55": 100,
            "adx": 10,
        })
        score = confluence_score(latest)
        assert score == 0  # All fail

    def test_calc_risk(self):
        from analysis.breakout import calc_risk
        result = calc_risk(100.0, 97.0)
        assert result["stop"] == 97.0
        assert result["target_1"] == 106.0  # 2R
        assert result["target_2"] == 109.0  # 3R
        assert result["rr"] == 2.0
        assert result["risk_ok"] is True  # 3% risk


class TestAlertFormatter:

    def test_format_breakout_telegram(self):
        from alerts.formatter import format_breakout_telegram
        candidate = {
            "symbol": "RELIANCE.NS",
            "status": "ACTIONABLE",
            "price": 2500.0,
            "pivot": 2480.0,
            "rvol": 2.5,
            "scores": {"composite": 82.5, "breakout_probability": 75},
            "risk": {"stop": 2430, "target_1": 2570,
                     "target_2": 2640, "target_3": 2850, "rr": 2.0},
            "base_info": {
                "base_len": 10, "has_sfp": True, "doji_count": 2,
                "has_vcp": True, "vol_coil": False,
                "tightening": True, "net_accum": 4,
                "squeeze_active": True,
            },
            "strength": {"rs_composite": 0.08},
            "patterns": [],
        }
        msg = format_breakout_telegram(candidate)
        assert "RELIANCE" in msg
        assert "BREAKOUT" in msg
        assert "2500" in msg
        assert "VCP" in msg

    def test_format_watchlist(self):
        from alerts.formatter import format_watchlist_telegram
        candidates = [
            {
                "symbol": "TCS.NS", "price": 3500.0, "pivot": 3600.0,
                "scores": {"composite": 70},
                "base_info": {"base_len": 8, "has_vcp": True,
                              "vol_coil": False, "tightening": False,
                              "squeeze_active": False},
            }
        ]
        msg = format_watchlist_telegram(candidates)
        assert "TCS" in msg
        assert "WATCHLIST" in msg
