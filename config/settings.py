"""
Centralized configuration for NSE Breakout Detection Engine.
All tunable parameters live here. Values can be overridden via config.yaml.
"""

import os
import datetime as dt
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ============================================================
# DEFAULTS
# ============================================================

_DEFAULTS = {
    # ── Data ──
    "data_period": "2y",
    "data_interval": "1d",
    "min_data_days": 150,
    "cache_ttl_hours": 24,
    "chunk_size": 200,
    "max_workers": 8,

    # ── Universe ──
    "min_price": 50,
    "max_price": 5000,
    "min_turnover_cr": 10,          # Avg daily turnover in ₹ Crores

    # ── EMA Periods ──
    "ema_periods": [8, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200],
    "sma_periods": [50, 200],

    # ── Phase 1: Uptrend ──
    "uptrend_ema_stack": [8, 21, 55],       # Must be in descending order
    "uptrend_ema_long": 144,
    "uptrend_ema_long_lookback": 20,
    "uptrend_ema_long_tolerance": 0.99,
    "uptrend_ud_ratio_min": 1.4,
    "uptrend_proximity_pct": 0.85,          # Within 15% of 120-day high
    "uptrend_rsi_max": 85,
    "uptrend_extension_ema21": 1.15,
    "uptrend_extension_ema144": 2.0,

    # ── Phase 2: Base ──
    "base_min_days": 5,
    "base_max_days": 15,
    "base_small_candle_pct": 0.65,
    "base_small_candle_adr_mult": 1.2,
    "base_vol_dryup_mult": 0.75,
    "base_price_near_pivot_pct": 0.92,
    "base_supply_bar_wick_ratio": 3,
    "base_supply_bar_vol_mult": 1.5,
    "base_max_consecutive_down": 3,
    "base_distribution_vol_mult": 1.5,
    "base_vcp_contraction_pct": 0.80,
    "base_tightening_ratio": 0.5,
    "base_net_accum_threshold": 3,

    # ── Phase 3: Breakout ──
    "breakout_body_pct": 0.60,
    "breakout_close_top_pct": 0.08,
    "breakout_rvol_min": 2.0,
    "breakout_move_min_pct": 0.045,
    "breakout_no_chase_days": 3,

    # ── Confluence ──
    "confluence_min": 4,
    "confluence_checks": [
        "macd_hist_positive",
        "rsi_above_55",
        "stoch_above_50",
        "rvol_above_1_2",
        "ema_stack_aligned",
        "adx_above_20",
    ],

    # ── Risk Management ──
    "risk_max_pct": 0.03,
    "risk_rr_min": 1.5,
    "risk_max_stop_pct": 0.05,
    "risk_position_risk_pct": 0.0075,       # 0.75% of capital per trade

    # ── Scoring Weights ──
    "scoring_weights": {
        "technical": 0.10,
        "liquidity": 0.08,
        "momentum": 0.10,
        "volume": 0.10,
        "trend": 0.10,
        "relative_strength": 0.08,
        "pattern": 0.08,
        "news": 0.06,
        "risk": 0.08,
        "market": 0.05,
        "volatility": 0.05,
        "base_quality": 0.07,
        "breakout_probability": 0.05,
    },
    "scoring_shortlist_threshold": 65,
    "scoring_top_n": 15,

    # ── News Engine ──
    "news_enabled": True,
    "news_cache_ttl_minutes": 60,
    "news_max_rps": 2,
    "news_time_decay_halflife_days": 3,
    "groq_model": "llama-3.3-70b-versatile",
    "groq_max_tokens": 500,
    "groq_temperature": 0.2,
    "groq_verdict_max_tokens": 150,

    # ── Adaptive Learning ──
    "learning_enabled": True,
    "learning_db_name": "scan_history.db",
    "learning_outcome_check_days": 5,
    "learning_min_samples_optimize": 50,
    "learning_max_weight_change": 0.20,

    # ── Alerts ──
    "telegram_message_max_len": 4096,

    # ── Market Context ──
    "nifty_symbol": "^NSEI",
    "nifty_bank_symbol": "^NSEBANK",
    "india_vix_symbol": "^INDIAVIX",
    "vix_high_threshold": 20,
    "vix_extreme_threshold": 30,

    # ── Feature Flags ──
    "enable_news_engine": True,
    "enable_learning": True,
    "enable_liquidity_analysis": True,
    "enable_pattern_detection": True,
    "enable_market_context": True,
}


# ============================================================
# SETTINGS CLASS
# ============================================================

class Settings:
    """
    Global settings singleton.
    Loads defaults, then overlays config.yaml, then env vars.
    """

    _instance: Optional["Settings"] = None

    def __init__(self):
        self._data: Dict = dict(_DEFAULTS)
        self._load_yaml()
        self._load_env()
        self._compute_derived()

    @classmethod
    def get(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reload(cls) -> "Settings":
        cls._instance = None
        return cls.get()

    # ── Loaders ──

    def _load_yaml(self):
        """Overlay settings from config.yaml if it exists."""
        config_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
        ]
        for p in config_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        user_cfg = yaml.safe_load(f) or {}
                    self._deep_merge(self._data, user_cfg)
                    print(f"[CONFIG] Loaded {p}")
                    return
                except Exception as e:
                    print(f"[CONFIG] Warning: failed to load {p}: {e}")

    def _load_env(self):
        """Load secrets and overrides from environment variables."""
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")

        # Allow env overrides for common settings
        if os.environ.get("NSE_MIN_PRICE"):
            self._data["min_price"] = int(os.environ["NSE_MIN_PRICE"])
        if os.environ.get("NSE_MAX_PRICE"):
            self._data["max_price"] = int(os.environ["NSE_MAX_PRICE"])

    def _compute_derived(self):
        """Compute values derived from config."""
        cache_base = os.environ.get(
            "NSE_SCREENER_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "nse_screener_cache"))
        self.cache_dir = Path(cache_base)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ist_offset = dt.timedelta(hours=5, minutes=30)

        db_dir = self.cache_dir / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.learning_db_path = db_dir / self._data["learning_db_name"]

        self.state_file = Path("watchlist_persistent.json")
        self.min_turnover = self._data["min_turnover_cr"] * 1e7  # Crores → absolute

    # ── Accessors ──

    def __getattr__(self, name: str):
        if name.startswith("_") or name in (
            "telegram_token", "telegram_chat_id", "groq_api_key",
            "cache_dir", "ist_offset", "learning_db_path",
            "state_file", "min_turnover",
        ):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"Setting '{name}' not found. Check config/settings.py defaults.")

    def __getitem__(self, key: str):
        return self._data[key]

    def get_val(self, key: str, default=None):
        return self._data.get(key, default)

    # ── Helpers ──

    @staticmethod
    def _deep_merge(base: dict, overlay: dict):
        """Recursively merge overlay into base."""
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Settings._deep_merge(base[k], v)
            else:
                base[k] = v

    def get_now_ist(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc) + self.ist_offset

    def to_dict(self) -> dict:
        return dict(self._data)
