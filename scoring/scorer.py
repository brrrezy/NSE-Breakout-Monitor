"""
Multi-Dimensional Scoring Engine.

Computes 13 individual dimension scores (0-100 each)
and a weighted Composite Institutional Score (0-100).
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import Settings


def _clamp(val: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, val))


# ============================================================
# INDIVIDUAL DIMENSION SCORERS
# ============================================================

def score_technical(latest: pd.Series, df: pd.DataFrame) -> float:
    """
    Technical Score: EMA alignment, ADX strength, Supertrend direction.
    """
    s = 0.0

    # EMA stack alignment (8 > 21 > 55 > 144)
    ema_cols = ["ema8", "ema21", "ema55", "ema144"]
    vals = [latest.get(c, 0) for c in ema_cols if c in latest.index]
    if len(vals) >= 3:
        aligned = sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1])
        s += (aligned / (len(vals) - 1)) * 30

    # ADX strength
    adx = latest.get("adx", 0)
    if not pd.isna(adx):
        if adx >= 25:
            s += 25
        elif adx >= 20:
            s += 15
        elif adx >= 15:
            s += 5

    # Supertrend bullish
    if latest.get("supertrend_dir", 0) == 1:
        s += 15

    # MA Ribbon score
    ribbon = latest.get("ribbon_score", 0)
    if not pd.isna(ribbon):
        s += ribbon * 15

    # Trend maturity (longer alignment = better)
    maturity = latest.get("trend_maturity", 0)
    if not pd.isna(maturity):
        s += min(maturity / 20, 1.0) * 15

    return _clamp(s)


def score_momentum(latest: pd.Series) -> float:
    """
    Momentum Score: RSI, MACD, Stoch RSI, CCI, ROC.
    """
    s = 0.0

    # RSI zone scoring
    rsi = latest.get("rsi", 50)
    if not pd.isna(rsi):
        if 55 <= rsi <= 75:
            s += 25  # Sweet spot
        elif 45 <= rsi < 55:
            s += 10
        elif rsi > 75:
            s += 15  # Strong but potentially overbought
        elif rsi < 45:
            s += 0   # Weak

    # MACD histogram positive and accelerating
    hist = latest.get("macd_hist", 0)
    if not pd.isna(hist) and hist > 0:
        s += 20
        if latest.get("macd_hist_accel", False):
            s += 5

    # Stochastic RSI in bullish zone
    stoch_rsi = latest.get("stoch_rsi_k", 50)
    if not pd.isna(stoch_rsi) and stoch_rsi > 20:
        s += min(stoch_rsi / 100 * 20, 20)

    # CCI bullish
    cci = latest.get("cci", 0)
    if not pd.isna(cci) and cci > 0:
        s += min(cci / 200 * 15, 15)

    # ROC positive
    roc = latest.get("roc_10", 0)
    if not pd.isna(roc) and roc > 0:
        s += min(roc / 10 * 10, 10)

    # No bearish divergence penalty
    if latest.get("bearish_divergence", False):
        s -= 15

    # Bullish divergence bonus
    if latest.get("bullish_divergence", False):
        s += 10

    return _clamp(s)


def score_volume(latest: pd.Series) -> float:
    """
    Volume Score: RVol, OBV trend, CMF, MFI, accumulation.
    """
    s = 0.0

    # Relative volume
    rvol = latest.get("rvol", 1.0)
    if not pd.isna(rvol):
        if rvol >= 2.0:
            s += 25
        elif rvol >= 1.5:
            s += 20
        elif rvol >= 1.0:
            s += 10
        else:
            s += 5

    # OBV slope (rising = accumulation)
    obv_slope = latest.get("obv_slope", 0)
    if not pd.isna(obv_slope) and obv_slope > 0:
        s += min(obv_slope * 100, 20)

    # CMF positive (money flowing in)
    cmf = latest.get("cmf", 0)
    if not pd.isna(cmf) and cmf > 0:
        s += min(cmf * 100, 15)

    # MFI in bullish zone
    mfi = latest.get("mfi", 50)
    if not pd.isna(mfi) and mfi > 50:
        s += min((mfi - 50) / 50 * 15, 15)

    # Net accumulation days
    net_accum = latest.get("net_accum_20", 0)
    if not pd.isna(net_accum) and net_accum > 0:
        s += min(net_accum * 3, 15)

    # Buying pressure
    bp = latest.get("buying_pressure_10", 0)
    if not pd.isna(bp) and bp > 0:
        s += min(bp * 10, 10)

    return _clamp(s)


def score_trend(latest: pd.Series, df: pd.DataFrame) -> float:
    """
    Trend Score: EMA stack quality, slopes, maturity, acceleration.
    """
    s = 0.0

    # EMA-21 slope (rising = strong trend)
    slope_21 = latest.get("ema21_slope", 0)
    if not pd.isna(slope_21) and slope_21 > 0:
        s += min(slope_21 * 500, 25)

    # EMA-55 slope
    slope_55 = latest.get("ema55_slope", 0)
    if not pd.isna(slope_55) and slope_55 > 0:
        s += min(slope_55 * 500, 15)

    # Trend quality (consistency of higher lows)
    tq = latest.get("trend_quality", 0.5)
    if not pd.isna(tq):
        s += tq * 20

    # Trend maturity
    maturity = latest.get("trend_maturity", 0)
    if not pd.isna(maturity):
        if maturity >= 20:
            s += 20
        elif maturity >= 10:
            s += 15
        elif maturity >= 5:
            s += 10

    # U/D ratio
    ud = latest.get("ud_ratio", 1.0)
    if not pd.isna(ud):
        if ud >= 2.0:
            s += 20
        elif ud >= 1.5:
            s += 15
        elif ud >= 1.2:
            s += 10

    return _clamp(s)


def score_volatility(latest: pd.Series) -> float:
    """
    Volatility Score: Squeeze, ATR percentile, NR7, contraction quality.
    Higher score = better setup (compressed, ready to expand).
    """
    s = 0.0

    # Squeeze active (BB inside Keltner)
    if latest.get("squeeze_on", False):
        s += 30
        squeeze_count = latest.get("squeeze_count", 0)
        if not pd.isna(squeeze_count):
            s += min(squeeze_count * 2, 10)

    # Squeeze just fired (immediate catalyst)
    if latest.get("squeeze_fire", False):
        s += 20

    # ATR contracting (volatility coiling)
    if latest.get("atr_contracting", False):
        s += 15

    # Low volatility percentile (compressed)
    vol_pct = latest.get("vol_percentile", 50)
    if not pd.isna(vol_pct):
        if vol_pct < 20:
            s += 15
        elif vol_pct < 40:
            s += 10

    # NR7
    if latest.get("nr7", False):
        s += 10

    # Inside bar
    if latest.get("inside_bar", False):
        s += 5

    return _clamp(s)


def score_relative_strength(strength_data: Dict) -> float:
    """
    Relative Strength Score: RS vs index, beta-adjusted.
    """
    s = 50.0  # Start at neutral

    rs_composite = strength_data.get("rs_composite", 0)
    # Strong outperformance
    if rs_composite > 0.10:
        s += 30
    elif rs_composite > 0.05:
        s += 20
    elif rs_composite > 0:
        s += 10
    elif rs_composite < -0.05:
        s -= 20
    elif rs_composite < 0:
        s -= 10

    # Beta: prefer moderate beta for breakouts
    beta = strength_data.get("beta", 1.0)
    if 0.8 <= beta <= 1.5:
        s += 10
    elif beta > 2.0:
        s -= 10  # Too volatile

    # Short-term RS acceleration
    rs_20 = strength_data.get("rs_20d", 0)
    rs_50 = strength_data.get("rs_50d", 0)
    if rs_20 > rs_50 and rs_20 > 0:
        s += 10  # Accelerating relative strength

    return _clamp(s)


def score_pattern(patterns: list) -> float:
    """
    Pattern Score: based on detected chart patterns.
    """
    if not patterns:
        return 30.0  # Neutral — absence of pattern isn't bad

    best = patterns[0]  # Already sorted by confidence
    s = 0.0

    # Pattern confidence
    s += best.confidence * 50

    # Bullish pattern bonus
    if best.type == "bullish":
        s += 25
    elif best.type == "neutral":
        s += 10

    # Named pattern bonuses
    premium_patterns = {
        "Ascending Triangle", "Cup and Handle",
        "Bull Flag", "Darvas Box",
    }
    if best.name in premium_patterns:
        s += 15

    return _clamp(s)


def score_news(news_data: Dict) -> float:
    """
    News Score: based on sentiment analysis results.
    """
    if not news_data:
        return 50.0  # Neutral

    sentiment = news_data.get("sentiment_score", 0)
    catalyst = news_data.get("catalyst_strength", 0)
    confidence = news_data.get("confidence", 0)

    s = 50.0  # Start neutral
    s += sentiment * 30 * confidence  # Sentiment-weighted by confidence
    s += catalyst * 2                  # Catalyst boost

    return _clamp(s)


def score_risk(risk_data: Dict) -> float:
    """
    Risk Score: lower risk = higher score.
    """
    s = 50.0

    risk_pct = risk_data.get("risk_pct", 0.03)
    rr = risk_data.get("rr", 2.0)

    # Tight stop (low risk %)
    if risk_pct < 0.02:
        s += 30
    elif risk_pct < 0.03:
        s += 20
    elif risk_pct < 0.05:
        s += 10
    else:
        s -= 20

    # Good R:R ratio
    if rr >= 3.0:
        s += 20
    elif rr >= 2.0:
        s += 10

    return _clamp(s)


def score_market(market_context: Dict) -> float:
    """Market environment score."""
    return _clamp(market_context.get("market_score", 50.0))


def score_base_quality(base_info: Dict) -> float:
    """
    Base Quality Score: VCP, vol coil, tightening, etc.
    """
    s = 0.0

    base_len = base_info.get("base_len", 0)
    if 7 <= base_len <= 12:
        s += 15  # Ideal base length
    elif 5 <= base_len <= 15:
        s += 10

    if base_info.get("has_vcp"):
        s += 20
    if base_info.get("vol_coil"):
        s += 15
    if base_info.get("tightening"):
        s += 10
    if base_info.get("has_sfp"):
        s += 10
    if base_info.get("doji_count", 0) >= 2:
        s += 5
    if base_info.get("net_accum", 0) >= 3:
        s += 15
    if base_info.get("squeeze_active"):
        s += 10
    if base_info.get("has_nr7"):
        s += 5
    if base_info.get("inside_bar_count", 0) >= 2:
        s += 5

    # Volume dry-up quality
    vol_ratio = base_info.get("base_vol_ratio", 1.0)
    if vol_ratio < 0.5:
        s += 10
    elif vol_ratio < 0.75:
        s += 5

    return _clamp(s)


def score_breakout_probability(all_scores: Dict[str, float]) -> float:
    """
    Breakout Probability: meta-score based on all other scores.
    Stocks with high scores across multiple dimensions are
    more likely to break out successfully.
    """
    # Count how many scores are above 60
    high_count = sum(1 for v in all_scores.values() if v >= 60)
    # Count how many are above 40
    mid_count = sum(1 for v in all_scores.values() if v >= 40)
    total = max(len(all_scores), 1)

    s = (high_count / total * 60) + (mid_count / total * 30)

    # Average of all scores
    avg = sum(all_scores.values()) / total
    s = (s * 0.4) + (avg * 0.6)

    return _clamp(s)


# ============================================================
# COMPOSITE SCORING
# ============================================================

def compute_composite_score(
    df: pd.DataFrame,
    latest: pd.Series,
    base_info: Dict,
    patterns: list,
    strength_data: Dict,
    news_data: Dict,
    risk_data: Dict,
    market_context: Dict,
    liquidity_data: Dict = None,
) -> Dict[str, Any]:
    """
    Compute the full multi-dimensional score for a stock.

    Returns dict with all individual scores and the composite.
    """
    cfg = Settings.get()
    weights = cfg.scoring_weights

    # Compute individual scores
    scores = {
        "technical": score_technical(latest, df),
        "momentum": score_momentum(latest),
        "volume": score_volume(latest),
        "trend": score_trend(latest, df),
        "volatility": score_volatility(latest),
        "relative_strength": score_relative_strength(strength_data),
        "pattern": score_pattern(patterns),
        "news": score_news(news_data),
        "risk": score_risk(risk_data),
        "market": score_market(market_context),
        "base_quality": score_base_quality(base_info),
    }

    # Liquidity score
    if liquidity_data:
        liq_s = 50.0
        if liquidity_data.get("untested_bullish_obs", 0) > 0:
            liq_s += 15
        if liquidity_data.get("zone") == "discount":
            liq_s += 10
        if liquidity_data.get("sweep_count", 0) > 0:
            liq_s += 10
        if liquidity_data.get("relevant_fvgs", 0) > 0:
            liq_s += 10
        if liquidity_data.get("absorption", {}).get("absorption_detected"):
            liq_s += 15
        scores["liquidity"] = _clamp(liq_s)
    else:
        scores["liquidity"] = 50.0

    # Breakout probability (meta-score)
    scores["breakout_probability"] = score_breakout_probability(scores)

    # Weighted composite
    composite = 0.0
    total_weight = 0.0
    for dim, weight in weights.items():
        if dim in scores:
            composite += scores[dim] * weight
            total_weight += weight

    if total_weight > 0:
        composite = composite / total_weight
    else:
        composite = sum(scores.values()) / max(len(scores), 1)

    scores["composite"] = round(_clamp(composite), 1)

    # Confidence: how much agreement between dimensions
    score_vals = [v for k, v in scores.items() if k != "composite"]
    if score_vals:
        std = float(np.std(score_vals))
        # Low std = high agreement = high confidence
        scores["confidence"] = round(_clamp(100 - std, 0, 100), 1)
    else:
        scores["confidence"] = 50.0

    return scores
