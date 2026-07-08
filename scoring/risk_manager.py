"""
Risk Management & Rejection Framework.

Applies hard filters to reject setups that don't meet
institutional quality standards.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from config.settings import Settings


# ============================================================
# REJECTION RULES
# ============================================================

def apply_risk_filters(candidate: Dict[str, Any],
                       df: pd.DataFrame,
                       market_context: Dict[str, Any]
                       ) -> Tuple[bool, List[str]]:
    """
    Apply all risk rejection filters to a candidate.

    Returns:
        (passes, rejection_reasons)
        passes = True if the candidate passes all filters.
        rejection_reasons = list of reasons if rejected.
    """
    cfg = Settings.get()
    reasons: List[str] = []
    latest = df.iloc[-1]

    # 1. Minimum liquidity
    turnover = latest.get("turnover_ma50", 0)
    if pd.isna(turnover) or turnover < cfg.min_turnover:
        reasons.append(
            f"Low liquidity: ₹{turnover/1e7:.1f}Cr avg turnover "
            f"(min: ₹{cfg.min_turnover_cr}Cr)")

    # 2. Weak trend (ADX < 15 and no EMA alignment)
    adx = latest.get("adx", 0)
    ribbon = latest.get("ribbon_score", 0)
    if (not pd.isna(adx) and adx < 15 and
            not pd.isna(ribbon) and ribbon < 0.5):
        reasons.append(f"Weak trend: ADX={adx:.0f}, Ribbon={ribbon:.2f}")

    # 3. Bearish market + high beta
    regime = market_context.get("regime", "unknown")
    beta = candidate.get("strength", {}).get("beta", 1.0)
    if regime == "bear" and beta > 1.2:
        reasons.append(
            f"Bear market regime with high beta ({beta:.2f})")

    # 4. Negative news catalyst
    news_score = candidate.get("news", {}).get("news_score", 0)
    if news_score < -50:
        reasons.append(f"Negative news catalyst (score: {news_score})")

    # 5. Poor volume on "breakout" day
    if candidate.get("status") == "ACTIONABLE":
        rvol = latest.get("rvol", 1.0)
        if not pd.isna(rvol) and rvol < 0.5:
            reasons.append(f"Poor volume on breakout (RVol: {rvol:.1f}x)")

    # 6. Late-stage breakout (already extended)
    price = float(latest["Close"])
    pivot = candidate.get("pivot", price)
    if pivot > 0:
        extension = (price - pivot) / pivot
        if extension > 0.15:
            reasons.append(
                f"Late-stage: price {extension*100:.0f}% above pivot")

    # 7. High risk (stop too far from entry)
    risk_pct = candidate.get("risk", {}).get("risk_pct", 0)
    if risk_pct > cfg.risk_max_stop_pct:
        reasons.append(
            f"High risk: stop {risk_pct*100:.1f}% from entry "
            f"(max: {cfg.risk_max_stop_pct*100}%)")

    # 8. Unfavorable R:R
    rr = candidate.get("risk", {}).get("rr", 0)
    if rr > 0 and rr < cfg.risk_rr_min:
        reasons.append(
            f"Low R:R ratio: {rr}:1 (min: {cfg.risk_rr_min}:1)")

    # 9. RSI overbought on breakout day
    rsi = latest.get("rsi", 50)
    if candidate.get("status") == "ACTIONABLE":
        if not pd.isna(rsi) and rsi > 85:
            reasons.append(f"RSI overbought: {rsi:.0f}")

    # 10. VIX extreme
    if not market_context.get("favorable_for_longs", True):
        vix = market_context.get("vix_value", 0)
        reasons.append(f"India VIX extreme: {vix:.1f}")

    passes = len(reasons) == 0
    return passes, reasons


def generate_failure_conditions(candidate: Dict[str, Any]) -> List[str]:
    """
    Generate potential failure conditions for a setup.
    These are warnings, not rejection criteria.
    """
    conditions = []
    scores = candidate.get("scores", {})

    if scores.get("volume", 50) < 40:
        conditions.append("Volume not confirming the move")

    if scores.get("momentum", 50) < 40:
        conditions.append("Weak momentum behind the setup")

    if scores.get("market", 50) < 40:
        conditions.append("Unfavorable market environment")

    if candidate.get("base_info", {}).get("base_vol_ratio", 1.0) > 0.9:
        conditions.append("Volume did not dry up sufficiently in base")

    if scores.get("relative_strength", 50) < 40:
        conditions.append("Stock underperforming the index")

    if not candidate.get("base_info", {}).get("has_vcp"):
        conditions.append("No volatility contraction pattern detected")

    # Liquidity trap risk
    liquidity = candidate.get("liquidity", {})
    if liquidity.get("equal_highs", 0) >= 3:
        conditions.append(
            "Multiple equal highs above — potential liquidity trap")

    return conditions


def suggest_monitoring_frequency(candidate: Dict[str, Any]) -> str:
    """
    Suggest how frequently to monitor this stock based on its
    setup maturity and scores.
    """
    status = candidate.get("status", "WATCHLIST")
    scores = candidate.get("scores", {})
    composite = scores.get("composite", 50)

    if status == "ACTIONABLE":
        return "15 min (active trade management)"

    if composite >= 80:
        return "Hourly (high probability setup, breakout imminent)"
    elif composite >= 65:
        return "Every 2 hours (strong setup, watch for catalyst)"
    elif composite >= 50:
        return "End of day (developing setup)"
    else:
        return "Daily (early stage, patience required)"
