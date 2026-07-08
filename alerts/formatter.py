"""
Alert Formatter — Structured output for every shortlisted stock.

Produces two output formats:
  1. Telegram: clean, monospaced, professional (no emojis)
  2. Detailed: full structured report (dict/JSON)
"""

from typing import Any, Dict, List

from config.settings import Settings
from scoring.risk_manager import generate_failure_conditions, \
    suggest_monitoring_frequency


def _short(sym: str) -> str:
    """Strip .NS suffix for display."""
    return sym.replace(".NS", "")


# ============================================================
# FULL STRUCTURED OUTPUT
# ============================================================

def format_detailed(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the complete structured output per the specification.
    """
    scores = candidate.get("scores", {})
    risk = candidate.get("risk", {})
    base_info = candidate.get("base_info", {})
    strength = candidate.get("strength", {})
    structure = candidate.get("structure", {})
    liquidity = candidate.get("liquidity", {})
    patterns = candidate.get("patterns", [])
    news = candidate.get("news", {})
    market = candidate.get("market_context", {})

    pattern_name = patterns[0].name if patterns else "None detected"
    pattern_desc = patterns[0].description if patterns else ""

    reasons_for = _build_reasons_for(candidate)
    reasons_against = _build_reasons_against(candidate)
    failure_conditions = generate_failure_conditions(candidate)
    monitoring = suggest_monitoring_frequency(candidate)

    return {
        "Ticker": candidate.get("symbol", ""),
        "Company Name": _short(candidate.get("symbol", "")),
        "Status": candidate.get("status", "WATCHLIST"),
        "Rank": candidate.get("rank", 0),
        "Current Trend": structure.get("trend", "unknown"),
        "Market Structure": (
            f"Highs: {structure.get('recent_high_labels', [])}, "
            f"Lows: {structure.get('recent_low_labels', [])}"
        ),
        "Trendline Analysis": (
            f"Resistance lines: {structure.get('resistance_trendlines', 0)}, "
            f"Support lines: {structure.get('support_trendlines', 0)}, "
            f"Breakout pressure: {structure.get('breakout_pressure', 0):.0%}"
        ),
        "Liquidity Analysis": (
            f"Equal highs: {liquidity.get('equal_highs', 0)}, "
            f"Equal lows: {liquidity.get('equal_lows', 0)}, "
            f"Bullish OBs: {liquidity.get('bullish_order_blocks', 0)}, "
            f"FVGs: {liquidity.get('unfilled_fvg_count', 0)}, "
            f"Zone: {liquidity.get('zone', 'unknown')}"
        ),
        "Pattern": pattern_name,
        "Pattern Detail": pattern_desc,
        "Volume Analysis": (
            f"RVol: {candidate.get('rvol', 0):.1f}x, "
            f"Net Accum: {base_info.get('net_accum', 0)}, "
            f"Base Vol Ratio: {base_info.get('base_vol_ratio', 0):.2f}"
        ),
        "Momentum": (
            f"RSI: {candidate.get('rsi', 0):.0f}, "
            f"MACD Hist: {'Positive' if candidate.get('macd_hist', 0) > 0 else 'Negative'}, "
            f"ADX: {candidate.get('adx', 0):.0f}"
        ),
        "Relative Strength": (
            f"RS vs Nifty: {strength.get('rs_composite', 0)*100:+.1f}%, "
            f"Beta: {strength.get('beta', 1.0):.2f}, "
            f"Sector: {strength.get('sector', 'N/A')}"
        ),
        "News Summary": news.get("summary", "No recent news"),
        "Catalyst": news.get("catalyst_type", "None"),
        "Risk Level": _risk_level(risk.get("risk_pct", 0)),
        "Reward Potential": _reward_level(scores.get("composite", 50)),
        "Support": round(candidate.get("support", 0), 2),
        "Resistance": round(candidate.get("pivot", 0), 2),
        "Entry Zone": (
            f"₹{candidate.get('entry_low', 0):.0f} – "
            f"₹{candidate.get('entry_high', 0):.0f}"
        ),
        "Breakout Level": round(candidate.get("pivot", 0), 2),
        "Stop Loss": risk.get("stop", 0),
        "Target 1": risk.get("target_1", 0),
        "Target 2": risk.get("target_2", 0),
        "Target 3": risk.get("target_3", 0),
        "Risk/Reward Ratio": f"{risk.get('rr', 0)}:1",
        "Institutional Score": scores.get("composite", 0),
        "Confidence Score": scores.get("confidence", 0),
        "Breakout Probability": scores.get("breakout_probability", 0),
        "Reasons for Selection": reasons_for,
        "Reasons Against Selection": reasons_against,
        "Potential Failure Conditions": failure_conditions,
        "Suggested Monitoring Frequency": monitoring,
        "scores": scores,
    }


# ============================================================
# TELEGRAM FORMATS (CLEAN & PROFESSIONAL)
# ============================================================

def format_breakout_telegram(candidate: Dict[str, Any]) -> str:
    """Format a breakout alert for Telegram (monospaced, no emojis)."""
    sym = _short(candidate.get("symbol", ""))
    scores = candidate.get("scores", {})
    risk = candidate.get("risk", {})
    base_info = candidate.get("base_info", {})
    strength = candidate.get("strength", {})
    news = candidate.get("news", {})

    edge_tags = []
    if base_info.get("has_vcp"): edge_tags.append("VCP")
    if base_info.get("vol_coil"): edge_tags.append("VolCoil")
    if base_info.get("squeeze_active"): edge_tags.append("Squeeze")
    edge_str = ", ".join(edge_tags) if edge_tags else "None"

    patterns = candidate.get("patterns", [])
    pattern_str = patterns[0].name if patterns else "Base Breakout"
    catalyst = news.get("catalyst_type", "None").title()

    msg = (
        f"```text\n"
        f"[ BREAKOUT ALERT: {sym} ]\n"
        f"===================================\n"
        f"ENTRY     : {candidate.get('price', 0):.2f}\n"
        f"STOP LOSS : {risk.get('stop', 0):.2f}\n"
        f"TARGET 1  : {risk.get('target_1', 0):.2f}\n"
        f"TARGET 2  : {risk.get('target_2', 0):.2f}\n"
        f"R:R RATIO : {risk.get('rr', 0)} : 1\n"
        f"-----------------------------------\n"
        f"SCORE     : {scores.get('composite', 0):.0f} / 100\n"
        f"PROB      : {scores.get('breakout_probability', 0):.0f}%\n"
        f"RVOL      : {candidate.get('rvol', 0):.1f}x\n"
        f"PATTERN   : {pattern_str}\n"
        f"CATALYST  : {catalyst}\n"
        f"EDGE      : {edge_str}\n"
        f"===================================\n"
        f"```"
    )
    return msg


def format_watchlist_telegram(candidates: List[Dict[str, Any]],
                              title: str = "WATCHLIST",
                              is_eod: bool = False) -> str:
    """Format the watchlist summary (top 5 max, monospaced)."""
    if not candidates:
        return ""

    lines = [
        "```text",
        f"[ {title.upper()} SUMMARY ]",
        "=========================================",
        f"{'TICKER':<12} {'PRICE':<9} {'PIVOT':<9} {'SCORE'}",
        "-----------------------------------------"
    ]

    # Limit to top 5 to avoid spam
    for c in candidates[:5]:
        sym = _short(c.get("symbol", ""))
        scores = c.get("scores", {})
        pivot = c.get("pivot", 0)
        price = c.get("price", 0)
        score = scores.get('composite', 0)
        
        lines.append(f"{sym:<12} {price:<9.2f} {pivot:<9.2f} {score:.0f}")

    lines.append("=========================================")
    lines.append("```")
    return "\n".join(lines)


def format_market_context_telegram(context: Dict[str, Any]) -> str:
    """Format market context (clean monospaced)."""
    regime = context.get("regime", "unknown").title()
    vix = context.get("vix_value", 0)
    vix_regime = context.get("vix_regime", "neutral").title()
    market_score = context.get("market_score", 50)
    nifty = context.get("nifty_close", 0)
    breadth = context.get("breadth", {}).get("breadth_score", 50)

    msg = (
        f"```text\n"
        f"[ MARKET CONTEXT ]\n"
        f"===================================\n"
        f"REGIME    : {regime}\n"
        f"NIFTY 50  : {nifty:,.0f}\n"
        f"INDIA VIX : {vix:.1f} ({vix_regime})\n"
        f"BREADTH   : {breadth:.0f} / 100\n"
        f"MKT SCORE : {market_score:.0f} / 100\n"
        f"===================================\n"
        f"```"
    )
    return msg


# ============================================================
# HELPERS
# ============================================================

def _build_reasons_for(candidate: Dict) -> List[str]:
    reasons = []
    scores = candidate.get("scores", {})
    base_info = candidate.get("base_info", {})
    strength = candidate.get("strength", {})

    if scores.get("composite", 0) >= 75:
        reasons.append("High composite institutional score")
    if scores.get("trend", 0) >= 70:
        reasons.append("Strong established uptrend")
    if base_info.get("has_vcp"):
        reasons.append("Volatility contraction pattern detected")
    if base_info.get("vol_coil"):
        reasons.append("Volume coiling — institutional loading")
    if strength.get("rs_composite", 0) > 0.05:
        reasons.append("Strong relative strength vs Nifty 50")
    if base_info.get("squeeze_active"):
        reasons.append("Bollinger/Keltner squeeze active")

    return reasons if reasons else ["Meets all phase criteria"]


def _build_reasons_against(candidate: Dict) -> List[str]:
    concerns = []
    scores = candidate.get("scores", {})

    if scores.get("volume", 50) < 45:
        concerns.append("Below-average volume support")
    if scores.get("market", 50) < 45:
        concerns.append("Unfavorable market environment")
    if scores.get("news", 50) < 40:
        concerns.append("Negative or absent news catalyst")

    rejection_reasons = candidate.get("rejection_reasons", [])
    concerns.extend(rejection_reasons)

    return concerns if concerns else ["No significant concerns identified"]


def _risk_level(risk_pct: float) -> str:
    if risk_pct < 0.02: return "Low (< 2%)"
    elif risk_pct < 0.03: return "Moderate (2-3%)"
    elif risk_pct < 0.05: return "Elevated (3-5%)"
    return "High (> 5%)"


def _reward_level(composite: float) -> str:
    if composite >= 80: return "High (A+ setup)"
    elif composite >= 65: return "Good (A setup)"
    elif composite >= 50: return "Moderate (B setup)"
    return "Low (C setup)"
