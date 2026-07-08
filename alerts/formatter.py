"""
Alert Formatter — Structured output for every shortlisted stock.

Produces two output formats:
  1. Telegram: condensed, emoji-rich, mobile-friendly
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

    # Build reasons for/against
    reasons_for = _build_reasons_for(candidate)
    reasons_against = _build_reasons_against(candidate)
    failure_conditions = generate_failure_conditions(candidate)
    monitoring = suggest_monitoring_frequency(candidate)

    return {
        "Ticker": candidate.get("symbol", ""),
        "Company Name": _short(candidate.get("symbol", "")),
        "Status": candidate.get("status", "WATCHLIST"),
        "Rank": candidate.get("rank", 0),

        # ── Structure ──
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

        # ── Liquidity ──
        "Liquidity Analysis": (
            f"Equal highs: {liquidity.get('equal_highs', 0)}, "
            f"Equal lows: {liquidity.get('equal_lows', 0)}, "
            f"Bullish OBs: {liquidity.get('bullish_order_blocks', 0)}, "
            f"FVGs: {liquidity.get('unfilled_fvg_count', 0)}, "
            f"Zone: {liquidity.get('zone', 'unknown')}"
        ),

        # ── Pattern ──
        "Pattern": pattern_name,
        "Pattern Detail": pattern_desc,

        # ── Volume ──
        "Volume Analysis": (
            f"RVol: {candidate.get('rvol', 0):.1f}x, "
            f"Net Accum: {base_info.get('net_accum', 0)}, "
            f"Base Vol Ratio: {base_info.get('base_vol_ratio', 0):.2f}"
        ),

        # ── Momentum ──
        "Momentum": (
            f"RSI: {candidate.get('rsi', 0):.0f}, "
            f"MACD Hist: {'Positive' if candidate.get('macd_hist', 0) > 0 else 'Negative'}, "
            f"ADX: {candidate.get('adx', 0):.0f}"
        ),

        # ── Relative Strength ──
        "Relative Strength": (
            f"RS vs Nifty: {strength.get('rs_composite', 0)*100:+.1f}%, "
            f"Beta: {strength.get('beta', 1.0):.2f}, "
            f"Sector: {strength.get('sector', 'N/A')}"
        ),

        # ── News ──
        "News Summary": news.get("summary", "No recent news"),
        "Catalyst": news.get("catalyst_type", "None"),

        # ── Risk / Reward ──
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

        # ── Scores ──
        "Institutional Score": scores.get("composite", 0),
        "Confidence Score": scores.get("confidence", 0),
        "Breakout Probability": scores.get("breakout_probability", 0),

        # ── Reasoning ──
        "Reasons for Selection": reasons_for,
        "Reasons Against Selection": reasons_against,
        "Potential Failure Conditions": failure_conditions,
        "Suggested Monitoring Frequency": monitoring,

        # ── Raw scores (for analysis) ──
        "scores": scores,
    }


# ============================================================
# TELEGRAM FORMATS
# ============================================================

def format_breakout_telegram(candidate: Dict[str, Any]) -> str:
    """Format a breakout alert for Telegram."""
    sym = _short(candidate.get("symbol", ""))
    scores = candidate.get("scores", {})
    risk = candidate.get("risk", {})
    base_info = candidate.get("base_info", {})
    strength = candidate.get("strength", {})

    # Edge tags
    edge_tags = []
    if base_info.get("has_vcp"):
        edge_tags.append("VCP")
    if base_info.get("vol_coil"):
        edge_tags.append("VolCoil")
    if base_info.get("tightening"):
        edge_tags.append("Tight")
    if base_info.get("net_accum", 0) >= 3:
        edge_tags.append(f"Acc+{base_info['net_accum']}")
    if strength.get("rs_composite", 0) > 0.05:
        edge_tags.append(f"RS+{strength['rs_composite']*100:.0f}%")
    if base_info.get("squeeze_active"):
        edge_tags.append("Squeeze")

    edge_str = " | ".join(edge_tags) if edge_tags else "—"

    patterns = candidate.get("patterns", [])
    pattern_str = patterns[0].name if patterns else "Base Breakout"

    return (
        f"🚀 *{sym}* — BREAKOUT ALERT\n"
        f"{'=' * 28}\n"
        f"📍 Entry  : ₹{candidate.get('price', 0):.0f}\n"
        f"🛡️ Stop   : ₹{risk.get('stop', 0):.0f}\n"
        f"🎯 T1     : ₹{risk.get('target_1', 0):.0f}\n"
        f"🎯 T2     : ₹{risk.get('target_2', 0):.0f}\n"
        f"🎯 T3     : ₹{risk.get('target_3', 0):.0f}\n"
        f"⚖️ R:R    : {risk.get('rr', 0)}:1\n"
        f"📈 RVol   : {candidate.get('rvol', 0):.1f}x\n"
        f"{'─' * 28}\n"
        f"📊 Score  : {scores.get('composite', 0):.0f}/100\n"
        f"🎲 Prob   : {scores.get('breakout_probability', 0):.0f}%\n"
        f"💪 RS     : {strength.get('rs_composite', 0)*100:+.0f}% vs Nifty\n"
        f"🔲 Pattern: {pattern_str}\n"
        f"🧠 Edge   : {edge_str}\n"
        f"📝 Base   : {base_info.get('base_len', 0)}d, "
        f"{'SFP ' if base_info.get('has_sfp') else ''}"
        f"Doji:{base_info.get('doji_count', 0)}\n"
        f"{'─' * 28}"
    )


def format_watchlist_telegram(candidates: List[Dict[str, Any]],
                              title: str = "WATCHLIST",
                              is_eod: bool = False) -> str:
    """Format the watchlist summary for Telegram."""
    lines = [f"📋 *{title}*", "=" * 24]

    for c in candidates[:10]:
        sym = _short(c.get("symbol", ""))
        scores = c.get("scores", {})
        pivot = c.get("pivot", 0)
        price = c.get("price", 0)
        dist = ((pivot - price) / price * 100) if price > 0 else 0

        edge_tags = []
        base_info = c.get("base_info", {})
        if base_info.get("has_vcp"):
            edge_tags.append("VCP")
        if base_info.get("vol_coil"):
            edge_tags.append("Coil")
        if base_info.get("tightening"):
            edge_tags.append("Tight")
        if base_info.get("squeeze_active"):
            edge_tags.append("Squeeze")

        edge_str = "+".join(edge_tags) if edge_tags else "—"

        lines.append(f"👀 *{sym}* [{scores.get('composite', 0):.0f}]")
        lines.append(f"  ₹{price:.0f} → ₹{pivot:.0f} ({dist:+.1f}%)")
        lines.append(f"  Edge: {edge_str} | "
                     f"Base: {base_info.get('base_len', 0)}d")
        lines.append("─" * 24)

    return "\n".join(lines)


def format_market_context_telegram(context: Dict[str, Any]) -> str:
    """Format market context for Telegram."""
    regime = context.get("regime", "unknown").title()
    vix = context.get("vix_value", 0)
    vix_regime = context.get("vix_regime", "neutral")
    market_score = context.get("market_score", 50)
    nifty = context.get("nifty_close", 0)
    breadth = context.get("breadth", {}).get("breadth_score", 50)

    return (
        f"📊 *Market Context*\n"
        f"{'─' * 24}\n"
        f"🏛️ Regime  : {regime}\n"
        f"📈 Nifty   : {nifty:,.0f}\n"
        f"😰 VIX     : {vix:.1f} ({vix_regime})\n"
        f"📊 Breadth : {breadth:.0f}/100\n"
        f"⭐ Score   : {market_score:.0f}/100\n"
        f"{'─' * 24}"
    )


# ============================================================
# HELPERS
# ============================================================

def _build_reasons_for(candidate: Dict) -> List[str]:
    """Build list of reasons this stock was selected."""
    reasons = []
    scores = candidate.get("scores", {})
    base_info = candidate.get("base_info", {})
    strength = candidate.get("strength", {})

    if scores.get("composite", 0) >= 75:
        reasons.append("High composite institutional score")
    if scores.get("trend", 0) >= 70:
        reasons.append("Strong established uptrend")
    if scores.get("base_quality", 0) >= 70:
        reasons.append("High quality base formation")
    if base_info.get("has_vcp"):
        reasons.append("Volatility contraction pattern detected")
    if base_info.get("vol_coil"):
        reasons.append("Volume coiling — institutional loading")
    if base_info.get("tightening"):
        reasons.append("Close-to-close moves tightening (coiled spring)")
    if base_info.get("net_accum", 0) >= 3:
        reasons.append(f"Net accumulation: {base_info['net_accum']} days")
    if strength.get("rs_composite", 0) > 0.05:
        reasons.append("Strong relative strength vs Nifty 50")
    if scores.get("volatility", 0) >= 60:
        reasons.append("Volatility compressed — ready to expand")
    if base_info.get("squeeze_active"):
        reasons.append("Bollinger/Keltner squeeze active")

    return reasons if reasons else ["Meets all phase criteria"]


def _build_reasons_against(candidate: Dict) -> List[str]:
    """Build list of concerns about this setup."""
    concerns = []
    scores = candidate.get("scores", {})

    if scores.get("volume", 50) < 45:
        concerns.append("Below-average volume support")
    if scores.get("market", 50) < 45:
        concerns.append("Unfavorable market environment")
    if scores.get("news", 50) < 40:
        concerns.append("Negative or absent news catalyst")
    if scores.get("relative_strength", 50) < 40:
        concerns.append("Underperforming the broader market")
    if scores.get("momentum", 50) < 40:
        concerns.append("Weak momentum indicators")

    rejection_reasons = candidate.get("rejection_reasons", [])
    concerns.extend(rejection_reasons)

    return concerns if concerns else ["No significant concerns identified"]


def _risk_level(risk_pct: float) -> str:
    if risk_pct < 0.02:
        return "Low (< 2%)"
    elif risk_pct < 0.03:
        return "Moderate (2-3%)"
    elif risk_pct < 0.05:
        return "Elevated (3-5%)"
    else:
        return "High (> 5%)"


def _reward_level(composite: float) -> str:
    if composite >= 80:
        return "High (A+ setup)"
    elif composite >= 65:
        return "Good (A setup)"
    elif composite >= 50:
        return "Moderate (B setup)"
    else:
        return "Low (C setup)"
