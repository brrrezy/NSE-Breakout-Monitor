"""
Groq-Powered Sentiment Analysis.

Analyzes fetched headlines to produce:
  - News Score (-100 to +100)
  - Sentiment Score (-1.0 to +1.0)
  - Catalyst Strength (0-10)
  - Confidence (0-1.0)
  - Time Decay Factor
  - Summary text
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from config.settings import Settings
from news.fetcher import fetch_headlines_batch, deduplicate_headlines


def analyze_sentiment_batch(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch headlines and analyze sentiment for multiple tickers.
    Returns dict of {ticker: sentiment_data}.
    """
    cfg = Settings.get()

    if not cfg.enable_news_engine or not cfg.groq_api_key:
        return {t.replace(".NS", ""): _neutral_sentiment()
                for t in tickers}

    # Fetch headlines
    all_headlines = fetch_headlines_batch(tickers)

    results = {}
    for ticker, headlines in all_headlines.items():
        headlines = deduplicate_headlines(headlines)
        if not headlines:
            results[ticker] = _neutral_sentiment()
            continue

        try:
            analysis = _analyze_with_groq(ticker, headlines)
            results[ticker] = analysis
        except Exception as e:
            print(f"[SENTIMENT] Error for {ticker}: {e}", file=sys.stderr)
            results[ticker] = _neutral_sentiment()

    return results


def _analyze_with_groq(ticker: str,
                       headlines: List[dict]) -> Dict[str, Any]:
    """
    Use Groq API to analyze headlines and extract sentiment.
    """
    cfg = Settings.get()

    # Format headlines for the prompt
    headline_text = "\n".join(
        f"- [{h.get('source', 'Unknown')}] {h.get('title', '')}"
        for h in headlines[:8]
    )

    prompt = f"""Analyze these recent news headlines for the Indian stock {ticker} (NSE).

Headlines:
{headline_text}

Return a JSON object with EXACTLY these fields:
{{
  "sentiment_score": <float between -1.0 and 1.0, where -1=very bearish, 0=neutral, 1=very bullish>,
  "news_score": <integer between -100 and 100>,
  "catalyst_strength": <integer 0-10, where 0=no catalyst, 10=major catalyst like M&A>,
  "catalyst_type": <string: one of "earnings", "contract", "upgrade", "downgrade", "insider", "m&a", "sector", "macro", "policy", "none">,
  "confidence": <float 0-1, based on headline quality and source reliability>,
  "summary": <string: 1-2 sentence summary of the news impact>
}}

Rules:
- Only analyze what the headlines say. Do not infer or fabricate information.
- If headlines are generic/irrelevant, return neutral scores.
- Be conservative with scores. Only give extreme values for clearly significant news.
- Return ONLY the JSON object, no other text."""

    try:
        headers = {
            "Authorization": f"Bearer {cfg.groq_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": cfg.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.groq_max_tokens,
            "temperature": cfg.groq_temperature,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=data, timeout=15)
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        # Validate and clamp values
        result["sentiment_score"] = max(-1.0, min(1.0,
            float(result.get("sentiment_score", 0))))
        result["news_score"] = max(-100, min(100,
            int(result.get("news_score", 0))))
        result["catalyst_strength"] = max(0, min(10,
            int(result.get("catalyst_strength", 0))))
        result["confidence"] = max(0.0, min(1.0,
            float(result.get("confidence", 0.5))))

        # Apply time decay
        result["time_decay"] = _compute_time_decay(headlines)

        # Adjust scores by time decay
        decay = result["time_decay"]
        result["news_score"] = int(result["news_score"] * decay)
        result["sentiment_score"] = round(
            result["sentiment_score"] * decay, 3)

        return result

    except json.JSONDecodeError:
        print(f"[SENTIMENT] Failed to parse Groq response for {ticker}",
              file=sys.stderr)
        return _neutral_sentiment()
    except Exception as e:
        print(f"[SENTIMENT] Groq API error for {ticker}: {e}",
              file=sys.stderr)
        return _neutral_sentiment()


def _compute_time_decay(headlines: List[dict]) -> float:
    """
    Compute time decay factor based on headline freshness.
    Half-life = 3 days.
    """
    cfg = Settings.get()
    half_life = cfg.news_time_decay_halflife_days
    now = datetime.now()

    # Try to parse published dates
    ages_days = []
    for h in headlines:
        pub = h.get("published", "")
        if pub:
            try:
                # feedparser dates are typically in various formats
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub)
                age = (now - dt.replace(tzinfo=None)).total_seconds() / 86400
                ages_days.append(max(0, age))
            except Exception:
                pass

    if not ages_days:
        return 0.7  # Default moderate decay for undated news

    avg_age = sum(ages_days) / len(ages_days)

    import math
    # Exponential decay: 2^(-age/half_life)
    decay = math.pow(2, -avg_age / half_life)
    return round(max(0.1, min(1.0, decay)), 3)


def _neutral_sentiment() -> Dict[str, Any]:
    """Return neutral sentiment (no news or analysis failed)."""
    return {
        "sentiment_score": 0.0,
        "news_score": 0,
        "catalyst_strength": 0,
        "catalyst_type": "none",
        "confidence": 0.0,
        "summary": "No recent news",
        "time_decay": 1.0,
    }


def get_groq_verdict(candidates: List[Dict],
                     is_eod: bool = False) -> str:
    """
    Get Groq AI's overall verdict on the watchlist.
    Enhanced from the original scanner.
    """
    cfg = Settings.get()

    if not cfg.groq_api_key or not candidates:
        return ""

    # Build summary text for each candidate
    wl_text = ""
    for c in candidates[:10]:
        sym = c.get("symbol", "").replace(".NS", "")
        scores = c.get("scores", {})
        base_info = c.get("base_info", {})
        strength = c.get("strength", {})
        price = c.get("price", 0)
        pivot = c.get("pivot", 0)
        dist = ((pivot - price) / price * 100) if price > 0 else 0

        wl_text += (
            f"[{sym}] Price: {price:.0f}, Pivot: {pivot:.0f} "
            f"(Gap: {dist:.1f}%), "
            f"Score: {scores.get('composite', 0):.0f}/100, "
            f"Base: {base_info.get('base_len', 0)}d, "
            f"VCP: {'✅' if base_info.get('has_vcp') else '❌'}, "
            f"Coil: {'✅' if base_info.get('vol_coil') else '❌'}, "
            f"Tight: {'✅' if base_info.get('tightening') else '❌'}, "
            f"RS: {strength.get('rs_composite', 0)*100:+.0f}%, "
            f"Squeeze: {'✅' if base_info.get('squeeze_active') else '❌'}\n"
        )

    time_ctx = ("End of Day (building tomorrow's watchlist)"
                if is_eod else
                "Morning (looking for immediate breakouts)")

    prompt = f"""You are the best stock analyst in the Indian stock market with 50 years of experience.
Your goal is to find A+ grade momentum bursts while strictly avoiding fakeouts and exhausted setups.

Time of day: {time_ctx}

Key signals:
- VCP (Volatility Contraction): supply drying up
- VolCoil: institutions loading before breakout
- Tightening: stock coiling like a spring
- Squeeze: Bollinger inside Keltner — compressed volatility
- RS vs Nifty: positive = market leader
- Score: composite institutional quality (0-100)

Watchlist:
{wl_text}

Provide a short, punchy verdict (max 4-5 sentences):
1. Prioritize stocks with highest Score AND VCP+Coil combo
2. Dismiss weak setups clearly
3. If VCP + Coil + Tight + Squeeze all present, flag as TOP PRIORITY
4. If none are A+ setups, say so clearly
Plain text with emojis. No markdown headers."""

    try:
        headers = {
            "Authorization": f"Bearer {cfg.groq_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": cfg.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.groq_verdict_max_tokens,
            "temperature": 0.3,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip()
        return f"\n\n🤖 *Groq Verdict:*\n{verdict}"
    except Exception as e:
        print(f"[GROQ] Error: {e}", file=sys.stderr)
        return ""
