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

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        result["sentiment_score"] = max(-1.0, min(1.0,
            float(result.get("sentiment_score", 0))))
        result["news_score"] = max(-100, min(100,
            int(result.get("news_score", 0))))
        result["catalyst_strength"] = max(0, min(10,
            int(result.get("catalyst_strength", 0))))
        result["confidence"] = max(0.0, min(1.0,
            float(result.get("confidence", 0.5))))

        result["time_decay"] = _compute_time_decay(headlines)
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
    cfg = Settings.get()
    half_life = cfg.news_time_decay_halflife_days
    now = datetime.now()

    ages_days = []
    for h in headlines:
        pub = h.get("published", "")
        if pub:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub)
                age = (now - dt.replace(tzinfo=None)).total_seconds() / 86400
                ages_days.append(max(0, age))
            except Exception:
                pass

    if not ages_days:
        return 0.7

    avg_age = sum(ages_days) / len(ages_days)
    import math
    decay = math.pow(2, -avg_age / half_life)
    return round(max(0.1, min(1.0, decay)), 3)


def _neutral_sentiment() -> Dict[str, Any]:
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
    Get Groq AI's verdict focused solely on News Catalysts.
    """
    cfg = Settings.get()

    if not cfg.groq_api_key or not candidates:
        return ""

    # Build summary text of news for the top 5 candidates
    news_text = ""
    for c in candidates[:5]:
        sym = c.get("symbol", "").replace(".NS", "")
        news = c.get("news", {})
        summary = news.get("summary", "No recent news.")
        cat_type = news.get("catalyst_type", "None")
        cat_str = news.get("catalyst_strength", 0)
        news_text += f"[{sym}] Catalyst: {cat_type} (Strength: {cat_str}/10). Summary: {summary}\n"

    if not news_text:
        return ""

    prompt = f"""You are a fundamental market analyst. I have a watchlist of top technical setups.
Your job is to read their recent news summaries and provide a single paragraph verdict on their fundamental catalysts.

Watchlist News Data:
{news_text}

Task:
Write a single, highly professional 3-4 sentence briefing.
- Tell me which stock has the best fundamental tailwind to support a breakout.
- Tell me which stock to avoid due to bearish, conflicting, or non-existent news.
- DO NOT use emojis.
- DO NOT use generic phrases like "The market is dynamic."
- Be clinical, direct, and analytical.
"""

    try:
        headers = {
            "Authorization": f"Bearer {cfg.groq_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": cfg.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
            "temperature": 0.2,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Wrap in clean monospaced block
        return (
            f"\n```text\n"
            f"[ CATALYST VERDICT ]\n"
            f"=========================================\n"
            f"{verdict}\n"
            f"=========================================\n"
            f"```"
        )
    except Exception as e:
        print(f"[GROQ] Error: {e}", file=sys.stderr)
        return ""
