"""
News Fetcher — Multi-source RSS aggregation.

Sources:
  - Google News RSS (Indian market focus)
  - Deduplication by headline similarity
  - Rate limiting
"""

import hashlib
import sys
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import requests

from config.settings import Settings

# Attempt to import feedparser; if unavailable, news fetching degrades gracefully
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("[NEWS] feedparser not installed — news fetching disabled.",
          file=sys.stderr)


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_rps: float = 2.0):
        self._min_interval = 1.0 / max_rps
        self._last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()


_limiter = RateLimiter(max_rps=2.0)


# ============================================================
# HEADLINE CACHE
# ============================================================

_headline_cache: Dict[str, dict] = {}
_cache_timestamp: Optional[datetime] = None


def _cache_valid() -> bool:
    """Check if the headline cache is still valid."""
    if _cache_timestamp is None:
        return False
    cfg = Settings.get()
    ttl = timedelta(minutes=cfg.news_cache_ttl_minutes)
    return datetime.now() - _cache_timestamp < ttl


# ============================================================
# FETCHERS
# ============================================================

def fetch_google_news(ticker: str, max_results: int = 10
                      ) -> List[dict]:
    """
    Fetch news headlines from Google News RSS for an NSE ticker.
    """
    if not HAS_FEEDPARSER:
        return []

    clean_ticker = ticker.replace(".NS", "")
    url = (
        f"https://news.google.com/rss/search?"
        f"q={clean_ticker}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
    )

    _limiter.wait()

    try:
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:max_results]:
            headlines.append({
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "ticker": clean_ticker,
            })
        return headlines
    except Exception as e:
        print(f"[NEWS] Google News error for {clean_ticker}: {e}",
              file=sys.stderr)
        return []


def fetch_headlines_batch(tickers: List[str],
                          max_per_ticker: int = 5
                          ) -> Dict[str, List[dict]]:
    """
    Fetch news for multiple tickers, with caching.
    Returns dict of {ticker: [headlines]}.
    """
    global _headline_cache, _cache_timestamp

    if _cache_valid():
        # Return cached results for already-fetched tickers
        result = {}
        uncached = []
        for t in tickers:
            clean = t.replace(".NS", "")
            if clean in _headline_cache:
                result[clean] = _headline_cache[clean]
            else:
                uncached.append(t)
        if not uncached:
            return result
        tickers = uncached
    else:
        _headline_cache.clear()
        result = {}

    for ticker in tickers:
        clean = ticker.replace(".NS", "")
        try:
            headlines = fetch_google_news(ticker, max_per_ticker)
            _headline_cache[clean] = headlines
            result[clean] = headlines
        except Exception as e:
            print(f"[NEWS] Error fetching {clean}: {e}", file=sys.stderr)
            result[clean] = []

    _cache_timestamp = datetime.now()
    return result


def deduplicate_headlines(headlines: List[dict]) -> List[dict]:
    """
    Deduplicate headlines by title similarity.
    Uses simple hash-based dedup.
    """
    seen = set()
    unique = []
    for h in headlines:
        title = h.get("title", "").strip().lower()
        # Use first 50 chars as fingerprint
        fp = hashlib.md5(title[:50].encode()).hexdigest()
        if fp not in seen:
            seen.add(fp)
            unique.append(h)
    return unique
