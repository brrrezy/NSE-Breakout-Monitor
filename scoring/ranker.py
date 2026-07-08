"""
Ranking & Filtering Engine.

Sorts candidates by composite score and applies configurable
ranking criteria. Outputs the top-N shortlist.
"""

from typing import Any, Dict, List

from config.settings import Settings


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank candidates by multiple criteria.

    Primary: Composite Score (descending)
    Tiebreakers:
      1. Breakout Probability
      2. Risk Score (higher = lower risk = better)
      3. Base Quality
      4. Relative Strength
    """
    if not candidates:
        return []

    def _sort_key(c):
        scores = c.get("scores", {})
        return (
            scores.get("composite", 0),
            scores.get("breakout_probability", 0),
            scores.get("risk", 0),
            scores.get("base_quality", 0),
            scores.get("relative_strength", 0),
        )

    ranked = sorted(candidates, key=_sort_key, reverse=True)

    # Assign rank
    for i, c in enumerate(ranked):
        c["rank"] = i + 1

    # Apply top-N filter
    top_n = Settings.get().scoring_top_n
    return ranked[:top_n]


def filter_by_threshold(candidates: List[Dict[str, Any]],
                        threshold: float = None
                        ) -> List[Dict[str, Any]]:
    """
    Filter candidates by minimum composite score.
    """
    if threshold is None:
        threshold = Settings.get().scoring_shortlist_threshold

    return [c for c in candidates
            if c.get("scores", {}).get("composite", 0) >= threshold]


def separate_actionable_watchlist(
    candidates: List[Dict[str, Any]]
) -> tuple:
    """
    Separate candidates into ACTIONABLE (breakout today) and
    WATCHLIST (base forming, watch for breakout).
    """
    actionable = [c for c in candidates if c.get("status") == "ACTIONABLE"]
    watchlist = [c for c in candidates if c.get("status") == "WATCHLIST"]
    return actionable, watchlist
