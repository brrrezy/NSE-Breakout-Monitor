"""
Adaptive Weight Optimizer.

Analyzes historical outcomes to identify which scoring dimensions
correlate most with winning trades, then updates weights accordingly.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from config.settings import Settings
from learning.tracker import PerformanceTracker


class WeightOptimizer:
    """
    Analyzes past alerts and optimizes scoring weights based on
    which dimensions correlated most with successful outcomes.
    """

    def __init__(self, tracker: PerformanceTracker):
        self._tracker = tracker
        cfg = Settings.get()
        self._min_samples = cfg.learning_min_samples_optimize
        self._max_change = cfg.learning_max_weight_change
        self._history_file = cfg.cache_dir / "weight_history.json"

    def should_optimize(self) -> bool:
        """Check if we have enough data to optimize."""
        metrics = self._tracker.get_performance_metrics(days=90)
        return metrics.get("total", 0) >= self._min_samples

    def optimize(self) -> Optional[Dict[str, float]]:
        """
        Analyze historical alerts and compute optimal weights.

        Uses correlation analysis: for each scoring dimension,
        compute the correlation between that dimension's score
        and the outcome (win=1, loss=-1, false_bo=-0.5).

        Returns new weights dict, or None if not enough data.
        """
        if not self.should_optimize():
            return None

        if self._tracker._conn is None:
            return None

        try:
            cursor = self._tracker._conn.execute("""
                SELECT scores_json, outcome, return_pct
                FROM alerts
                WHERE outcome != 'pending'
                  AND scores_json IS NOT NULL
                  AND scores_json != ''
            """)

            rows = cursor.fetchall()
            if len(rows) < self._min_samples:
                return None

            # Parse scores and outcomes
            dimensions = list(Settings.get().scoring_weights.keys())
            score_matrix = []
            outcomes = []

            for scores_json, outcome, ret_pct in rows:
                try:
                    scores = json.loads(scores_json)
                except Exception:
                    continue

                row_scores = [scores.get(dim, 50) for dim in dimensions]
                score_matrix.append(row_scores)

                # Outcome encoding
                if outcome == "win":
                    outcomes.append(1.0)
                elif outcome == "loss":
                    outcomes.append(-1.0)
                elif outcome == "false_breakout":
                    outcomes.append(-0.5)
                else:
                    outcomes.append(0.0)

            if len(outcomes) < self._min_samples:
                return None

            X = np.array(score_matrix)
            y = np.array(outcomes)

            # Compute correlation of each dimension with outcomes
            correlations = {}
            for i, dim in enumerate(dimensions):
                col = X[:, i]
                if col.std() > 0:
                    corr = np.corrcoef(col, y)[0, 1]
                    correlations[dim] = float(corr) if not np.isnan(corr) \
                        else 0.0
                else:
                    correlations[dim] = 0.0

            # Convert correlations to weights
            # Positive correlation → higher weight
            # Negative correlation → lower weight (but keep positive)
            current_weights = dict(Settings.get().scoring_weights)
            new_weights = {}

            for dim in dimensions:
                corr = correlations.get(dim, 0)
                current = current_weights.get(dim, 0.05)

                # Adjustment: proportional to correlation
                adjustment = corr * 0.05  # Max 5% absolute change per cycle
                new_val = current + adjustment

                # Clamp change to max_change
                change = new_val - current
                if abs(change) > self._max_change * current:
                    change = self._max_change * current * np.sign(change)
                    new_val = current + change

                # Floor at 0.01
                new_weights[dim] = max(0.01, new_val)

            # Normalize to sum to 1.0
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: round(v / total, 4)
                               for k, v in new_weights.items()}

            # Save history
            self._save_history(current_weights, new_weights, correlations)

            print(f"[OPTIMIZER] Updated weights based on "
                  f"{len(outcomes)} samples")
            return new_weights

        except Exception as e:
            print(f"[OPTIMIZER] Error: {e}", file=sys.stderr)
            return None

    def _save_history(self, old_weights: dict, new_weights: dict,
                      correlations: dict):
        """Save weight change history for auditability."""
        from datetime import datetime
        history = []

        if self._history_file.exists():
            try:
                history = json.loads(self._history_file.read_text())
            except Exception:
                history = []

        history.append({
            "date": datetime.now().isoformat(),
            "old_weights": old_weights,
            "new_weights": new_weights,
            "correlations": correlations,
        })

        # Keep last 50 entries
        history = history[-50:]
        self._history_file.write_text(json.dumps(history, indent=2))
