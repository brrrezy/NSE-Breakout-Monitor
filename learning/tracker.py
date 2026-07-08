"""
Performance Tracker — SQLite-based alert tracking and outcome analysis.

Tracks:
  - Every alert issued (ticker, date, entry, stop, target, scores)
  - Outcomes after N trading days (win, loss, false breakout, missed)
  - Rolling performance metrics (precision, recall, win rate, Sharpe)
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings


class PerformanceTracker:
    """
    Track scan predictions and their outcomes in SQLite.
    """

    def __init__(self):
        cfg = Settings.get()
        self._db_path = cfg.learning_db_path
        self._check_days = cfg.learning_outcome_check_days
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    alert_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    target_1 REAL,
                    target_2 REAL,
                    target_3 REAL,
                    composite_score REAL,
                    confidence_score REAL,
                    breakout_prob REAL,
                    scores_json TEXT,
                    outcome TEXT DEFAULT 'pending',
                    exit_price REAL,
                    exit_date TEXT,
                    return_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_alerts INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    false_breakouts INTEGER,
                    win_rate REAL,
                    avg_return REAL,
                    sharpe REAL,
                    precision_val REAL,
                    recall_val REAL
                )
            """)
            self._conn.commit()
        except Exception as e:
            print(f"[TRACKER] DB init error: {e}", file=sys.stderr)
            self._conn = None

    def record_alert(self, candidate: Dict[str, Any]):
        """Record a new alert to the database."""
        if self._conn is None:
            return

        scores = candidate.get("scores", {})
        risk = candidate.get("risk", {})

        try:
            import json
            self._conn.execute("""
                INSERT INTO alerts (
                    ticker, alert_date, status, entry_price,
                    stop_loss, target_1, target_2, target_3,
                    composite_score, confidence_score, breakout_prob,
                    scores_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate.get("symbol", ""),
                datetime.now().strftime("%Y-%m-%d"),
                candidate.get("status", "WATCHLIST"),
                candidate.get("price", 0),
                risk.get("stop", 0),
                risk.get("target_1", 0),
                risk.get("target_2", 0),
                risk.get("target_3", 0),
                scores.get("composite", 0),
                scores.get("confidence", 0),
                scores.get("breakout_probability", 0),
                json.dumps(scores),
            ))
            self._conn.commit()
        except Exception as e:
            print(f"[TRACKER] Record error: {e}", file=sys.stderr)

    def check_outcomes(self, price_lookup) -> int:
        """
        Check outcomes for pending alerts older than N days.

        price_lookup: callable(ticker) -> current_price or None
        Returns count of resolved alerts.
        """
        if self._conn is None:
            return 0

        cutoff = (datetime.now() - timedelta(
            days=self._check_days)).strftime("%Y-%m-%d")

        try:
            cursor = self._conn.execute("""
                SELECT id, ticker, entry_price, stop_loss, target_1,
                       alert_date, status
                FROM alerts
                WHERE outcome = 'pending'
                  AND alert_date <= ?
            """, (cutoff,))

            resolved = 0
            for row in cursor.fetchall():
                alert_id, ticker, entry, stop, target_1, _, status = row
                current = price_lookup(ticker)
                if current is None:
                    continue

                if status == "ACTIONABLE":
                    if current >= target_1:
                        outcome = "win"
                    elif current <= stop:
                        outcome = "loss"
                    elif current < entry * 0.97:
                        outcome = "false_breakout"
                    else:
                        outcome = "pending"  # Still in play
                else:
                    # WATCHLIST: did it break out?
                    if current > entry * 1.05:
                        outcome = "win"
                    else:
                        outcome = "missed"

                if outcome != "pending":
                    ret_pct = (current - entry) / entry if entry > 0 else 0
                    self._conn.execute("""
                        UPDATE alerts
                        SET outcome = ?, exit_price = ?,
                            exit_date = ?, return_pct = ?
                        WHERE id = ?
                    """, (outcome, current,
                          datetime.now().strftime("%Y-%m-%d"),
                          round(ret_pct, 4), alert_id))
                    resolved += 1

            self._conn.commit()
            return resolved

        except Exception as e:
            print(f"[TRACKER] Outcome check error: {e}", file=sys.stderr)
            return 0

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get rolling performance metrics."""
        if self._conn is None:
            return {}

        cutoff = (datetime.now() - timedelta(
            days=days)).strftime("%Y-%m-%d")

        try:
            cursor = self._conn.execute("""
                SELECT outcome, return_pct, composite_score
                FROM alerts
                WHERE outcome != 'pending'
                  AND alert_date >= ?
            """, (cutoff,))

            rows = cursor.fetchall()
            if not rows:
                return {
                    "total": 0, "win_rate": 0,
                    "avg_return": 0, "period_days": days,
                }

            total = len(rows)
            wins = sum(1 for r in rows if r[0] == "win")
            losses = sum(1 for r in rows if r[0] == "loss")
            false_bos = sum(1 for r in rows if r[0] == "false_breakout")
            returns = [r[1] for r in rows if r[1] is not None]

            avg_ret = sum(returns) / len(returns) if returns else 0
            win_rate = wins / total if total > 0 else 0

            # Precision = wins / (wins + false_breakouts)
            actionable_total = wins + losses + false_bos
            precision = wins / actionable_total if actionable_total > 0 else 0

            # Simple Sharpe proxy (annualized)
            import numpy as np
            if len(returns) >= 5:
                ret_arr = np.array(returns)
                sharpe = (ret_arr.mean() / ret_arr.std() *
                          np.sqrt(252)) if ret_arr.std() > 0 else 0
            else:
                sharpe = 0

            return {
                "total": total,
                "wins": wins,
                "losses": losses,
                "false_breakouts": false_bos,
                "win_rate": round(win_rate, 3),
                "avg_return": round(avg_ret, 4),
                "precision": round(precision, 3),
                "sharpe": round(float(sharpe), 2),
                "period_days": days,
            }

        except Exception as e:
            print(f"[TRACKER] Metrics error: {e}", file=sys.stderr)
            return {}

    def get_performance_report(self) -> str:
        """Generate a text performance report for Telegram."""
        m30 = self.get_performance_metrics(30)
        m7 = self.get_performance_metrics(7)

        if not m30.get("total"):
            return "📊 *Performance*: No tracked results yet."

        return (
            f"📊 *Performance Report*\n"
            f"{'─' * 24}\n"
            f"*Last 7 days:*\n"
            f"  Alerts: {m7.get('total', 0)} | "
            f"Win: {m7.get('wins', 0)} | "
            f"Loss: {m7.get('losses', 0)}\n"
            f"  Win Rate: {m7.get('win_rate', 0)*100:.0f}% | "
            f"Avg: {m7.get('avg_return', 0)*100:+.1f}%\n\n"
            f"*Last 30 days:*\n"
            f"  Alerts: {m30.get('total', 0)} | "
            f"Win: {m30.get('wins', 0)} | "
            f"Loss: {m30.get('losses', 0)} | "
            f"False BO: {m30.get('false_breakouts', 0)}\n"
            f"  Win Rate: {m30.get('win_rate', 0)*100:.0f}% | "
            f"Precision: {m30.get('precision', 0)*100:.0f}%\n"
            f"  Avg Return: {m30.get('avg_return', 0)*100:+.1f}% | "
            f"Sharpe: {m30.get('sharpe', 0):.2f}\n"
            f"{'─' * 24}"
        )

    def close(self):
        if self._conn:
            self._conn.close()
