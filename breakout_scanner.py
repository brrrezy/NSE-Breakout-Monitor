"""
NSE Breakout Detection Engine
==============================
Elite stock breakout monitoring system for NSE/Indian markets.

This is the entry point. All logic lives in the modular packages:
  - config/    : Settings & configuration
  - core/      : Data provider, universe, engine orchestrator
  - indicators/: Trend, momentum, volume, volatility
  - analysis/  : Market structure, liquidity, patterns, base, breakout
  - scoring/   : Multi-dimensional scorer, ranker, risk manager
  - news/      : RSS fetcher, Groq sentiment analysis
  - learning/  : Performance tracker, adaptive weight optimizer
  - alerts/    : Telegram alerts, structured output formatter
"""

import argparse
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Breakout Detection Engine")
    p.add_argument("--period", default="2y",
                   help="Data period (default: 2y)")
    p.add_argument("--interval", default="1d",
                   help="Data interval (default: 1d)")
    p.add_argument("--test-alert", action="store_true",
                   help="Send a test Telegram message and exit")
    p.add_argument("--perf-report", action="store_true",
                   help="Print performance report and exit")
    p.add_argument("--force", action="store_true",
                   help="Bypass time/weekend checks")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to config
    from config.settings import Settings
    cfg = Settings.get()
    if args.period != "2y":
        cfg._data["data_period"] = args.period
    if args.interval != "1d":
        cfg._data["data_interval"] = args.interval

    # Test alert mode
    if args.test_alert:
        from alerts.telegram import send_telegram
        send_telegram("*Test Alert*: NSE Breakout Engine connected. ✅")
        print("Test alert sent.")
        sys.exit(0)

    # Performance report mode
    if args.perf_report:
        from learning.tracker import PerformanceTracker
        tracker = PerformanceTracker()
        print(tracker.get_performance_report())
        tracker.close()
        sys.exit(0)

    # Run the full scan
    from core.engine import run_scan
    run_scan()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCRITICAL ERROR:")
        traceback.print_exc()
        try:
            from alerts.telegram import send_telegram
            send_telegram(f"*Scanner Crashed*: {str(e)[:200]}")
        except Exception:
            pass
        sys.exit(1)
