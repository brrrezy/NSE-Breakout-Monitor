web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: python3 swing_screener.py --live --universe-limit 500 --interval-min 15
