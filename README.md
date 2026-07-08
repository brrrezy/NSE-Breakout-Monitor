# NSE Breakout Detection Engine

An elite, institutional-grade breakout detection system for NSE stocks. Identifies stocks **before** they break out by analyzing market structure, smart money footprints, volatility compression, and 40+ technical indicators — then scores every candidate on a 0-100 composite institutional scale.

Runs on GitHub Actions — **$0 monthly cost**. Uses only free data sources (yfinance, Google News RSS, Groq free tier).

---

## 🧠 What Makes This Different

This isn't a simple screener. It's a multi-layered detection engine:

| Layer | What It Does |
|-------|-------------|
| **40+ Indicators** | Trend (EMAs, ADX, Supertrend, MA Ribbon), Momentum (RSI, MACD, Stoch RSI, CCI, ROC, divergence), Volume (OBV, CMF, MFI, VWAP, Volume Profile), Volatility (ATR, BB Squeeze, Keltner, Donchian, NR7) |
| **Market Structure** | Swing highs/lows, HH/HL/LH/LL classification, trendline fitting, breakout pressure, compression detection |
| **Smart Money (SMC)** | Order Blocks, Fair Value Gaps, liquidity sweeps, equal highs/lows, rejection candles, absorption detection, premium/discount zones |
| **Chart Patterns** | Ascending/Descending/Symmetrical Triangles, Bull Flags, Cup & Handle, Darvas Box, Double Bottom, Flat Base |
| **13-Dimension Scoring** | Technical, Momentum, Volume, Trend, Volatility, Relative Strength, Pattern, News, Risk, Market, Liquidity, Base Quality, Breakout Probability → Composite 0-100 score |
| **News Sentiment** | Google News RSS + Groq AI analysis per ticker with time-decay weighting |
| **Risk Management** | 10 hard rejection filters (liquidity, trend, VIX, R:R, extension, etc.) |
| **Adaptive Learning** | SQLite outcome tracking, rolling performance metrics, automatic weight optimization |

---

## 📖 Strategy Overview

Based on the Momentum Burst method combined with institutional SMC concepts:

> A stock in an uptrend that has rested quietly for 5-15 days with volatility contraction, showing smart money accumulation, near a clear resistance level — with compressed Bollinger Bands, institutional order blocks nearby, and strong relative strength vs Nifty — is scored and ranked before the breakout happens.

### The 3-Phase Detection

| Phase | What | Checks |
|-------|------|--------|
| **Market Context** | Is the environment favorable? | Nifty 50 regime, India VIX, breadth, Bank Nifty |
| **Phase 1 — Uptrend** | Does the stock have prior momentum? | EMA stack (8>21>55), EMA-144 rising, U/D ≥ 1.4, near 120d high |
| **Phase 2 — Base** | Is the stock coiling near resistance? | 5-15 day base, quiet volume, VCP, tightening, squeeze, accumulation |
| **Phase 3 — Breakout** | Is today the breakout day? | Close > pivot, body > 60%, RVol ≥ 2x, confluence ≥ 4/6 |

### Composite Institutional Score (0-100)

Every stock gets scored across 13 dimensions with configurable weights:

```
Technical (10%) + Momentum (10%) + Volume (10%) + Trend (10%)
+ Relative Strength (8%) + Pattern (8%) + Liquidity (8%)
+ Risk (8%) + Base Quality (7%) + News (6%)
+ Market (5%) + Volatility (5%) + Breakout Probability (5%)
```

Only stocks above the configurable threshold (default: 65) get shortlisted.

---

## ⏰ Schedule (IST)

| Time | Purpose |
|------|---------| 
| **9:10 AM** | Pre-market — check watchlist stocks for gaps/setups |
| **11:00 AM** | Mid-morning — catch breakouts in the first session |
| **3:35 PM** | End of day — final scan, build tomorrow's watchlist, performance report (Fridays) |

---

## 📊 Alert Output

Every shortlisted stock includes:

- **Price levels**: Entry zone, Stop Loss, Target 1/2/3
- **Scores**: Composite (0-100), Confidence, Breakout Probability
- **Analysis**: Market structure, liquidity, pattern, volume, momentum, RS
- **News**: Sentiment score, catalyst type, time-decayed impact
- **Risk**: Risk level, R:R ratio, failure conditions
- **Reasoning**: Why selected, concerns, monitoring frequency

### Sample Telegram Alert

```
🚀 CONFIPET — BREAKOUT ALERT
════════════════════════════
📍 Entry  : ₹257
🛡️ Stop   : ₹249
🎯 T1     : ₹265
🎯 T2     : ₹273
🎯 T3     : ₹297
⚖️ R:R    : 2.0:1
📈 RVol   : 2.3x
────────────────────────────
📊 Score  : 78/100
🎲 Prob   : 72%
💪 RS     : +8% vs Nifty
🔲 Pattern: Ascending Triangle
🧠 Edge   : VCP | Coil | Tight | Squeeze
📝 Base   : 10d, SFP Doji:3
────────────────────────────
```

---

## 🏗 Architecture

```
NSE-Breakout-Monitor/
├── config/
│   ├── settings.py              # All configurable parameters
│   └── __init__.py
├── core/
│   ├── engine.py                # Main orchestrator
│   ├── data_provider.py         # yfinance + parquet caching
│   └── universe.py              # Nifty 500, sectors, state
├── indicators/
│   ├── trend.py                 # EMAs, ADX, Supertrend, MA Ribbon
│   ├── momentum.py              # RSI, MACD, Stoch RSI, CCI, ROC
│   ├── volume.py                # OBV, CMF, MFI, VWAP, Vol Profile
│   └── volatility.py            # ATR, BB Squeeze, Keltner, NR7
├── analysis/
│   ├── market_structure.py      # Swing points, HH/HL, trendlines
│   ├── liquidity.py             # Order Blocks, FVGs, sweeps, SMC
│   ├── patterns.py              # Chart patterns (7 types)
│   ├── base_detection.py        # Phase 2: consolidation base
│   ├── breakout.py              # Phase 3: breakout confirmation
│   ├── strength.py              # RS vs index, beta, sector rotation
│   └── market_context.py        # Nifty regime, VIX, breadth
├── scoring/
│   ├── scorer.py                # 13-dimension scoring engine
│   ├── ranker.py                # Multi-criteria ranking
│   └── risk_manager.py          # 10 rejection filters
├── news/
│   ├── fetcher.py               # Google News RSS aggregation
│   └── sentiment.py             # Groq AI sentiment analysis
├── learning/
│   ├── tracker.py               # SQLite performance tracking
│   └── optimizer.py             # Adaptive weight optimization
├── alerts/
│   ├── telegram.py              # Message sending + splitting
│   └── formatter.py             # Structured output formatting
├── tests/                       # 51 unit tests
├── breakout_scanner.py          # Entry point
├── config.yaml                  # User configuration
└── requirements.txt
```

---

## 🛠 Setup

### 1. Create a Telegram Bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and save the **Bot Token**
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your **Chat ID**

### 2. Get a Free Groq API Key
1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key (free tier: 30 req/min)

### 3. Add GitHub Secrets
Go to your repo → **Settings** → **Secrets and variables** → **Actions**:

| Secret Name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Your bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |
| `GROQ_API_KEY` | Your Groq API key |

### 4. Enable the Workflow
The scanner runs automatically via GitHub Actions cron (Mon-Fri, 3x daily). You can also trigger it manually from the **Actions** tab.

### 5. Run Locally (Optional)
```bash
pip install -r requirements.txt
python breakout_scanner.py
python breakout_scanner.py --test-alert      # Test Telegram
python breakout_scanner.py --perf-report     # View performance
```

---

## ⚙️ Configuration

Edit `config.yaml` to tune parameters without changing code:

```yaml
# Price filters
min_price: 50
max_price: 5000

# Scoring
scoring_shortlist_threshold: 65    # Minimum score to shortlist
scoring_weights:
  technical: 0.10
  momentum: 0.10
  volume: 0.10
  # ... (see config.yaml for all weights)

# Risk
risk_max_pct: 0.03                 # Max stop-loss distance
risk_rr_min: 1.5                   # Minimum R:R ratio

# Feature flags
enable_news_engine: true
enable_learning: true
enable_liquidity_analysis: true
enable_pattern_detection: true
```

---

## ⚠️ Risk Management

The engine enforces **10 hard rejection filters**:

1. ❌ Low liquidity (< ₹10Cr avg turnover)
2. ❌ Weak trend (ADX < 15 + poor ribbon alignment)
3. ❌ Bear market + high beta stock
4. ❌ Negative news catalyst
5. ❌ Poor volume on breakout day (RVol < 0.5)
6. ❌ Late-stage / already extended (> 15% above pivot)
7. ❌ High risk (stop > 5% from entry)
8. ❌ Unfavorable R:R (< 1.5:1)
9. ❌ RSI overbought on breakout (> 85)
10. ❌ India VIX extreme (> 30)

---

## 📈 Adaptive Learning

The engine tracks every alert and checks outcomes after 5 trading days:

- **Win**: Price hit Target 1
- **Loss**: Price hit Stop Loss
- **False Breakout**: Reversed below pivot within 3 days
- **Missed**: Watchlist stock never broke out

Weekly performance reports are sent via Telegram (Fridays at EOD) with:
win rate, precision, average return, and Sharpe ratio.

After 50+ tracked alerts, the optimizer automatically adjusts scoring weights based on which dimensions correlated most with winning trades.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v    # 51 tests
```

---

## Free Data Sources Used

| Source | Data | Rate Limit |
|--------|------|------------|
| **yfinance** | OHLCV, indices (Nifty 50, VIX, Bank Nifty) | ~2000 req/hour |
| **NSE Archives** | Nifty 500 constituent list + sectors | 1 req/day |
| **Google News RSS** | News headlines per ticker | ~2 req/sec |
| **Groq API (free)** | AI sentiment analysis + verdict | 30 req/min |

---

## License

Copyright © 2026 Shivanshu Srivastav. All rights reserved.
