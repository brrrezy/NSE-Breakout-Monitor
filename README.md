# NSE Momentum Burst Scanner

An automated breakout detection engine for NSE stocks, implementing the **Momentum Burst Strategy** for Indian markets. Scans the Nifty 500 universe daily for stocks forming tight bases near resistance and alerts you on Telegram the moment they break out.

Runs on GitHub Actions — **$0 monthly cost**.

---

## 📖 Strategy Overview

Based on the Momentum Burst method (Pradeep Bonde) combined with swing trading best practices:

> A stock that has been trending up, rested quietly for 5–15 days with doji candles and SFPs near a clear resistance level, and today broke out above that level with a strong body candle on 1.5x+ volume while the overall market is in an uptrend.

### The 3-Phase Detection

| Phase | What | Checks |
|-------|------|--------|
| **Market Regime** | Is the market healthy? | Nifty 50 above EMA-21 and EMA-55, both pointing up |
| **Phase 1 — Uptrend** | Does the stock have prior momentum? | EMA-8 > EMA-21 > EMA-55, EMA-144 rising, U/D ratio ≥ 1.4 |
| **Phase 2 — Base** | Is the stock resting near resistance? | 5–15 day tight consolidation, small candles, quiet volume, no close below EMA-21, SFP/Doji patterns |
| **Phase 3 — Breakout** | Is today the breakout day? | Close > pivot, body > 60% of range, close in top 10%, RVol ≥ 1.5x, move ≥ 4%, confluence ≥ 4/6 |

### Confluence Check (need 4 of 6)

- MACD: positive histogram
- RSI: above 55
- Stochastic: above 50
- Volume: expanding vs base
- Trend: EMA-8 > EMA-21 > EMA-55
- ADX: above 20

---

## ⏰ Schedule (IST)

| Time | Purpose |
|------|---------|
| **9:10 AM** | Pre-market — check watchlist stocks for gaps/setups |
| **11:00 AM** | Mid-morning — catch breakouts in the first session |
| **3:35 PM** | End of day — final scan, build tomorrow's watchlist |

---

## 📊 Alert Types

| Status | Meaning |
|--------|---------|
| 🚀 **ACTIONABLE** | Breakout confirmed today. All Phase 3 checks passed. Entry candidate. |
| • **WATCHLIST** | Stock in a valid base near resistance. Watch for breakout tomorrow. |

### Sample Telegram Alert

```
🚀 CONFIPET — BREAKOUT

Entry: ₹57
Stop: ₹54 | Target: ₹63
R:R 2.0:1 | RVol: 2.3x

Base:12d, SFP, Doji:3, Conf:5/6
```

---

## 🛠 Setup

### 1. Create a Telegram Bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and save the **Bot Token**
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your **Chat ID**

### 2. Add GitHub Secrets
Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Your bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |

### 3. Enable the Workflow
The scanner runs automatically via GitHub Actions cron (Mon–Fri). You can also trigger it manually from the **Actions** tab.

---

## 📁 Project Structure

```
├── breakout_scanner.py          # Main scanner engine
├── requirements.txt             # Python dependencies
├── swing_trading_playbook.pdf   # Strategy reference document
├── Chart-Snippets/              # Reference chart patterns
├── .github/workflows/
│   └── nse_breakout_monitor.yml # GitHub Actions workflow (3x daily cron)
└── watchlist_persistent.json    # Runtime state (auto-generated, gitignored)
```

---

## ⚠️ Risk Management

The scanner enforces strict playbook rules:

- **Stop Loss**: Low of the breakout candle — non-negotiable
- **Max Risk**: Warns if stop is > 3% from entry (skip the trade)
- **Position Sizing**: Risk 0.75% of capital per trade (calculate manually)
- **No Chasing**: Rejects stocks up 3+ consecutive days before breakout

---

## License

Copyright © 2026 Shivanshu Srivastav. All rights reserved.
