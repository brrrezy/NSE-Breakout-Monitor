# NSE Breakout Scanner

A **free, automated breakout detection engine** for NSE stocks, powered by **Mark Minervini's VCP (Volatility Contraction Pattern)** methodology. Runs entirely on GitHub Actions — no servers, no hosting costs.

## How It Works

```
09:10 IST  →  Pre-open: Scan yesterday's watchlist for gap-ups
09:20 IST  →  Full market scan begins
Every 20m  →  Re-scan entire NSE for new breakouts
15:30 IST  →  Final scan → Save tomorrow's watchlist
```

### Detection Pipeline

1. **Trend Filter** — Minervini Stage 2 Template (price > 50 EMA > 150 EMA > 200 EMA)
2. **Setup Detection** — VCP (tightness + volume drying), SFP (swing failure), IPO Base
3. **Breakout Trigger** — Price breaks pivot high with 1.5x volume expansion
4. **Alert** — Instant Telegram notification with price, pivot, and signal details

### Stock Classification

| Status | Meaning |
|---|---|
| `ACTIONABLE` | Breakout confirmed — entry trigger fired |
| `WATCHLIST` | Setup forming — monitor for breakout |
| `TRENDING` | In Minervini trend — no setup yet |

## Setup (5 minutes)

### 1. Create a Telegram Bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Save the **Bot Token**
4. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to get your **Chat ID**

### 2. Add GitHub Secrets
Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |

### 3. Done
The scanner runs automatically every trading day during NSE market hours.

## Files

| File | Purpose |
|---|---|
| `breakout_scanner.py` | Core engine — all detection logic |
| `.github/workflows/nse_breakout_monitor.yml` | GitHub Actions schedule & config |
| `watchlist_persistent.json` | Auto-managed — carries watchlist between days |
| `requirements.txt` | Python dependencies |

## Manual Test

Trigger a test scan: Go to **Actions** → **NSE Breakout Scanner** → **Run workflow**

Test Telegram connection:
```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python3 breakout_scanner.py --test-alert
```

## Cost

**$0**. GitHub Actions Free tier provides 2,000 minutes/month. This scanner uses ~540 minutes/month (~27%).

## License

MIT
