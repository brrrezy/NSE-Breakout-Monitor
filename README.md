# NSE Breakout Scanner Pro

A **fully automated, institutional-grade breakout detection engine** for NSE stocks. It combines **Mark Minervini's VCP methodology** with advanced statistical backtesting and volume-price analysis.

Runs entirely on GitHub Actions — **$0 monthly cost**.

## 🚀 Advanced Features

### 1. The VCP 2.0 Engine
*   **Tightness Scoring (1-10):** Uses the math of standard deviation to measure consolidation. A score of `9/10` means the price is coiling like a spring.
*   **Institutional Footprint (Absorption):** Detects narrow-spread candles on massive volume, identifying "quiet" institutional accumulation before the breakout happens.
*   **IPO Base & SFP:** Specialized detection for recent listings and Swing Failure Patterns (fake-out reversals).

### 2. Historical Win-Rate Predictor (WinProb)
*   Whenever a stock triggers, the engine runs an **on-the-fly mini-backtest** over the last 1 year of that specific stock's history.
*   It calculates the success rate of similar breakouts reaching a +10% target.
*   Look for the `⭐` in your alerts for setups with >65% historical win probability.

### 3. Sector Intelligence & Group Moves
*   **Tailwind Detection:** Identifies the sector for every stock.
*   **Hot Sector Heatmap:** At the End of Day, the bot groups your watchlist by sector to identify where "smart money" is flowing.
*   **Group Moves:** Detects when multiple stocks in the same sector (e.g., Chemicals or IT) are breaking out simultaneously—a signal with a much higher success rate.

### 4. Smart Risk Management
*   **Price Filter:** Automatically filters out stocks > ₹5,000 to ensure your swing trading position sizing remains efficient.
*   **Fakeout Protection:** Only triggers a breakout if the stock shows a "Strong Close" (upper 60% of daily range), avoiding "wick traps."

---

## ⏰ How It Works (IST)

| Time | Event |
|---|---|
| **09:10** | **Morning Priority:** Scans your watchlist for overnight setups. |
| **09:15 - 15:30** | **Live Monitor:** Scans every **10 minutes** for new breakouts. |
| **15:30** | **Market Close:** Final scan → Generates Top 10 Watchlist & Hot Sectors for tomorrow. |

---

## 🛠 Setup (5 minutes)

### 1. Create a Telegram Bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram.
2. Send `/newbot` and save the **Bot Token**.
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your **Chat ID**.

### 2. Add GitHub Secrets
Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |

---

## 📊 Stock Status Guide

| Status | Meaning |
|---|---|
| `⭐ ACTIONABLE` | Elite setup + Breakout trigger. High probability of immediate move. |
| `• WATCHLIST` | High-quality setup forming (VCP/IPO Base). Add to tracking list. |
| `TRENDING` | In Stage 2 trend. Waiting for price contraction. |

---

## License
MIT
