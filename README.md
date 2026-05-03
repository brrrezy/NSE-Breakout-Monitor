# 🚀 NSE Swing Screener: Professional-Grade Trading Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/Market-NSE%20India-orange.svg)](https://www.nseindia.com/)
[![Automation](https://img.shields.io/badge/Automation-GitHub%20Actions-green.svg)](https://github.com/features/actions)

A high-performance, automated quantitative screening terminal designed for the **National Stock Exchange (NSE)**. This engine identifies high-probability swing trading setups using institutional-grade technical models, including **Minervini's Trend Template** and **Volatility Contraction Patterns (VCP)**.

---

## ⚡ Key Capabilities

### 🔍 Advanced Quantitative Engine
- **Minervini Stage 2 Template**: Full implementation of Mark Minervini's trend requirements (50/150/200 EMA stacks, 52-week high/low proximity, and trend persistence).
- **VCP Detector**: Identifies volatility contraction setups with price tightening, volume drying, and pivot-point proximity.
- **RS Rating (Relative Strength)**: Calculates the 0-99 percentile ranking for every stock relative to the entire market, mimicking the IBD/MarketSmith model.
- **Confluence Scoring (0-15)**: A weighted multi-factor model combining RSI, MACD, Volume Multiplier, ADX, and CLV (Close Location Value) Delta.
- **SFP & IPO Base Detection**: Specialized logic for Swing Failure Patterns and high-momentum IPO breakouts.

### 🤖 Autonomous Operations
- **GitHub Actions Integration**: Runs automatically every 10 minutes during NSE trading hours (09:15 - 15:30 IST).
- **Telegram Alerts**: Delivers instant notifications for **ACTIONABLE** breakouts and **WATCHLIST** setups directly to your phone.
- **Persistent Memory**: Maintains a cross-session watchlist using SQLite and file caching to track setups as they mature from watchlist to entry.

### 📊 Performance & UI
- **Hyper-Scanning**: Analyzes 2,600+ symbols in under 60 seconds using parallel processing and bulk data downloads.
- **Sleek Dashboard**: A modern, dark-mode web interface built with FastAPI and Vanilla JS for real-time manual scans.

---

## 📂 Project Architecture

```text
.
├── swing_screener.py          # Core Quantitative & Alerting Engine
├── main.py                    # FastAPI Web Server & API Layer
├── static/                    # Frontend: HTML/CSS/JS Dashboard
├── .github/workflows/         # Automation: Live Market Monitor
├── nse_screener_cache.db      # SQLite Fundamental Data Vault
└── watchlist_persistent.json  # Cross-session memory for setups
```

---

## 🛠️ Installation & Setup

### 1. Local Environment
```bash
# Clone the repository
git clone https://github.com/brrrezy/nse-screener.git
cd nse-screener

# Install requirements
pip install -r requirements.txt

# Run the web dashboard
python main.py
```
Access the dashboard at `http://localhost:8000`.

### 2. Configure Telegram Alerts
To receive live alerts, set the following environment variables:
- `TELEGRAM_BOT_TOKEN`: Your bot token from @BotFather.
- `TELEGRAM_CHAT_ID`: Your personal chat ID or Group ID.

### 3. Deploy Automation (GitHub Actions)
1. Fork/Push this repo to GitHub.
2. Go to **Settings > Secrets and variables > Actions**.
3. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as **Repository Secrets**.
4. The workflow in `.github/workflows/monitor.yml` will automatically start on the next market session.

---

## 📈 Technical Confluence Model

The engine scores stocks based on a **15-point confluence model**:

| Category | Factor | Points |
| :--- | :--- | :--- |
| **Trend** | Minervini Stage 2 | +1 |
| **Setup** | VCP Detected | +2 |
| **Setup** | Swing Failure / IPO Base | +1 to +2 |
| **Trigger** | Breakout + Volume Expansion | +5 |
| **Momentum** | RSI > 55 & MACD Bullish | +2 |
| **Strength** | RS Rating > 80 | +1 |
| **Fundamentals** | ROE > 15%, Positive Revenue Growth | +1 |

---

## 📝 Legal Disclaimer

**Not Financial Advice**: This software is provided for educational and research purposes only. The author is not a SEBI-registered advisor. Trading in financial markets involves significant risk. Always perform your own due diligence or consult a certified financial professional before making investment decisions.

---

## 🤝 Support & Contribution

Created by **[brrrezy](https://github.com/brrrezy)**.
- **Portfolio**: [shivanshusr.vercel.app](https://shivanshusr.vercel.app/)
- **GitHub**: [@brrrezy](https://github.com/brrrezy)

If you find this tool useful, consider giving it a ⭐!
