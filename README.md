# 🚀 NSE Swing Screener: Institutional Terminal

An institutional-grade quantitative screening terminal for the **National Stock Exchange (NSE) of India**. Designed for hyper-speed analysis, predicting high-probability swing setups using state-of-the-art technical models.

![License](https://img.shields.io/badge/License-Proprietary-red.svg)
![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)
![Framework](https://img.shields.io/badge/Backend-FastAPI-green.svg)
![UI](https://img.shields.io/badge/UI-Vanilla%20JS-orange.svg)

---

## ⚡ Key Highlights

- **Hyper-Speed Scanning**: Analyzes the entire NSE universe (2,600+ stocks) in **under 60 seconds** using bulk price downloads and parallel finalist processing.
- **Smart Data Vault**: Integrated SQLite caching reduces network overhead by 90% on repeated scans.
- **Institutional Predictors**: 
  - **Minervini Trend Template**: Identifies Stage 2 uptrends.
  - **RS Rating (Relative Strength)**: Percentile ranking (0-99) of every stock relative to the entire market.
  - **Setup Detectors**: VCP (Volatility Contraction), SFP (Swing Failure), and IPO Base detection.
- **Modern Dashboard**: A sleek, institutional dark-mode UI with live ticker autocomplete and interactive table filtering.

---

## 📂 Project Structure

```text
.
├── main.py                    # FastAPI Backend & API Layer
├── swing_screener.py          # Core Quantitative Engine
├── static/                    # Frontend Assets (HTML/CSS/JS)
├── nse_screener_cache.db      # SQLite Fundamental Data Vault
├── requirements.txt           # Python Dependencies
└── Procfile                   # Deployment Config for Railway/Render
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/brrrezy/nse_screrner.git
cd nse_screrner
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
python main.py
```
Open your browser to `http://localhost:8000`.

---

## 📈 Quantitative Engine Details

The terminal uses a **Confluence Scoring Model (0-13)**:
- **Technical Confluence (0-8)**: RSI, MACD, Stochastics, EMA Stack, Volume Multiplier, ADX, and CLV Delta.
- **Setup Bonus (0-4)**: 
  - VCP Detected (+2)
  - Swing Failure (+2)
  - IPO Base (+1)
  - Minervini Stage 2 (+1)
- **Fundamental Quality (0-1)**: Based on ROE, Debt/Equity, and Revenue Growth.

---

## 🚀 Deployment (Railway.app)

This project is ready for one-click deployment on **Railway**:
1. Connect your GitHub repository to Railway.
2. Railway will automatically detect the `Procfile` and `requirements.txt`.
3. Add a **Persistent Volume** if you want to keep the `nse_screener_cache.db` between restarts.

---

## 🤝 Credits & Support

Made with ❤️ by **[brrrezy](https://github.com/brrrezy)** in India.

- **Portfolio**: [shivanshusr.vercel.app](https://shivanshusr.vercel.app/)
- **GitHub**: [@brrrezy](https://github.com/brrrezy)

---

## 📝 Legal Notice & Advisory

**Advisory**: This tool is for educational and research purposes only. I am not a SEBI-registered advisor. Trading in the stock market involves a high risk of loss. Always conduct your own research or consult with a professional financial advisor before making any investment decisions.

---

Happy Trading! 📊📉
