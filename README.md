# CAN SLIM Daily Screener

Screens the **S&P 500** every weekday morning and emails you the **top 10 stocks** scored against William O'Neil's CAN SLIM fundamentals.

## How it works

| Criterion | Weight | What is measured |
|-----------|--------|-----------------|
| **C** — Current quarterly earnings | 25 pts | EPS growth YoY ≥ 25% |
| **A** — Annual earnings growth | 20 pts | 3-year EPS CAGR |
| **N** — New highs | 15 pts | % of 52-week high (breakout proxy) |
| **S** — Supply & demand | 15 pts | Volume surge + small float |
| **L** — Leader vs laggard | 15 pts | 12-month relative strength vs SPY |
| **I** — Institutional ownership | 10 pts | Sweet spot 30–70% institutional |

**M (Market direction)** is shown as context in the email — SPY above/below its 50 & 200-day MAs.

Data is fetched from Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance).  
The job runs Monday–Friday at **06:00 UTC (08:00 CEST / 07:00 CET)**.

---

## One-time setup

### 1. Get a Gmail App Password

You need a **Gmail App Password** (not your regular password):

1. Go to [myaccount.google.com/security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** if not already on
3. Search for "App passwords" → Create one named `canslim-screener`
4. Copy the 16-character password shown (you won't see it again)

### 2. Add GitHub Secrets

In this repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|-------------|-------|
| `GMAIL_ADDRESS` | Your Gmail address (e.g. `you@gmail.com`) |
| `GMAIL_APP_PASSWORD` | The 16-char App Password from step 1 |
| `RECIPIENT_EMAIL` | Where to send the email (can be same address) |

### 3. Enable Actions (if needed)

Go to the **Actions** tab and click **"I understand my workflows, go ahead and enable them"** if prompted.

### 4. Test it manually

Actions tab → **CAN SLIM Daily Screener** → **Run workflow** → watch the logs.  
You should receive an email within ~20 minutes.

---

## Run locally

```bash
pip install -r requirements.txt
export GMAIL_ADDRESS="you@gmail.com"
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
export RECIPIENT_EMAIL="you@gmail.com"
python screener.py
```

---

> **Disclaimer:** Not financial advice. Always do your own research before investing.
