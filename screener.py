#!/usr/bin/env python3
"""CAN SLIM daily screener — emails top 10 S&P 500 watchlist."""

import io
import os
import logging
import resend
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_WORKERS = 4

# Max points per criterion (total = 100)
WEIGHTS = {"C": 25, "A": 20, "N": 15, "S": 15, "L": 15, "I": 10}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (canslim-screener/1.0; +https://github.com/ai-kento/canslim-screener)"},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def _yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    return s


def get_spy_benchmark() -> dict:
    hist = yf.Ticker("SPY", session=_yf_session()).history(period="1y")
    if hist.empty:
        return {}
    close = hist["Close"]
    current = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    return {
        "return_1y": (current / float(close.iloc[0]) - 1) * 100,
        "above_ma50": current > ma50,
        "above_ma200": current > ma200,
    }


# ---------------------------------------------------------------------------
# CAN SLIM scoring
# ---------------------------------------------------------------------------

def _c_score(stock: yf.Ticker) -> tuple[float, str]:
    """C — Current quarterly EPS growth ≥25% YoY."""
    try:
        q = stock.quarterly_income_stmt
        if q is None or q.empty:
            return 0, ""
        # Find an EPS-like row
        for row in ("Diluted EPS", "Basic EPS"):
            if row in q.index:
                eps = q.loc[row].dropna()
                if len(eps) >= 5:
                    recent, year_ago = float(eps.iloc[0]), float(eps.iloc[4])
                    if year_ago != 0:
                        growth = (recent - year_ago) / abs(year_ago) * 100
                        score = min(25, max(0, growth)) if growth >= 25 else max(0, growth * 0.5)
                        return score, f"{growth:+.0f}%"
        # Fallback: net income growth
        if "Net Income" in q.index:
            ni = q.loc["Net Income"].dropna()
            if len(ni) >= 5:
                r, ya = float(ni.iloc[0]), float(ni.iloc[4])
                if ya != 0:
                    g = (r - ya) / abs(ya) * 100
                    return min(25, max(0, g * 0.7)), f"NI {g:+.0f}%"
    except Exception:
        pass
    return 0, ""


def _a_score(stock: yf.Ticker) -> tuple[float, str]:
    """A — Annual EPS growth, 3-year CAGR."""
    try:
        a = stock.income_stmt
        if a is None or a.empty:
            return 0, ""
        for row in ("Diluted EPS", "Basic EPS", "Net Income"):
            if row in a.index:
                eps = a.loc[row].dropna()
                if len(eps) >= 4:
                    newest, oldest = float(eps.iloc[0]), float(eps.iloc[3])
                    if oldest > 0 and newest > 0:
                        cagr = ((newest / oldest) ** (1 / 3) - 1) * 100
                        return min(20, max(0, (cagr / 25) * 20)), f"{cagr:+.0f}%/yr"
    except Exception:
        pass
    return 0, ""


def _n_score(info: dict) -> tuple[float, str]:
    """N — Near 52-week high (proxy for new product / breakout)."""
    high = info.get("fiftyTwoWeekHigh")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if high and price and high > 0:
        pct = (price / high) * 100
        if pct >= 95:
            return 15, f"{pct:.0f}% of 52wk high"
        elif pct >= 85:
            return 10, f"{pct:.0f}% of 52wk high"
        elif pct >= 75:
            return 5, f"{pct:.0f}% of 52wk high"
        return 0, f"{pct:.0f}% of 52wk high"
    return 0, ""


def _s_score(info: dict) -> tuple[float, str]:
    """S — Supply & demand: volume surge + small float."""
    score, parts = 0.0, []
    avg_vol = info.get("averageVolume") or 0
    cur_vol = info.get("volume") or 0
    if avg_vol > 0 and cur_vol > 0:
        ratio = cur_vol / avg_vol
        score += min(8.0, ratio * 5)
        parts.append(f"vol {ratio:.1f}x avg")
    float_sh = info.get("floatShares") or 0
    sh_out = info.get("sharesOutstanding") or 0
    if float_sh > 0 and sh_out > 0:
        float_pct = float_sh / sh_out * 100
        # Lower float = tighter supply = higher score
        score += max(0.0, 7.0 - float_pct / 15)
        parts.append(f"float {float_pct:.0f}%")
    return min(15.0, score), " | ".join(parts)


def _l_score(stock: yf.Ticker, spy_return_1y: float) -> tuple[float, str]:
    """L — Leader: relative strength vs SPY over 12 months."""
    try:
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 200:
            return 0, ""
        stock_ret = (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100
        rs = stock_ret - spy_return_1y
        # Full 15 pts if +20% above SPY; 0 if >20% below SPY
        score = min(15.0, max(0.0, (rs + 20) / 40 * 15))
        return score, f"RS {rs:+.0f}% vs SPY"
    except Exception:
        return 0, ""


def _i_score(info: dict) -> tuple[float, str]:
    """I — Institutional ownership: sweet spot 30–70%."""
    inst = info.get("heldPercentInstitutions") or 0
    if isinstance(inst, float) and inst <= 1.0:
        inst *= 100
    if inst <= 0:
        return 0, ""
    if 30 <= inst <= 70:
        score = 10
    elif 20 <= inst < 30 or 70 < inst <= 85:
        score = 6
    else:
        score = 2
    return float(score), f"inst {inst:.0f}%"


def score_stock(ticker: str, spy_data: dict) -> dict | None:
    try:
        stock = yf.Ticker(ticker, session=_yf_session())
        info = stock.info
        if not info or info.get("quoteType") != "EQUITY":
            return None
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or price <= 0:
            return None

        c, c_detail = _c_score(stock)
        a, a_detail = _a_score(stock)
        n, n_detail = _n_score(info)
        s, s_detail = _s_score(info)
        l, l_detail = _l_score(stock, spy_data.get("return_1y", 0))
        i, i_detail = _i_score(info)

        details = {k: v for k, v in {
            "C": c_detail, "A": a_detail, "N": n_detail,
            "S": s_detail, "L": l_detail, "I": i_detail,
        }.items() if v}

        return {
            "ticker": ticker,
            "name": (info.get("longName") or ticker)[:38],
            "sector": info.get("sector") or "N/A",
            "price": price,
            "score": c + a + n + s + l + i,
            "details": details,
        }
    except Exception as e:
        log.debug(f"{ticker} skipped: {e}")
        return None


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def market_label(spy: dict) -> str:
    if not spy:
        return "⚪ Unknown"
    if spy["above_ma50"] and spy["above_ma200"]:
        return "🟢 Confirmed Uptrend (SPY above 50 & 200 MA)"
    elif spy["above_ma200"]:
        return "🟡 Weakening — SPY below 50-day MA"
    return "🔴 Downtrend — SPY below 200-day MA (use caution)"


def build_email_html(top10: list[dict], spy: dict) -> str:
    date_str = datetime.now().strftime("%B %d, %Y")
    spy_ret = spy.get("return_1y", 0)

    rows = ""
    for rank, s in enumerate(top10, 1):
        bg = "#f4f7ff" if rank % 2 == 0 else "#ffffff"
        facts = "  ·  ".join(f"<b>{k}:</b> {v}" for k, v in s["details"].items())
        bar_width = int(s["score"])
        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:12px 8px;text-align:center;font-size:22px;font-weight:bold;color:#1a237e;width:36px">{rank}</td>
          <td style="padding:12px 8px;font-weight:bold;font-size:15px;white-space:nowrap">{s['ticker']}</td>
          <td style="padding:12px 8px;font-size:13px">{s['name']}</td>
          <td style="padding:12px 8px;font-size:11px;color:#666">{s['sector']}</td>
          <td style="padding:12px 8px;font-weight:bold;white-space:nowrap">${s['price']:,.2f}</td>
          <td style="padding:12px 8px;text-align:center;white-space:nowrap">
            <div style="background:#e8eaf6;border-radius:8px;height:18px;width:80px;display:inline-block;vertical-align:middle">
              <div style="background:#1a237e;border-radius:8px;height:18px;width:{bar_width}px"></div>
            </div>
            <span style="font-weight:bold;margin-left:6px">{s['score']:.0f}<small style="color:#999">/100</small></span>
          </td>
          <td style="padding:12px 8px;font-size:11px;color:#444;line-height:1.6">{facts}</td>
        </tr>"""

    legend = "".join(
        f"<span style='margin-right:14px'><b>{k}</b>&nbsp;{v}pts</span>"
        for k, v in WEIGHTS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><body style="font-family:Arial,Helvetica,sans-serif;background:#eef0f5;margin:0;padding:20px">
<div style="max-width:1000px;margin:auto">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;padding:24px 28px;border-radius:10px 10px 0 0">
    <h1 style="margin:0;font-size:22px;letter-spacing:.5px">📊 CAN SLIM Top 10 Watchlist</h1>
    <p style="margin:6px 0 0;opacity:.75;font-size:13px">{date_str} &nbsp;·&nbsp; S&amp;P 500 Universe</p>
  </div>

  <!-- Market signal -->
  <div style="background:#fff;padding:14px 20px;border-left:5px solid #1a237e;font-size:13px">
    <strong>Market Direction (M):</strong> {market_label(spy)}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>SPY 12-month return:</strong> {spy_ret:+.1f}%
  </div>

  <!-- Table -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;background:#fff;border:1px solid #dde">
    <tr style="background:#e8eaf6;font-size:11px;font-weight:bold;color:#1a237e">
      <th style="padding:10px 8px">#</th>
      <th style="padding:10px 8px;text-align:left">Ticker</th>
      <th style="padding:10px 8px;text-align:left">Company</th>
      <th style="padding:10px 8px;text-align:left">Sector</th>
      <th style="padding:10px 8px;text-align:left">Price</th>
      <th style="padding:10px 8px">Score</th>
      <th style="padding:10px 8px;text-align:left">CAN SLIM Factors</th>
    </tr>
    {rows}
  </table>

  <!-- Legend -->
  <div style="background:#fff;padding:12px 20px;font-size:11px;color:#555;border-top:1px solid #eee">
    <strong>Scoring key:</strong>&nbsp; {legend}
  </div>

  <!-- Disclaimer -->
  <div style="background:#fff8e1;padding:14px 20px;font-size:11px;color:#795548;border-radius:0 0 10px 10px;border-top:2px solid #ffe082">
    ⚠️ <strong>Not financial advice.</strong> CAN SLIM is William O'Neil's growth-investing methodology.
    Always do your own due diligence. Data via Yahoo Finance (yfinance). Generated automatically by GitHub Actions.
  </div>

</div>
</body></html>"""


def send_email(html: str):
    resend.api_key = os.environ["RESEND_API_KEY"]
    recipient = os.environ["RECIPIENT_EMAIL"]
    date_str = datetime.now().strftime("%b %d, %Y")

    resend.Emails.send({
        "from": "CAN SLIM Screener <onboarding@resend.dev>",
        "to": [recipient],
        "subject": f"📊 CAN SLIM Top 10 — {date_str}",
        "html": html,
    })
    log.info(f"Email sent to {recipient}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== CAN SLIM Screener starting ===")

    log.info("Fetching SPY benchmark...")
    spy = get_spy_benchmark()
    log.info(f"SPY 1yr return: {spy.get('return_1y', 'N/A'):.1f}%  "
             f"above50MA={spy.get('above_ma50')}  above200MA={spy.get('above_ma200')}")

    log.info("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    log.info(f"Screening {len(tickers)} stocks ({MAX_WORKERS} workers)...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(score_stock, t, spy): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                log.info(f"  Progress: {done}/{len(tickers)}")
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    top10 = results[:10]

    log.info("Top 10 results:")
    for rank, s in enumerate(top10, 1):
        log.info(f"  {rank:2}. {s['ticker']:<6} score={s['score']:.1f}  {s['name']}")

    html = build_email_html(top10, spy)
    send_email(html)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
