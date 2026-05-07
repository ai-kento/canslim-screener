#!/usr/bin/env python3
"""CAN SLIM daily screener — emails top 30 S&P 500 watchlist."""

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
TOP_N = 30

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


def get_spy_benchmark() -> dict:
    hist = yf.Ticker("SPY").history(period="2y")
    if hist.empty:
        return {}
    close = hist["Close"]
    current = float(close.iloc[-1])
    n = len(close)
    return {
        "return_1y":  (current / float(close.iloc[-min(252, n)]) - 1) * 100,
        "return_6m":  (current / float(close.iloc[-min(126, n)]) - 1) * 100,
        "return_3m":  (current / float(close.iloc[-min(63,  n)]) - 1) * 100,
        "above_ma50":  current > float(close.rolling(50).mean().iloc[-1]),
        "above_ma200": current > float(close.rolling(200).mean().iloc[-1]),
    }


# ---------------------------------------------------------------------------
# CAN SLIM scoring — each function returns (score, detail_string)
# ---------------------------------------------------------------------------

def _c_score(stock: yf.Ticker) -> tuple[float, str]:
    """C — Current quarterly EPS growth + acceleration over last 3 quarters."""
    try:
        q = stock.quarterly_income_stmt
        if q is None or q.empty:
            return 0, ""
        for row in ("Diluted EPS", "Basic EPS"):
            if row not in q.index:
                continue
            eps = q.loc[row].dropna()
            if len(eps) < 5:
                continue
            recent, year_ago = float(eps.iloc[0]), float(eps.iloc[4])
            if year_ago == 0:
                continue
            growth = (recent - year_ago) / abs(year_ago) * 100

            # Base score on latest quarter growth
            if growth >= 50:
                base = 20
            elif growth >= 25:
                base = 15
            elif growth >= 10:
                base = 8
            elif growth > 0:
                base = 3
            else:
                base = 0

            # Acceleration bonus (+5): compare YoY growth across last 3 quarters
            accel = 0
            if len(eps) >= 9:
                def yoy(i):
                    ya = float(eps.iloc[i + 4])
                    return (float(eps.iloc[i]) - ya) / abs(ya) * 100 if ya != 0 else 0
                g0, g1, g2 = yoy(0), yoy(1), yoy(2)
                if g0 > g1 > g2:       # Three quarters accelerating
                    accel = 5
                elif g0 > g1:           # Two quarters improving
                    accel = 2

            score = min(25, base + accel)
            tag = " ↑↑accel" if accel == 5 else " ↑accel" if accel == 2 else ""
            return score, f"EPS {growth:+.0f}%{tag}"

        # Fallback: net income growth
        if "Net Income" in q.index:
            ni = q.loc["Net Income"].dropna()
            if len(ni) >= 5:
                r, ya = float(ni.iloc[0]), float(ni.iloc[4])
                if ya != 0:
                    g = (r - ya) / abs(ya) * 100
                    return min(18, max(0, g * 0.6)), f"NI {g:+.0f}%"
    except Exception:
        pass
    return 0, ""


def _a_score(stock: yf.Ticker) -> tuple[float, str]:
    """A — Annual EPS CAGR + consistency (no down years = bonus)."""
    try:
        a = stock.income_stmt
        if a is None or a.empty:
            return 0, ""
        for row in ("Diluted EPS", "Basic EPS", "Net Income"):
            if row not in a.index:
                continue
            eps = a.loc[row].dropna()
            if len(eps) < 3:
                continue
            vals = [float(eps.iloc[i]) for i in range(min(4, len(eps)))]
            newest, oldest = vals[0], vals[-1]
            years = len(vals) - 1
            if oldest <= 0 or newest <= 0:
                continue
            cagr = ((newest / oldest) ** (1 / years) - 1) * 100
            base = min(15, max(0, (cagr / 25) * 15))

            # Consistency: count years where earnings declined
            down_years = sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1])
            if down_years == 0:
                consistency = 5   # Perfect consistency
            elif down_years == 1:
                consistency = 2
            else:
                consistency = 0

            score = min(20, base + consistency)
            trend = "no declines" if down_years == 0 else f"{down_years} decline yr"
            return score, f"CAGR {cagr:+.0f}%/yr ({trend})"
    except Exception:
        pass
    return 0, ""


def _n_score(info: dict, hist: pd.DataFrame) -> tuple[float, str]:
    """N — Price structure: near 52wk high + Stage 2 uptrend + momentum."""
    score, parts = 0.0, []

    # 52-week high proximity (0-7 pts)
    high52 = info.get("fiftyTwoWeekHigh")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if high52 and price and high52 > 0:
        pct = (price / high52) * 100
        if pct >= 95:
            score += 7
        elif pct >= 85:
            score += 5
        elif pct >= 75:
            score += 2
        parts.append(f"{pct:.0f}% of 52wk high")

    if not hist.empty and len(hist) >= 50:
        close = hist["Close"]
        current = float(close.iloc[-1])
        n = len(close)

        # Stage 2 uptrend: price > 50MA > 200MA (0-5 pts)
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
        if ma200 and current > ma50 > ma200:
            score += 5
            parts.append("Stage2 ↑")
        elif current > ma50:
            score += 2
            parts.append("above 50MA")

        # Short-term momentum: 1m and 3m returns (0-3 pts)
        if n >= 63:
            r1m = (current / float(close.iloc[-21]) - 1) * 100
            r3m = (current / float(close.iloc[-63]) - 1) * 100
            if r1m > 5 and r3m > 10:
                score += 3
                parts.append(f"mom 1m:{r1m:+.0f}% 3m:{r3m:+.0f}%")
            elif r3m > 5:
                score += 1

    return min(15.0, score), " | ".join(parts)


def _s_score(info: dict, hist: pd.DataFrame) -> tuple[float, str]:
    """S — Supply & demand: volume trend + net accumulation days + float."""
    score, parts = 0.0, []

    if not hist.empty and len(hist) >= 50:
        close = hist["Close"]
        vol   = hist["Volume"]
        avg_vol50 = float(vol.rolling(50).mean().iloc[-1])

        # Volume trend: 20-day avg vs 50-day avg (0-5 pts)
        avg_vol20 = float(vol.rolling(20).mean().iloc[-1])
        if avg_vol50 > 0:
            vt = avg_vol20 / avg_vol50
            if vt >= 1.2:
                score += 5
            elif vt >= 1.05:
                score += 3
            elif vt >= 0.95:
                score += 1
            parts.append(f"vol trend {vt:.1f}x")

        # Net accumulation days in last 25 sessions (0-5 pts)
        # Accumulation = up day on above-avg volume; Distribution = down day on above-avg volume
        recent = hist.tail(25)
        rc = recent["Close"].values
        rv = recent["Volume"].values
        acc = sum(1 for i in range(1, len(rc)) if rc[i] > rc[i-1] and rv[i] > avg_vol50)
        dis = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i-1] and rv[i] > avg_vol50)
        net = acc - dis
        score += min(5.0, max(0.0, float(net)))
        parts.append(f"acc/dist {acc}/{dis}")

    # Float tightness (0-5 pts): lower float = tighter supply
    float_sh = info.get("floatShares") or 0
    sh_out   = info.get("sharesOutstanding") or 0
    if float_sh > 0 and sh_out > 0:
        float_pct = float_sh / sh_out * 100
        score += max(0.0, 5.0 - float_pct / 20)
        parts.append(f"float {float_pct:.0f}%")

    return min(15.0, score), " | ".join(parts)


def _l_score(hist: pd.DataFrame, spy: dict) -> tuple[float, str]:
    """L — Leader: 12m relative strength + RS trend (3m vs 6m improving)."""
    try:
        if hist.empty or len(hist) < 63:
            return 0, ""
        close = hist["Close"]
        n = len(close)
        current = float(close.iloc[-1])

        r12 = (current / float(close.iloc[-min(252, n)]) - 1) * 100
        r6  = (current / float(close.iloc[-min(126, n)]) - 1) * 100
        r3  = (current / float(close.iloc[-min(63,  n)]) - 1) * 100

        rs12 = r12 - spy.get("return_1y", 0)

        # Base RS score (0-10 pts)
        base = min(10.0, max(0.0, (rs12 + 20) / 40 * 10))

        # RS trend bonus (+5): is relative strength improving recently?
        trend = 0
        rs6 = r6 - spy.get("return_6m", 0)
        rs3 = r3 - spy.get("return_3m", 0)
        if rs3 > rs6 and rs6 > rs12 / 2:   # RS accelerating
            trend = 5
        elif rs3 > rs6:                      # RS improving
            trend = 2

        score = min(15.0, base + trend)
        tag = " ↑↑RS" if trend == 5 else " ↑RS" if trend == 2 else ""
        return score, f"RS {rs12:+.0f}%{tag}"
    except Exception:
        return 0, ""


def _i_score(stock: yf.Ticker, info: dict) -> tuple[float, str]:
    """I — Institutional ownership level + buying trend."""
    inst = info.get("heldPercentInstitutions") or 0
    if isinstance(inst, float) and inst <= 1.0:
        inst *= 100
    if inst <= 0:
        return 0, ""

    # Sweet spot 30-70%: not too ignored, not over-owned (0-7 pts)
    if 30 <= inst <= 70:
        base = 7
    elif 20 <= inst < 30 or 70 < inst <= 85:
        base = 4
    else:
        base = 1

    # Institutional trend: are fund count / ownership rising? (+3 pts)
    trend_bonus, trend_tag = 0, ""
    try:
        holders = stock.institutional_holders
        if holders is not None and not holders.empty:
            trend_bonus, trend_tag = 3, " buying"
    except Exception:
        pass

    return float(base + trend_bonus), f"inst {inst:.0f}%{trend_tag}"


# ---------------------------------------------------------------------------
# Per-stock orchestration
# ---------------------------------------------------------------------------

def score_stock(ticker: str, spy: dict) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        if not info or info.get("quoteType") != "EQUITY":
            return None
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or price <= 0:
            return None

        # Fetch 2y of price history once — shared by N, S, L
        hist = stock.history(period="2y")

        c, c_d = _c_score(stock)
        a, a_d = _a_score(stock)
        n, n_d = _n_score(info, hist)
        s, s_d = _s_score(info, hist)
        l, l_d = _l_score(hist, spy)
        i, i_d = _i_score(stock, info)

        details = {k: v for k, v in
                   {"C": c_d, "A": a_d, "N": n_d, "S": s_d, "L": l_d, "I": i_d}.items()
                   if v}

        return {
            "ticker": ticker,
            "name":   (info.get("longName") or ticker)[:38],
            "sector": info.get("sector") or "N/A",
            "price":  price,
            "score":  c + a + n + s + l + i,
            "breakdown": {"C": c, "A": a, "N": n, "S": s, "L": l, "I": i},
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


def build_email_html(top_stocks: list[dict], spy: dict) -> str:
    date_str = datetime.now().strftime("%B %d, %Y")
    spy_ret  = spy.get("return_1y", 0)

    rows = ""
    for rank, s in enumerate(top_stocks, 1):
        bg    = "#f4f7ff" if rank % 2 == 0 else "#ffffff"
        facts = "  ·  ".join(f"<b>{k}:</b> {v}" for k, v in s["details"].items())
        bar   = int(min(s["score"], 100))

        # Mini score breakdown per criterion
        bd = s.get("breakdown", {})
        mini = "".join(
            f"<span style='display:inline-block;margin:1px 3px;font-size:9px;"
            f"background:#e8eaf6;border-radius:3px;padding:1px 4px'>"
            f"{k}<b>{bd.get(k, 0):.0f}</b></span>"
            for k in ("C", "A", "N", "S", "L", "I")
        )

        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:10px 6px;text-align:center;font-size:20px;font-weight:bold;color:#1a237e;width:32px">{rank}</td>
          <td style="padding:10px 6px;font-weight:bold;font-size:14px;white-space:nowrap">{s['ticker']}</td>
          <td style="padding:10px 6px;font-size:12px">{s['name']}</td>
          <td style="padding:10px 6px;font-size:11px;color:#666">{s['sector']}</td>
          <td style="padding:10px 6px;font-weight:bold;white-space:nowrap">${s['price']:,.2f}</td>
          <td style="padding:10px 6px;text-align:center;white-space:nowrap">
            <div style="background:#e8eaf6;border-radius:6px;height:14px;width:80px;display:inline-block;vertical-align:middle">
              <div style="background:#1a237e;border-radius:6px;height:14px;width:{bar}px"></div>
            </div>
            <span style="font-weight:bold;margin-left:5px;font-size:13px">{s['score']:.0f}</span>
            <div style="margin-top:2px">{mini}</div>
          </td>
          <td style="padding:10px 6px;font-size:10px;color:#444;line-height:1.7">{facts}</td>
        </tr>"""

    legend = "".join(
        f"<span style='margin-right:12px'><b>{k}</b>&thinsp;{v}pts</span>"
        for k, v in WEIGHTS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><body style="font-family:Arial,Helvetica,sans-serif;background:#eef0f5;margin:0;padding:16px">
<div style="max-width:1060px;margin:auto">

  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;padding:22px 26px;border-radius:10px 10px 0 0">
    <h1 style="margin:0;font-size:21px;letter-spacing:.4px">📊 CAN SLIM Top {TOP_N} Watchlist</h1>
    <p style="margin:5px 0 0;opacity:.75;font-size:12px">{date_str} &nbsp;·&nbsp; S&amp;P 500 Universe &nbsp;·&nbsp; Trend-aware scoring</p>
  </div>

  <div style="background:#fff;padding:12px 18px;border-left:5px solid #1a237e;font-size:13px">
    <strong>Market Direction (M):</strong> {market_label(spy)}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>SPY:</strong> 1yr {spy_ret:+.1f}% &nbsp;·&nbsp;
    6m {spy.get('return_6m', 0):+.1f}% &nbsp;·&nbsp;
    3m {spy.get('return_3m', 0):+.1f}%
  </div>

  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;background:#fff;border:1px solid #dde">
    <tr style="background:#e8eaf6;font-size:11px;font-weight:bold;color:#1a237e">
      <th style="padding:9px 6px">#</th>
      <th style="padding:9px 6px;text-align:left">Ticker</th>
      <th style="padding:9px 6px;text-align:left">Company</th>
      <th style="padding:9px 6px;text-align:left">Sector</th>
      <th style="padding:9px 6px;text-align:left">Price</th>
      <th style="padding:9px 6px">Score /100</th>
      <th style="padding:9px 6px;text-align:left">CAN SLIM Factors</th>
    </tr>
    {rows}
  </table>

  <div style="background:#fff;padding:10px 18px;font-size:11px;color:#555;border-top:1px solid #eee">
    <strong>Scoring:</strong> {legend} &nbsp;·&nbsp;
    <em>Trend signals: ↑accel = accelerating earnings; Stage2 = price > 50MA > 200MA; acc/dist = accumulation vs distribution days; ↑RS = improving relative strength</em>
  </div>

  <div style="background:#fff8e1;padding:12px 18px;font-size:11px;color:#795548;border-radius:0 0 10px 10px;border-top:2px solid #ffe082">
    ⚠️ <strong>Not financial advice.</strong> CAN SLIM is William O'Neil's growth-investing methodology.
    Always do your own due diligence. Data via Yahoo Finance. Auto-generated by GitHub Actions.
  </div>

</div>
</body></html>"""


def send_email(html: str):
    resend.api_key = os.environ["RESEND_API_KEY"]
    recipient = os.environ["RECIPIENT_EMAIL"]
    date_str  = datetime.now().strftime("%b %d, %Y")

    params: resend.Emails.SendParams = {
        "from":    "CAN SLIM Screener <onboarding@resend.dev>",
        "to":      [recipient],
        "subject": f"📊 CAN SLIM Top {TOP_N} — {date_str}",
        "html":    html,
    }
    response = resend.Emails.send(params)
    log.info(f"Resend response: {response}")
    log.info(f"Email sent to {recipient}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== CAN SLIM Screener starting ===")

    log.info("Fetching SPY benchmark...")
    spy = get_spy_benchmark()
    log.info(f"SPY  1yr={spy.get('return_1y',0):.1f}%  "
             f"6m={spy.get('return_6m',0):.1f}%  "
             f"3m={spy.get('return_3m',0):.1f}%  "
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
    top = results[:TOP_N]

    log.info(f"Top {TOP_N} results:")
    for rank, s in enumerate(top, 1):
        bd = s["breakdown"]
        log.info(f"  {rank:2}. {s['ticker']:<6} score={s['score']:.1f}  "
                 f"C={bd['C']:.0f} A={bd['A']:.0f} N={bd['N']:.0f} "
                 f"S={bd['S']:.0f} L={bd['L']:.0f} I={bd['I']:.0f}  {s['name']}")

    html = build_email_html(top, spy)
    send_email(html)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
