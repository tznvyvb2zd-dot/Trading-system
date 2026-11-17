"""
Earnings-surprise data and features (no look-ahead).

For each company we use Yahoo's earnings calendar: announcement date/time, analysts' EPS
estimate, reported EPS and surprise %. A result is only "known" from the close of the first
trading day on which the market could react:
  * announced before the open (e.g. 7am)  -> that same trading day
  * after the close or time unknown       -> the next trading day (conservative)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

EARN_FEATURES = ["eps_surprise", "earn_reaction", "beat_streak", "days_since_earn",
                 "rank_eps_surprise", "rank_earn_reaction", "has_earn_data"]
RULE_WINDOW = 63   # earnings drift rule only uses results from the last ~3 months


def fetch_earnings(tickers, cache_path, log=print, max_age_days=25, pause=0.25):
    """Download up to 100 past quarters per ticker; cache to CSV, refresh entries older than max_age_days."""
    cols = ["ticker", "date", "eps_est", "eps_actual", "surprise_pct", "fetched"]
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    cache = pd.read_csv(cache_path, parse_dates=["date", "fetched"]) if cache_path.exists() else pd.DataFrame(columns=cols)
    last = cache.groupby("ticker")["fetched"].max() if len(cache) else pd.Series(dtype="datetime64[ns]")
    stale = [t for t in tickers if t not in last.index or (now - last[t]).days > max_age_days]
    log(f"earnings: {len(tickers) - len(stale)} cached, fetching {len(stale)}")
    rows, failures = [], 0
    for i, t in enumerate(stale):
        df = None
        for attempt in range(3):
            try:
                df = yf.Ticker(t).get_earnings_dates(limit=100)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        if df is None or len(df) == 0:
            failures += 1
            rows.append({"ticker": t, "date": pd.NaT, "eps_est": np.nan, "eps_actual": np.nan,
                         "surprise_pct": np.nan, "fetched": now})
        else:
            df = df.reset_index()
            dcol = df.columns[0]
            d = pd.to_datetime(df[dcol])
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_convert("America/New_York").dt.tz_localize(None)
            for dt, est, act, sur in zip(d, df.get("EPS Estimate"), df.get("Reported EPS"), df.get("Surprise(%)")):
                rows.append({"ticker": t, "date": dt, "eps_est": est, "eps_actual": act, "surprise_pct": sur, "fetched": now})
        if pause:
            time.sleep(pause)
        if (i + 1) % 100 == 0:
            log(f"earnings: {i + 1}/{len(stale)} fetched ({failures} without data)")
    if rows:
        cache = pd.concat([cache[~cache["ticker"].isin(stale)], pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    log(f"earnings: done ({failures} tickers without data)")
    return cache


def earnings_features(earn, dates, close, spy, mask):
    """Wide (dates x tickers) frames of earnings features, each known only from its reaction day onward."""
    cols = close.columns
    n = len(dates)
    sur = np.full((n, len(cols)), np.nan); rea = sur.copy(); stk = sur.copy(); evp = sur.copy()
    col_i = {t: j for j, t in enumerate(cols)}
    spy_v = spy.values
    ev = earn.dropna(subset=["date", "eps_actual"]).copy()
    ev = ev[ev["ticker"].isin(col_i)]
    ev["date"] = pd.to_datetime(ev["date"])
    for t, g in ev.sort_values("date").groupby("ticker"):
        j, px, streak = col_i[t], close[t].values, 0
        for _, e in g.iterrows():
            ann = e["date"]
            before_open = 0 < ann.hour < 10                       # e.g. "at 7 AM"; 0 = time unknown
            r = dates.searchsorted(ann.normalize(), side="left" if before_open else "right")
            if r < 1 or r >= n or np.isnan(px[r]) or np.isnan(px[r - 1]):
                continue
            s = e["surprise_pct"]
            if pd.isna(s) and pd.notna(e["eps_est"]) and e["eps_est"] != 0:
                s = (e["eps_actual"] - e["eps_est"]) / abs(e["eps_est"]) * 100
            if pd.isna(s):
                continue
            streak = (max(streak, 0) + 1) if s > 0 else ((min(streak, 0) - 1) if s < 0 else 0)
            sur[r, j] = np.clip(s, -50, 50) / 100
            rea[r, j] = (px[r] / px[r - 1]) - (spy_v[r] / spy_v[r - 1])      # move vs market on reaction day
            stk[r, j] = np.clip(streak, -8, 8)
            evp[r, j] = r
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols).ffill()
    sur, rea, stk, evp = mk(sur), mk(rea), mk(stk), mk(evp)
    days = pd.DataFrame(np.arange(n)[:, None] - evp.values, index=dates, columns=cols)
    has = sur.notna().astype(float)
    f = {
        "eps_surprise": sur.fillna(0.0),
        "earn_reaction": rea.fillna(0.0),
        "beat_streak": stk.fillna(0.0),
        "days_since_earn": days.clip(0, 130).fillna(130.0),
        "rank_eps_surprise": sur.where(mask).rank(axis=1, pct=True).fillna(0.5),
        "rank_earn_reaction": rea.where(mask).rank(axis=1, pct=True).fillna(0.5),
        "has_earn_data": has,
    }
    # Earnings drift rule score: reaction to the latest result, only if it was in the last RULE_WINDOW days
    rule = rea.where(days <= RULE_WINDOW)
    coverage = float(has.where(mask).stack().mean()) if mask.values.any() else 0.0
    return f, rule, coverage
