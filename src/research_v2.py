"""Research v2: dividend features + 3-month target. Rebuilds walk-forward predictions for 4 configurations and saves
compact predictions (float16) for offline backtests. Runs on GitHub (needs Yahoo)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trading_system as ts  # noqa: E402
from earnings import fetch_earnings, earnings_features  # noqa: E402

OUT = ts.ROOT / "research" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
KEEP = ["LR + earnings", "LightGBM + earnings", "Ensemble + earnings", "Logistic regression", "LightGBM"]

dates, bench = ts.market_data()
mask_all, all_tickers, sector_of = ts.load_membership(dates)
raw_c, raw_v = ts.download_all(all_tickers)
raw_c = ts.clean_prices(raw_c.reindex(dates)); raw_v = raw_v.reindex(dates)
have = [t for t in all_tickers if t in raw_c.columns and raw_c[t].notna().sum() > 300]
close, volume = raw_c[have].ffill(), raw_v[have].fillna(0)
fresh = raw_c[have].notna().astype(int).rolling(5, min_periods=1).max().astype(bool)
mask = mask_all.reindex(dates).reindex(columns=have).fillna(False).astype(bool) & fresh
mkt = bench[ts.MARKET]
earn = fetch_earnings(have, ts.DATA / "uk_earnings.csv.gz", log=ts.log)
earn_f, _, _ = earnings_features(earn, dates, close, mkt, mask)

# ---------------- dividends (full history, point-in-time: a dividend counts from its ex-date)
divs = {}
for i, t in enumerate(have):
    for attempt in range(3):
        try:
            d = yf.Ticker(t).dividends
            if len(d):
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                divs[t] = d.groupby(level=0).sum()
            break
        except Exception:
            time.sleep(2)
    if i % 50 == 0:
        ts.log(f"dividends {i}/{len(have)}")
div_daily = pd.DataFrame({t: s.reindex(dates.union(s.index)).fillna(0).cumsum().reindex(dates).ffill().fillna(0)
                          for t, s in divs.items()}).reindex(columns=have).fillna(0)   # cumulative dividends
ttm = div_daily - div_daily.shift(252)
prev = div_daily.shift(252) - div_daily.shift(504)
yld = ttm / close
# Yahoo mixes pence and pounds: rescale tickers whose typical yield is ~100x off
med = yld.where(yld > 0).median()
yld = yld.div(np.where(med > 0.5, 100.0, np.where(med < 0.0005, 0.01, 1.0)), axis=1).clip(0, 0.25)
growth = (ttm / prev.replace(0, np.nan) - 1).clip(-1, 3)
paying_years = (ttm > 0).astype(float).rolling(252 * 5, min_periods=252).mean()
extra = {"div_yield": yld.fillna(0), "div_growth": growth.fillna(0), "div_cut": (growth < -0.2).astype(float),
         "div_consistency": paying_years.fillna(0)}
extra["rank_div_yield"] = extra["div_yield"].where(mask).rank(axis=1, pct=True)
extra["vol_252d"] = close.pct_change(fill_method=None).rolling(252).std()
extra["rank_vol_252d"] = extra["vol_252d"].where(mask).rank(axis=1, pct=True)
DIV = list(extra)
pd.Series({t: len(s) for t, s in divs.items()}).to_csv(OUT / "div_counts.csv")

base_features = list(ts.FEATURES)
for name, horizon, feats in [("h21_base", 21, base_features), ("h21_div", 21, base_features + DIV),
                             ("h63_base", 63, base_features), ("h63_div", 63, base_features + DIV)]:
    ts.H = horizon
    ts.FEATURES = feats
    df = ts.build_dataset(close, volume, mkt, mask, sector_of, dates, earn_f, extra=extra)
    preds, _ = ts.walk_forward(df, dates, set(dates[::ts.TRAIN_STEP]))
    preds[[c for c in KEEP if c in preds]].astype("float16").to_pickle(OUT / f"preds_{name}.pkl.gz")
    ts.log(f"saved {name}")

close.astype("float32").to_pickle(OUT / "close.pkl.gz")
mask.to_pickle(OUT / "mask.pkl.gz")
bench.astype("float32").to_pickle(OUT / "bench.pkl.gz")
pd.Series(sector_of).to_csv(OUT / "sectors.csv")
ts.log("research v2 done")
