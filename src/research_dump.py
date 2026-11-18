"""Research helper: rebuild the UK walk-forward predictions and save them (with prices) for offline experiments."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trading_system as ts  # noqa: E402
from earnings import fetch_earnings, earnings_features  # noqa: E402

OUT = ts.ROOT / "research" / "cache"
OUT.mkdir(parents=True, exist_ok=True)

dates, bench = ts.market_data()
mask_all, all_tickers, sector_of = ts.load_membership(dates)
raw_c, raw_v = ts.download_all(all_tickers)
raw_c = ts.clean_prices(raw_c.reindex(dates))
raw_v = raw_v.reindex(dates)
have = [t for t in all_tickers if t in raw_c.columns and raw_c[t].notna().sum() > 300]
close, volume = raw_c[have].ffill(), raw_v[have].fillna(0)
fresh = raw_c[have].notna().astype(int).rolling(5, min_periods=1).max().astype(bool)
mask = mask_all.reindex(dates).reindex(columns=have).fillna(False).astype(bool) & fresh
mkt = bench[ts.MARKET]
earn = fetch_earnings(have, ts.DATA / "uk_earnings.csv.gz", log=ts.log)
earn_f, _, _ = earnings_features(earn, dates, close, mkt, mask)
df = ts.build_dataset(close, volume, mkt, mask, sector_of, dates, earn_f)
preds, targets = ts.walk_forward(df, dates, set(dates[::ts.TRAIN_STEP]))

preds.astype("float32").to_pickle(OUT / "preds.pkl.gz")
close.astype("float32").to_pickle(OUT / "close.pkl.gz")
mask.to_pickle(OUT / "mask.pkl.gz")
bench.astype("float32").to_pickle(OUT / "bench.pkl.gz")
pd.Series(sector_of).to_csv(OUT / "sectors.csv")
ts.log("saved research cache")
