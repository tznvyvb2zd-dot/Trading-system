"""Model lab v3, job C: the best setup (ranked features + LightGBM) with and without GDELT news features."""
import numpy as np
import pandas as pd

import research_v3_common as rc
import trading_system as ts

_orig = rc.load_all


def load_with_news(save_shared=False):
    df, dates, sets = _orig(save_shared)
    news = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in sorted((ts.ROOT / "research" / "news").glob("news_*.csv.gz"))])
    cal = pd.date_range(dates[0], dates[-1])
    tone = news.pivot_table(index="date", columns="ticker", values="tone").reindex(cal)
    arts = news.pivot_table(index="date", columns="ticker", values="articles").reindex(cal).fillna(0)
    covered = arts.cumsum() > 0                                   # GDELT history starts 2017-01-01
    arts.loc[:pd.Timestamp("2016-12-31")] = np.nan
    wsum = lambda x, n: x.rolling(n, min_periods=1).sum()
    tw = (tone.fillna(0) * arts)
    f = {"news_tone_7d": wsum(tw, 7) / wsum(arts, 7).replace(0, np.nan),
         "news_tone_30d": wsum(tw, 30) / wsum(arts, 30).replace(0, np.nan),
         "news_tone_90d": wsum(tw, 90) / wsum(arts, 90).replace(0, np.nan),
         "news_attention": np.log1p(wsum(arts, 7) / 7) - np.log1p(wsum(arts, 90) / 90),
         "news_volume_90d": np.log1p(wsum(arts, 90))}
    f["news_tone_shift"] = f["news_tone_7d"] - f["news_tone_90d"]
    d = df.index.get_level_values("date")
    names = []
    for k, w in f.items():
        w = w.shift(1).reindex(dates)                             # yesterday's news only: no same-day look-ahead
        w = w.where(covered.reindex(dates).fillna(False))
        df[k] = w.unstack().swaplevel().reindex(df.index).astype("float32")
        df[f"xr_{k}"] = df.groupby(d)[k].rank(pct=True).astype("float32")
        names.append(f"xr_{k}")
    sets["ranked_news"] = sets["ranked"] + names
    sets["ranked_news_raw"] = sets["ranked"] + list(f)
    ts.log(f"news features: {len(news)} rows, {news.ticker.nunique()} companies")
    return df, dates, sets


rc.load_all = load_with_news
rc.run({"ranked_lgbm": ("lgbm", "ranked"), "news_lgbm": ("lgbm", "ranked_news"),
        "news_raw_lgbm": ("lgbm", "ranked_news_raw"), "news_lr": ("lr", "ranked_news")}, "news", save_shared=False)
