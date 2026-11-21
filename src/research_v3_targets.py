"""Model lab round 2: better targets. Same ranked features + LightGBM; only the label changes.
- target        : beat the median stock over the next 21 days (current)
- t_sector      : beat the median after removing the stock's sector average return
- t_beta        : beat the median after removing beta x market (equal-weight members) return
- t_riskadj     : beat the median of return / recent volatility"""
import numpy as np

import research_v3_common as rc
import trading_system as ts

_orig = rc.load_all


def load_with_targets(save_shared=False):
    df, dates, sets = _orig(save_shared)
    d = df.index.get_level_values("date")
    tk = df.index.get_level_values("ticker")
    sector = pd.Series(tk.map(lambda t: SECTORS.get(t, "Unknown")), index=df.index)
    f = df["fwd_ret"]
    mkt = f.groupby(d).transform("mean")
    resid = {
        "t_sector": f - f.groupby([d, sector.values]).transform("mean"),
        "t_beta": f - df["beta_60d"].clip(-1, 3) * mkt,
        "t_riskadj": f / (df["vol_20d"].clip(lower=0.005) * np.sqrt(ts.H)),
    }
    for k, r in resid.items():
        med = r.groupby(d).transform("median")
        df[k] = (r > med).astype("float32").where(r.notna())
    ts.log("targets: " + ", ".join(f"{k} {df[k].notna().mean():.0%}" for k in resid))
    return df, dates, sets


import pandas as pd  # noqa: E402
SECTORS = pd.read_csv(ts.ROOT / "data" / "uk" / "ftse350_sectors.csv").set_index("ticker")["sector"].to_dict() \
    if "sector" in pd.read_csv(ts.ROOT / "data" / "uk" / "ftse350_sectors.csv", nrows=1).columns else {}
rc.load_all = load_with_targets
rc.run({"ranked_lgbm": ("lgbm", "ranked"), "t_sector": ("lgbm_t:t_sector", "ranked"),
        "t_beta": ("lgbm_t:t_beta", "ranked"), "t_riskadj": ("lgbm_t:t_riskadj", "ranked")}, "targets", save_shared=False)
