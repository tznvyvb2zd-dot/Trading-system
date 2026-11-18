"""Model lab v3 (shared): builds one big dataset with base + new features (raw and cross-sectionally ranked) and
runs a walk-forward over a list of model 'specs'. Predictions cover all years; the 2024+ vault is only scored at
the very end (offline)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trading_system as ts  # noqa: E402
from earnings import EARN_FEATURES, fetch_earnings, earnings_features  # noqa: E402

OUT = ts.ROOT / "research" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
MARKET_LEVEL = {"mkt_ret_20d", "mkt_vol_20d", "days_into_quarter"}


def new_features(close, volume, mkt, mask, sector_of):
    r1, m1 = close.pct_change(fill_method=None), mkt.pct_change()
    cols = close.columns
    sectors = np.array([sector_of.get(t, "Unknown") for t in cols])
    var_m60, var_m252 = m1.rolling(60).var(), m1.rolling(252).var()
    beta60 = r1.rolling(60).cov(m1).div(var_m60, axis=0)
    beta252 = r1.rolling(252).cov(m1).div(var_m252, axis=0)
    vol252 = r1.rolling(252).std()
    mom = close.shift(21) / close.shift(252) - 1
    mkt_mom = mkt.shift(21) / mkt.shift(252) - 1
    seas = [close.shift(252 * k - 21) / close.shift(252 * k) - 1 for k in range(1, 6)]
    seas = pd.concat(seas, keys=range(5)).groupby(level=1).mean().reindex(close.index)   # same month, past years
    seas_n = sum((close.shift(252 * k).notna()).astype(int) for k in range(1, 6))
    ret20 = close.pct_change(20, fill_method=None)
    f = {
        "idio_vol_60d": (r1.rolling(60).var() - beta60.pow(2).mul(var_m60, axis=0)).clip(lower=0) ** 0.5,
        "vol_252d": vol252,
        "downside_vol_60d": r1.clip(upper=0).pow(2).rolling(60).mean() ** 0.5,
        "skew_60d": r1.rolling(60).skew(),
        "amihud_60d": np.log1p((r1.abs() / (close * volume).replace(0, np.nan)).rolling(60, min_periods=20).mean() * 1e6),
        "turnover_trend": volume.rolling(20).mean() / volume.rolling(252).mean().replace(0, np.nan),
        "resid_mom_12_1": mom - beta252.mul(mkt_mom, axis=0),
        "vol_adj_mom": mom / (vol252 * np.sqrt(252)),
        "seasonality": seas.where(seas_n >= 2),
        "dist_52w_low": close / close.rolling(252).min() - 1,
        "mkt_corr_60d": r1.rolling(60).corr(m1),
        "vs_sector_20d": ret20 - ret20.where(mask).T.groupby(sectors).transform("mean").T,
        "ret_120d": close.pct_change(120, fill_method=None),
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def load_all(save_shared=True):
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
    nf = new_features(close, volume, mkt, mask, sector_of)
    NEW = list(nf)
    base = list(ts.FEATURES)
    df = ts.build_dataset(close, volume, mkt, mask, sector_of, dates, earn_f)     # same rows as the live model
    km = mask.loc[dates]
    for k, w in nf.items():                                  # new features may be missing (NaN): never drop rows
        df[k] = w.loc[dates].where(km).unstack().swaplevel().reindex(df.index).astype("float32")
    ts.FEATURES = base + NEW
    # cross-sectional ranks of every stock-level feature (removes market-wide swings)
    stock_level = [c for c in ts.FEATURES if c not in MARKET_LEVEL and not c.startswith("rank_")]
    d = df.index.get_level_values("date")
    for c in stock_level:
        df[f"xr_{c}"] = df.groupby(d)[c].rank(pct=True).astype("float32")
    df["fwd_rank"] = df.groupby(d)["fwd_ret"].rank(pct=True).astype("float32")
    sets = {
        "base": base + EARN_FEATURES,
        "new": base + NEW + EARN_FEATURES,
        "ranked": [f"xr_{c}" for c in stock_level] + sorted(MARKET_LEVEL) + EARN_FEATURES,
    }
    if save_shared:
        close.astype("float32").to_pickle(OUT / "close.pkl.gz")
        mask.to_pickle(OUT / "mask.pkl.gz")
        bench.astype("float32").to_pickle(OUT / "bench.pkl.gz")
        pd.Series(sector_of).to_csv(OUT / "sectors.csv")
    return df, dates, sets


def _lgbm_cls(**kw):
    import lightgbm as lgb
    p = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def _lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def fit_predict(spec, train, test, feats):
    """spec: (kind). Returns predictions (higher = better) for `test`."""
    import lightgbm as lgb
    X, Xt = train[feats], test[feats]
    if spec == "lgbm":
        m = _lgbm_cls(); m.fit(X, train["target"]); return m.predict_proba(Xt)[:, 1]
    if spec == "lr":
        m = _lr(); m.fit(X, train["target"]); return m.predict_proba(Xt)[:, 1]
    if spec == "lgbm_reg":
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
        m.fit(X, train["fwd_rank"]); return m.predict(Xt)
    if spec == "lgbm_rank":
        tr = train.sort_index(level="date")
        groups = tr.groupby(level="date").size().values
        rel = np.minimum((tr["fwd_rank"] * 5).astype(int), 4)
        m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
        m.fit(tr[feats], rel, group=groups); return m.predict(Xt)
    if spec == "lgbm_bag":
        ps = []
        for seed, kw in enumerate([dict(num_leaves=7, min_child_samples=500), dict(num_leaves=15),
                                   dict(num_leaves=31, min_child_samples=1000, n_estimators=500, learning_rate=0.02)]):
            m = _lgbm_cls(random_state=seed, **kw); m.fit(X, train["target"])
            ps.append(pd.Series(m.predict_proba(Xt)[:, 1]).rank(pct=True).values)
        return np.mean(ps, axis=0)
    if spec == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=2048,
                                                         learning_rate_init=1e-3, max_iter=25, early_stopping=True,
                                                         n_iter_no_change=4, random_state=0))
        m.fit(X.fillna(0.5), train["fwd_rank"]); return m.predict(Xt.fillna(0.5))
    raise ValueError(spec)


def run(specs, tag, save_shared=True):
    """specs: {name: (kind, feature_set)}."""
    df, dates, sets = load_all(save_shared)
    train_set = set(dates[::ts.TRAIN_STEP])
    lab = df.dropna(subset=["target"])
    lab = lab[lab.index.get_level_values("date").isin(train_set)]
    lab_d, all_d = lab.index.get_level_values("date"), df.index.get_level_values("date")
    outs = []
    for start in range(ts.MIN_TRAIN, len(dates), ts.RETRAIN_EVERY):
        window = dates[start:start + ts.RETRAIN_EVERY]
        test = df[(all_d >= window[0]) & (all_d <= window[-1])]
        if test.empty:
            continue
        train = lab[lab_d <= dates[start - ts.H - 1]]
        out = pd.DataFrame(index=test.index)
        for name, (kind, fs) in specs.items():
            out[name] = np.asarray(fit_predict(kind, train, test, sets[fs]), dtype="float32")
        outs.append(out)
        ts.log(f"[{tag}] trained to {dates[start - ts.H - 1].date()}")
    preds = pd.concat(outs)
    preds.astype("float32").to_pickle(OUT / f"preds_{tag}.pkl.gz")
    ts.log(f"[{tag}] done: {list(specs)}")
