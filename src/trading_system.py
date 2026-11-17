"""
UK trading system: FTSE 350, paper trading, £100.

Every weekday (GitHub Actions):
  * downloads FTSE 350 prices (point-in-time membership, incl. companies that later left the index)
  * re-scores every member with the current model
  * DAILY CHECK: sells any holding that has fallen out of the top BUFFER_RANK and swaps in the best-ranked
    replacement (keeps trading - and 0.5% stamp duty - low)
  * every ~21 trading days (monthly): re-runs the honest walk-forward backtest, picks the best model,
    and fully re-picks the top 10 (max 2 per sector)
  * writes data/results.json (for Claude) and docs/index.html (live dashboard)

Learning / research tool. Not financial advice.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from earnings import EARN_FEATURES, earnings_features, fetch_earnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- settings
ROOT = Path(os.environ.get("TS_ROOT", Path(__file__).resolve().parents[1]))
DATA, DOCS = ROOT / "data", ROOT / "docs"
UKDIR = ROOT / "data" / "uk"                      # membership + sector files (committed)
MEMBERSHIP = UKDIR / "ftse350_membership.csv"
SECTORS = UKDIR / "ftse350_sectors.csv"

MARKET = "ISF.L"                                  # iShares Core FTSE 100: market proxy for features
BENCHMARKS = {"FTSE 100 tracker": "ISF.L", "FTSE 250 tracker": "MIDD.L"}
START = "2010-01-01"
H = 21                    # monthly re-pick + prediction horizon (trading days)
TRAIN_STEP = 5            # train on every 5th day (overlapping monthly targets add little)
RETRAIN_EVERY = 126       # walk-forward: retrain every ~6 months
MIN_TRAIN = 504           # ~2 years before the first prediction

START_CAPITAL = 100.0     # £
BUY_COST_BPS = 60         # 0.5% stamp duty (SDRT) + ~0.1% spread on every UK share purchase
SELL_COST_BPS = 10        # ~0.1% spread on sales (no stamp duty)
TOP_N = 10
MAX_PER_SECTOR = 2
BUFFER_RANK = 50          # daily check: sell a holding only if its smoothed rank drops below this
SMOOTH_DAYS = 10          # daily check uses a 10-day smoothed score (ignores one-day noise)
MAX_SWAPS_PER_DAY = 2     # limits trading (each swap costs ~0.7% incl. stamp duty)
SIGNAL_MODEL = os.environ.get("TS_SIGNAL_MODEL", "auto")

AS_OF = os.environ.get("AS_OF")
FORCE_REBALANCE = os.environ.get("FORCE_REBALANCE", "").lower() in ("1", "true", "yes")
USE_EARNINGS = os.environ.get("TS_EARNINGS", "1") != "0"

ML_MODELS = ["Logistic regression", "LightGBM", "Ensemble"] + (
    ["LR + earnings", "LightGBM + earnings", "Ensemble + earnings"] if USE_EARNINGS else [])
SCORES = ML_MODELS + ["Momentum rule"]
ENSEMBLES = {"Ensemble": ["Logistic regression", "LightGBM"],
             "Ensemble + earnings": ["LR + earnings", "LightGBM + earnings"]}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def read_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def write_json(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------- data
def load_membership(dates):
    mem = pd.read_csv(MEMBERSHIP, parse_dates=["date"]).sort_values("date")
    sets = [set(s.split(",")) for s in mem["tickers"].fillna("")]
    tickers = sorted(set().union(*sets) - {""})
    rows = pd.DataFrame([[t in s for t in tickers] for s in sets], index=mem["date"].values, columns=tickers)
    mask = rows.reindex(dates, method="ffill").fillna(False).astype(bool)
    sec = pd.read_csv(SECTORS)
    return mask, tickers, dict(zip(sec["ticker"], sec["sector"]))


def _download(tickers, start=START):
    r = yf.download(list(tickers), start=start, auto_adjust=True, progress=False, threads=True)
    c, v = r["Close"], r["Volume"]
    if isinstance(c, pd.Series):
        c, v = c.to_frame(tickers[0]), v.to_frame(tickers[0])
    return c, v


def download_all(tickers, chunk=100):
    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                c, v = _download(part); break
            except Exception as e:  # noqa: BLE001
                log(f"download retry {attempt + 1} ({e})")
        else:
            continue
        closes.append(c); vols.append(v)
        log(f"downloaded {min(i + chunk, len(tickers))}/{len(tickers)}")
    c, v = pd.concat(closes, axis=1), pd.concat(vols, axis=1)
    return c.loc[:, ~c.columns.duplicated()], v.loc[:, ~v.columns.duplicated()]


def clean_prices(close):
    """Fix Yahoo's pence/pound unit flips (x100 jumps) and one-day spikes on London prices."""
    c = close.copy()
    for t in c.columns:
        s = c[t].dropna()
        if len(s) < 3:
            continue
        v = s.values.astype(float)
        ratio = v[1:] / v[:-1]
        for i in np.where((ratio > 50) | (ratio < 0.02))[0]:          # persistent unit change: rescale history
            f = v[i + 1] / v[i]
            v[: i + 1] *= f
        r = v[1:] / v[:-1]
        spike = np.where(((r[:-1] > 3) & (r[1:] < 0.4)) | ((r[:-1] < 0.33) & (r[1:] > 2.5)))[0] + 1
        v[spike] = np.nan                                              # one-day spikes that reverse
        c.loc[s.index, t] = v
    return c


def market_data():
    b, _ = _download(list(dict.fromkeys([MARKET] + list(BENCHMARKS.values()))))
    dates = b[MARKET].dropna().index
    if AS_OF:
        dates = dates[dates <= pd.Timestamp(AS_OF)]
    return dates, b.reindex(dates).ffill()


# ---------------------------------------------------------------- features
OLD_FEATURES = ["ret_5d", "ret_20d", "ret_60d", "rel_strength_20d", "vol_20d", "ma_gap_20_50",
                "dist_200ma", "rsi_14", "volume_z", "mkt_ret_20d", "mkt_vol_20d"]
RANKED = ["ret_5d", "ret_20d", "ret_60d", "vol_20d", "rsi_14", "dist_52w_high"]
NEW_FEATURES = ["ret_1d", "mom_12_1", "dist_52w_high", "max_ret_20d", "beta_60d", "sector_mom_60d",
                "vs_sector_60d", "days_into_quarter"] + [f"rank_{k}" for k in RANKED]
FEATURES = OLD_FEATURES + NEW_FEATURES


def rsi(prices, n=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_dataset(close, volume, mkt, mask, sector_of, keep_dates, earn_f=None):
    cols = close.columns
    r1, mkt_r1 = close.pct_change(fill_method=None), mkt.pct_change()
    mkt_ret20, mkt_vol20 = mkt.pct_change(20), mkt_r1.rolling(20).std()
    sectors = np.array([sector_of.get(t, "Unknown") for t in cols])
    bc = lambda s: pd.DataFrame(np.repeat(s.values[:, None], len(cols), axis=1), index=close.index, columns=cols)
    f = {
        "ret_5d": close.pct_change(5, fill_method=None), "ret_20d": close.pct_change(20, fill_method=None),
        "ret_60d": close.pct_change(60, fill_method=None), "vol_20d": r1.rolling(20).std(),
        "ma_gap_20_50": close.rolling(20).mean() / close.rolling(50).mean() - 1,
        "dist_200ma": close / close.rolling(200).mean() - 1, "rsi_14": rsi(close),
        "volume_z": (volume - volume.rolling(20).mean()) / volume.rolling(20).std(),
        "ret_1d": r1, "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "dist_52w_high": close / close.rolling(252).max() - 1, "max_ret_20d": r1.rolling(20).max(),
        "beta_60d": r1.rolling(60).cov(mkt_r1).div(mkt_r1.rolling(60).var(), axis=0),
    }
    f["rel_strength_20d"] = f["ret_20d"].sub(mkt_ret20, axis=0)
    f["mkt_ret_20d"], f["mkt_vol_20d"] = bc(mkt_ret20), bc(mkt_vol20)
    f["sector_mom_60d"] = f["ret_60d"].where(mask).T.groupby(sectors).transform("mean").T
    f["vs_sector_60d"] = f["ret_60d"] - f["sector_mom_60d"]
    f["days_into_quarter"] = bc(pd.Series((close.index - close.index.to_period("Q").start_time).days, index=close.index))
    for k in RANKED:
        f[f"rank_{k}"] = f[k].where(mask).rank(axis=1, pct=True)

    fwd = close.shift(-H) / close - 1                     # FUTURE return: training label only
    fwd_m = fwd.where(mask)
    target = fwd_m.gt(fwd_m.median(axis=1), axis=0).astype(float).where(fwd_m.notna())

    km = mask.loc[keep_dates]
    to_long = lambda w: w.loc[keep_dates].where(km).unstack().swaplevel()
    df = pd.concat({k: to_long(f[k]).astype("float32") for k in FEATURES}, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if earn_f is not None:
        for k in EARN_FEATURES:
            df[k] = to_long(earn_f[k]).reindex(df.index).fillna(0).astype("float32")
    df["fwd_ret"] = to_long(fwd).reindex(df.index)
    df["target"] = to_long(target).reindex(df.index)
    df.index.names = ["date", "ticker"]
    return df.sort_index()


def feats_for(name):
    return FEATURES + EARN_FEATURES if "earnings" in name else FEATURES


def make_models():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    gbm = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                     random_state=42, verbose=-1)
    models = {"Logistic regression": lr(), "LightGBM": gbm()}
    if USE_EARNINGS:
        models.update({"LR + earnings": lr(), "LightGBM + earnings": gbm()})
    return models


def add_derived(p):
    for ens, parts in ENSEMBLES.items():
        if all(c in p for c in parts):
            p[ens] = p[parts].mean(axis=1)
    p["Momentum rule"] = p["ret_60d"]
    return p


def walk_forward(df, dates, train_set):
    """Train on the past only (gap = H days), predict EVERY day of the next ~6 months."""
    lab = df.dropna(subset=["target"])
    lab = lab[lab.index.get_level_values("date").isin(train_set)]
    lab_d, all_d = lab.index.get_level_values("date"), df.index.get_level_values("date")
    outs = []
    for start in range(MIN_TRAIN, len(dates), RETRAIN_EVERY):
        window = dates[start:start + RETRAIN_EVERY]
        test = df[(all_d >= window[0]) & (all_d <= window[-1])]
        if test.empty:
            continue
        train = lab[lab_d <= dates[start - H - 1]]
        out = test[["ret_60d"]].copy()
        for name, m in make_models().items():
            m.fit(train[feats_for(name)], train["target"])
            out[name] = m.predict_proba(test[feats_for(name)])[:, 1].astype("float32")
        outs.append(out)
        log(f"walk-forward: trained to {dates[start - H - 1].date()}")
    preds = add_derived(pd.concat(outs))
    return preds, df.loc[preds.index, ["target"]]


# ---------------------------------------------------------------- portfolio rules
def pick_with_caps(ranked, n, cap, sector_of, held=()):
    picks, count = [], {}
    for t in held:
        s = sector_of.get(t, "Unknown"); count[s] = count.get(s, 0) + 1
    for t in ranked:
        if t in held or t in picks:
            continue
        s = sector_of.get(t, "Unknown")
        if cap and s != "Unknown" and count.get(s, 0) >= cap:
            continue
        picks.append(t); count[s] = count.get(s, 0) + 1
        if len(picks) == n:
            break
    return picks


def daily_swaps(weights, scores_today, sector_of):
    """Daily check: holdings ranked worse than BUFFER_RANK (or no longer in the index) are swapped out."""
    ranks = scores_today.rank(ascending=False)
    out = sorted([t for t in weights.index if t not in ranks.index or ranks[t] > BUFFER_RANK],
                 key=lambda t: -ranks.get(t, 1e9))[:MAX_SWAPS_PER_DAY]                  # worst first
    swaps = []
    for t in out:
        keep = [h for h in weights.index if h != t and h not in [s[0] for s in swaps]]
        repl = pick_with_caps(scores_today.sort_values(ascending=False).index, 1, MAX_PER_SECTOR, sector_of,
                              held=keep + [s[1] for s in swaps])
        if repl:
            swaps.append((t, repl[0]))
    return swaps, ranks


def simulate(score_w, ret_next, sector_of, daily_check, members):
    """Day-by-day simulation with stamp duty. Returns daily net returns."""
    days = score_w.index
    w = pd.Series(dtype=float)
    out = {}
    for i, d in enumerate(days[:-1]):
        s = score_w.loc[d].dropna()
        s = s[s.index.isin(members.columns[members.loc[d].values])] if members is not None else s
        cost = 0.0
        if i % H == 0:
            if s.empty:
                continue
            picks = pick_with_caps(s.sort_values(ascending=False).index, TOP_N, MAX_PER_SECTOR, sector_of)
            new = pd.Series(1 / len(picks), index=picks)
            diff = new.sub(w, fill_value=0)
            cost = diff.clip(lower=0).sum() * BUY_COST_BPS / 1e4 + (-diff.clip(upper=0)).sum() * SELL_COST_BPS / 1e4
            w = new
        elif daily_check and len(w) and not s.empty:
            swaps, _ = daily_swaps(w, s, sector_of)
            for old, new_t in swaps:
                val = w.pop(old)
                w[new_t] = val
                cost += val * (BUY_COST_BPS + SELL_COST_BPS) / 1e4
        if w.empty:
            continue
        r = ret_next.loc[d, w.index].fillna(0.0)
        gross = float((w * r).sum())
        out[d] = gross - cost
        w = w * (1 + r) / (1 + gross)                               # weights drift with prices
    return pd.Series(out)


def simulate_daily(score_w, smooth_w, ret_next, sector_of, members):
    """Monthly re-pick on raw scores; in between, daily check on SMOOTHED scores."""
    days = score_w.index
    w = pd.Series(dtype=float)
    out = {}
    for i, d in enumerate(days[:-1]):
        live = members.columns[members.loc[d].values]
        cost = 0.0
        if i % H == 0:
            s = score_w.loc[d].dropna(); s = s[s.index.isin(live)]
            if s.empty:
                continue
            picks = pick_with_caps(s.sort_values(ascending=False).index, TOP_N, MAX_PER_SECTOR, sector_of)
            new = pd.Series(1 / len(picks), index=picks)
            diff = new.sub(w, fill_value=0)
            cost = diff.clip(lower=0).sum() * BUY_COST_BPS / 1e4 + (-diff.clip(upper=0)).sum() * SELL_COST_BPS / 1e4
            w = new
        elif len(w):
            sm = smooth_w.loc[d].dropna(); sm = sm[sm.index.isin(live)]
            if not sm.empty:
                for old, new_t in daily_swaps(w, sm, sector_of)[0]:
                    val = w.pop(old); w[new_t] = val
                    cost += val * (BUY_COST_BPS + SELL_COST_BPS) / 1e4
        if w.empty:
            continue
        r = ret_next.loc[d, w.index].fillna(0.0)
        gross = float((w * r).sum())
        out[d] = gross - cost
        w = w * (1 + r) / (1 + gross)
    return pd.Series(out)


def stats(r, ppy=252):
    eq = (1 + r).cumprod()
    return {"CAGR": float(eq.iloc[-1] ** (ppy / len(r)) - 1), "Volatility": float(r.std() * np.sqrt(ppy)),
            "Sharpe": float(r.mean() / r.std() * np.sqrt(ppy)), "Max drawdown": float((eq / eq.cummax() - 1).min()),
            "£100 grew to": float(eq.iloc[-1] * 100)}


def run_backtest(preds, targets, close, bench, mask, sector_of):
    from sklearn.metrics import roc_auc_score
    ret_next = close.pct_change(fill_method=None).shift(-1)
    days = preds.index.get_level_values("date").unique().sort_values()
    members = mask.loc[days]
    curves = {}
    for s in SCORES:
        wide = preds[s].unstack("ticker").reindex(days)
        curves[f"{s} · monthly"] = simulate(wide, ret_next, sector_of, False, members)
        smooth = wide.ewm(span=SMOOTH_DAYS, min_periods=1).mean().where(wide.notna())      # past days only
        curves[f"{s} · daily check"] = simulate_daily(wide, smooth, ret_next, sector_of, members)
        log(f"simulated {s}")
    curves["Equal-weight (all)"] = _equal_weight(members, ret_next)
    idx = curves["Equal-weight (all)"].index
    for name, tk in BENCHMARKS.items():
        curves[name] = bench[tk].pct_change().shift(-1).reindex(idx).fillna(0)
    table = {k: {m: round(v, 4) for m, v in stats(r).items()} for k, r in curves.items()}
    sc = preds.join(targets).dropna(subset=["target"])
    auc = {m: round(float(roc_auc_score(sc["target"], sc[m])), 4) for m in SCORES}
    yearly = {k: {str(y): round(float(v), 4) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}
              for k, r in curves.items()}
    growth = {k: {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in ((1 + r).cumprod() * 100).iloc[::5].items()}
              for k, r in curves.items()}
    daily_vs_monthly = [{"model": s, "sharpe_monthly": table[f"{s} · monthly"]["Sharpe"],
                         "sharpe_daily": table[f"{s} · daily check"]["Sharpe"],
                         "cagr_monthly": table[f"{s} · monthly"]["CAGR"],
                         "cagr_daily": table[f"{s} · daily check"]["CAGR"]} for s in SCORES]
    return {"table": table, "auc": auc, "yearly": yearly, "growth": growth, "daily_vs_monthly": daily_vs_monthly,
            "period": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")]}


def _equal_weight(members, ret_next):
    """Own every FTSE 350 member equally, rebalanced monthly (with stamp duty on the top-ups)."""
    out, w = {}, pd.Series(dtype=float)
    for i, d in enumerate(members.index[:-1]):
        cost = 0.0
        if i % H == 0:
            names = members.columns[members.loc[d].values]
            names = [t for t in names if pd.notna(ret_next.loc[d, t])]
            new = pd.Series(1 / len(names), index=names)
            diff = new.sub(w, fill_value=0)
            cost = diff.clip(lower=0).sum() * BUY_COST_BPS / 1e4 + (-diff.clip(upper=0)).sum() * SELL_COST_BPS / 1e4
            w = new
        r = ret_next.loc[d, w.index].fillna(0.0)
        g = float((w * r).sum()); out[d] = g - cost
        w = w * (1 + r) / (1 + g)
    return pd.Series(out)


# ---------------------------------------------------------------- live paper portfolio
def gbp_price(t, px):
    """Price per share in £ (London quotes are usually in pence)."""
    try:
        cur = yf.Ticker(t).fast_info.get("currency") or "GBp"
    except Exception:  # noqa: BLE001
        cur = "GBp"
    return (px / 100 if cur in ("GBp", "GBX") else px), cur


def trade(state, today, sells, buys, prices):
    """Paper trades: sell `sells` fully, then buy `buys` {ticker: £amount}. Costs come out of cash."""
    pos, log_rows = state["positions"], []
    for t in sells:
        p = pos.pop(t)
        proceeds = p["value_gbp"]
        cost = proceeds * SELL_COST_BPS / 1e4
        state["cash_gbp"] += proceeds - cost
        log_rows.append({"date": today, "ticker": t, "action": "SELL", "change_gbp": -round(proceeds, 2), "cost_gbp": round(cost, 3)})
    for t, amount in buys.items():
        cost = amount * BUY_COST_BPS / 1e4
        state["cash_gbp"] -= amount + cost
        px = float(prices[t])
        old = pos.get(t)
        if old:
            old["value_gbp"] += amount
        else:
            gbp, cur = gbp_price(t, px)
            pos[t] = {"value_gbp": amount, "last_px": px, "entry_px": px, "entry_date": today,
                      "currency": cur, "px_gbp": gbp}
        log_rows.append({"date": today, "ticker": t, "action": "BUY" if not old else "TOP UP",
                         "change_gbp": round(amount, 2), "cost_gbp": round(cost, 3)})
    return log_rows


def main():
    DATA.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    state_path = DATA / "uk_state.json"
    state = read_json(state_path, {})
    dates, bench = market_data()
    today = dates[-1].strftime("%Y-%m-%d")
    log(f"latest London market date: {today}")

    mask_all, all_tickers, sector_of = load_membership(dates)
    raw_c, raw_v = download_all(all_tickers)
    raw_c = clean_prices(raw_c.reindex(dates))
    raw_v = raw_v.reindex(dates)
    have = [t for t in all_tickers if t in raw_c.columns and raw_c[t].notna().sum() > 300]
    close, volume = raw_c[have].ffill(), raw_v[have].fillna(0)
    fresh = raw_c[have].notna().astype(int).rolling(5, min_periods=1).max().astype(bool)
    mask = mask_all.reindex(columns=have).fillna(False).astype(bool) & fresh
    coverage = ((mask_all[have].sum(axis=1) / mask_all.sum(axis=1)).groupby(dates.year).mean()).round(3)
    mkt = bench[MARKET]

    days_since = len(dates[dates > pd.Timestamp(state["last_rebalance"])]) if state.get("last_rebalance") else None
    monthly = FORCE_REBALANCE or not state.get("positions") or (days_since is not None and days_since >= H)

    earn_f, earn_cov = None, None
    if USE_EARNINGS:
        earn = fetch_earnings(have, DATA / "uk_earnings.csv.gz", log=log)
        earn_f, _, earn_cov = earnings_features(earn, dates, close, mkt, mask)
        log(f"earnings data covers {earn_cov:.0%} of member-days")

    backtest = read_json(DATA / "uk_backtest.json", None)
    if monthly or backtest is None:
        log("MONTHLY: walk-forward backtest")
        keep = dates
        df = build_dataset(close, volume, mkt, mask, sector_of, keep, earn_f)
        log(f"dataset: {len(df):,} rows")
        train_set = set(dates[::TRAIN_STEP])
        preds, targets = walk_forward(df, dates, train_set)
        backtest = run_backtest(preds, targets, close, bench, mask, sector_of)
        backtest["coverage"] = {str(k): float(v) for k, v in coverage.items()}
        backtest["earnings_coverage"] = earn_cov
        backtest["generated"] = today
        daily_sharpes = {m: backtest["table"][f"{m} · daily check"]["Sharpe"] for m in ML_MODELS}
        backtest["model"] = max(daily_sharpes, key=daily_sharpes.get) if SIGNAL_MODEL == "auto" else SIGNAL_MODEL
        write_json(DATA / "uk_backtest.json", backtest)
    else:
        keep = dates[sorted(set(range(0, len(dates), TRAIN_STEP)) | set(range(len(dates) - 2 * SMOOTH_DAYS, len(dates))))]
        df = build_dataset(close, volume, mkt, mask, sector_of, keep, earn_f)
    model = backtest["model"]

    # --- score today with the chosen model (trained on all labelled history)
    lab = df.dropna(subset=["target"])
    lab = lab[lab.index.get_level_values("date").isin(set(dates[::TRAIN_STEP]))]
    recent_days = dates[-2 * SMOOTH_DAYS:]
    rdf = df[df.index.get_level_values("date").isin(recent_days)].copy()
    for name, m in make_models().items():
        if name in ENSEMBLES.get(model, [model]):
            m.fit(lab[feats_for(name)], lab["target"]); rdf[name] = m.predict_proba(rdf[feats_for(name)])[:, 1]
    rdf = add_derived(rdf)
    live_now = mask.columns[mask.iloc[-1].values]
    wide = rdf[model].unstack("ticker")
    raw_today = wide.iloc[-1].dropna()
    scores = raw_today[raw_today.index.isin(live_now)].sort_values(ascending=False)          # monthly re-pick
    smooth = wide.ewm(span=SMOOTH_DAYS, min_periods=1).mean().iloc[-1].dropna()
    smooth = smooth[smooth.index.isin(scores.index)].sort_values(ascending=False)          # daily check
    prices_today = close.iloc[-1]

    # --- mark to market
    if state.get("positions"):
        for t, p in state["positions"].items():
            px = float(prices_today.get(t, np.nan))
            if np.isfinite(px) and p["last_px"]:
                p["value_gbp"] *= px / p["last_px"]; p["last_px"] = px
    state.setdefault("cash_gbp", START_CAPITAL); state.setdefault("positions", {})
    if "bench_units" not in state:
        state["bench_units"] = {k: START_CAPITAL / float(bench[v].iloc[-1]) for k, v in BENCHMARKS.items()}
        state["start_date"] = today

    trades = []
    if monthly:
        value = state["cash_gbp"] + sum(p["value_gbp"] for p in state["positions"].values())
        picks = pick_with_caps(scores.index, TOP_N, MAX_PER_SECTOR, sector_of)
        target_each = value / len(picks) / (1 + BUY_COST_BPS / 1e4)
        sells = [t for t in state["positions"] if t not in picks]
        trades += trade(state, today, sells, {}, prices_today)
        buys = {}
        for t in picks:
            have_v = state["positions"].get(t, {}).get("value_gbp", 0.0)
            gap = target_each - have_v
            if gap > max(2.0, 0.02 * value):
                buys[t] = gap
            elif gap < -max(2.0, 0.02 * value):                      # trim
                state["positions"][t]["value_gbp"] += gap
                state["cash_gbp"] += -gap * (1 - SELL_COST_BPS / 1e4)
                trades.append({"date": today, "ticker": t, "action": "TRIM", "change_gbp": round(gap, 2),
                               "cost_gbp": round(-gap * SELL_COST_BPS / 1e4, 3)})
        trades += trade(state, today, [], buys, prices_today)
        state["last_rebalance"] = today
        state["model"] = model
        log(f"monthly re-pick: {', '.join(picks)}")
    else:
        w = pd.Series({t: p["value_gbp"] for t, p in state["positions"].items()})
        swaps, _ = daily_swaps(w, smooth, sector_of)
        for old, new in swaps:
            amount = state["positions"][old]["value_gbp"] * (1 - SELL_COST_BPS / 1e4) / (1 + BUY_COST_BPS / 1e4)
            trades += trade(state, today, [old], {new: amount}, prices_today)
        log(f"daily check: {len(swaps)} swap(s) {swaps}")

    ranks = smooth.rank(ascending=False)
    for t, p in state["positions"].items():
        p["rank_today"] = int(ranks.get(t, 999))
        p["sector"] = sector_of.get(t, "Unknown")
    if trades:
        tp = DATA / "uk_trades.csv"
        old = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(trades)], ignore_index=True).to_csv(tp, index=False)
        state["last_trades"] = trades
        state["last_trade_date"] = today

    value = state["cash_gbp"] + sum(p["value_gbp"] for p in state["positions"].values())
    bvals = {k: state["bench_units"][k] * float(bench[v].iloc[-1]) for k, v in BENCHMARKS.items()}
    nav_path = DATA / "uk_nav.csv"
    nav = pd.read_csv(nav_path) if nav_path.exists() else pd.DataFrame()
    row = {"date": today, "portfolio_gbp": round(value, 4), **{k: round(v, 4) for k, v in bvals.items()}}
    nav = pd.concat([nav[nav["date"] != today] if len(nav) else nav, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    nav.to_csv(nav_path, index=False)
    write_json(state_path, state)

    prev = nav.iloc[-2] if len(nav) > 1 else None
    table = backtest["table"]
    key = f"{model} · daily check"
    results = {
        "market": "UK (FTSE 350)", "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "market_date": today, "model": model, "buffer_rank": BUFFER_RANK,
        "paper": {"start_date": state["start_date"], "start_value_gbp": START_CAPITAL, "value_gbp": round(value, 2),
                  "cash_gbp": round(state["cash_gbp"], 2), "return_since_start": round(value / START_CAPITAL - 1, 4),
                  "benchmarks": {k: {"value_gbp": round(v, 2), "return_since_start": round(v / START_CAPITAL - 1, 4)}
                                 for k, v in bvals.items()},
                  "day_change": round(value / prev["portfolio_gbp"] - 1, 4) if prev is not None else None},
        "holdings": sorted([{"ticker": t, "sector": p["sector"], "value_gbp": round(p["value_gbp"], 2),
                             "rank_today": p["rank_today"], "return_since_entry": round(p["last_px"] / p["entry_px"] - 1, 4),
                             "entry_date": p["entry_date"], "price_gbp": round(p.get("px_gbp", 0) * p["last_px"] / p["entry_px"], 4),
                             "shares": round(p["value_gbp"] / (p.get("px_gbp") * p["last_px"] / p["entry_px"]), 4) if p.get("px_gbp") else None}
                            for t, p in state["positions"].items()], key=lambda h: h["rank_today"]),
        "watchlist": [{"ticker": t, "rank": int(ranks[t]), "sector": sector_of.get(t, "Unknown"), "score": round(float(smooth[t]), 4)}
                      for t in smooth.index[:15]],
        "smooth_days": SMOOTH_DAYS, "max_swaps_per_day": MAX_SWAPS_PER_DAY,
        "today": {"type": "monthly re-pick" if monthly else "daily check", "trades": trades},
        "last_trades": {"date": state.get("last_trade_date"), "trades": state.get("last_trades", [])},
        "last_rebalance": state["last_rebalance"],
        "next_rebalance_in_trading_days": max(H - len(dates[dates > pd.Timestamp(state["last_rebalance"])]), 0),
        "backtest": {k: backtest.get(k) for k in ("generated", "period", "auc", "table", "coverage", "earnings_coverage",
                                                    "daily_vs_monthly")},
        "verdict": {"model_sharpe": table[key]["Sharpe"],
                    "ftse100_sharpe": table["FTSE 100 tracker"]["Sharpe"], "ftse250_sharpe": table["FTSE 250 tracker"]["Sharpe"],
                    "equal_weight_sharpe": table["Equal-weight (all)"]["Sharpe"],
                    "beats_ftse100": table[key]["Sharpe"] > table["FTSE 100 tracker"]["Sharpe"],
                    "beats_ftse250": table[key]["Sharpe"] > table["FTSE 250 tracker"]["Sharpe"]},
        "costs": {"buy_bps": BUY_COST_BPS, "sell_bps": SELL_COST_BPS, "note": "0.5% stamp duty + spread on UK buys"},
        "disclaimer": "Paper trading / research only. Not financial advice.",
    }
    write_json(DATA / "results.json", results)
    write_json(DOCS / "results.json", results)
    from dashboard import build_dashboard
    (DOCS / "index.html").write_text(build_dashboard(results, backtest, nav), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    log(f"done: paper £{value:.2f} | " + ", ".join(f"{k} £{v:.2f}" for k, v in bvals.items()))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
