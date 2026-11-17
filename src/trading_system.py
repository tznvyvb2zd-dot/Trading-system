"""
Daily trading system (paper trading, £100).

Every weekday (run by GitHub Actions):
  * marks the paper portfolio to market and records its value vs an S&P 500 tracker
  * every ~1 month (21 trading days) re-runs the walk-forward backtest on the REAL
    point-in-time S&P 500, retrains the chosen model and rebalances the paper portfolio
  * writes data/results.json (machine-readable, for Claude) and docs/index.html (dashboard)

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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- settings
ROOT = Path(os.environ.get("TS_ROOT", Path(__file__).resolve().parents[1]))
DATA, DOCS = ROOT / "data", ROOT / "docs"

MEMBERSHIP_URL = os.environ.get("TS_MEMBERSHIP_URL",
    "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SECTORS_URL = os.environ.get("TS_SECTORS_URL", "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv")
RENAMES = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "PKI": "RVTY", "WLTW": "WTW", "CTL": "LUMN",
           "RE": "EG", "FLT": "CPAY", "HCP": "DOC", "PEAK": "DOC"}

BENCHMARK, FX_TICKER = "SPY", "GBPUSD=X"
START = "2012-01-01"
H = 21                 # rebalance / holding period in trading days (~1 month)
TRAIN_STEP = 5
RETRAIN_EVERY = 126
MIN_TRAIN = 504

START_CAPITAL = 100.0  # £
COST_BPS = 0           # Trading 212 USD wallet
TOP_N = 10
MAX_PER_SECTOR = 2
SIGNAL_MODEL = os.environ.get("TS_SIGNAL_MODEL", "auto")   # or a model name

AS_OF = os.environ.get("AS_OF")                       # testing/backfill: pretend today is this date
FORCE_REBALANCE = os.environ.get("FORCE_REBALANCE", "").lower() in ("1", "true", "yes")

ML_MODELS = ["Logistic regression", "LightGBM", "Ensemble"]
SCORES = ML_MODELS + ["Momentum rule"]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- data
def fix(t: str) -> str:
    t = t.strip()
    return RENAMES.get(t, t).replace(".", "-")


def load_membership():
    mem = pd.read_csv(MEMBERSHIP_URL, parse_dates=["date"]).sort_values("date")
    mem = mem[mem["date"] >= pd.Timestamp(START) - pd.Timedelta(days=400)]
    sets = [set(fix(t) for t in s.split(",")) for s in mem["tickers"]]
    tickers = sorted(set().union(*sets))
    rows = pd.DataFrame([[t in s for t in tickers] for s in sets], index=mem["date"].values, columns=tickers)
    rows = rows[~rows.index.duplicated(keep="last")]
    sec = pd.read_csv(SECTORS_URL)
    sector_of = {fix(s): g for s, g in zip(sec["Symbol"], sec["GICS Sector"])}
    return rows, tickers, sector_of


def _download(tickers, start=START):
    r = yf.download(list(tickers), start=start, auto_adjust=True, progress=False, threads=True)
    c, v = r["Close"], r["Volume"]
    if isinstance(c, pd.Series):  # single ticker
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


def market_data():
    b, _ = _download([BENCHMARK, FX_TICKER])
    dates = b[BENCHMARK].dropna().index
    if AS_OF:
        dates = dates[dates <= pd.Timestamp(AS_OF)]
    return dates, b[BENCHMARK].reindex(dates).ffill(), b[FX_TICKER].reindex(dates).ffill()


# ---------------------------------------------------------------- features (as notebook v5)
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


def build_dataset(close, volume, spy, mask, sector_of, keep_dates):
    cols = close.columns
    r1, mkt_r1 = close.pct_change(fill_method=None), spy.pct_change()
    mkt_ret20, mkt_vol20 = spy.pct_change(20), mkt_r1.rolling(20).std()
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

    fwd = close.shift(-H) / close - 1                      # FUTURE return: training label only
    fwd_m = fwd.where(mask)
    target = fwd_m.gt(fwd_m.median(axis=1), axis=0).astype(float).where(fwd_m.notna())

    km = mask.loc[keep_dates]
    to_long = lambda w: w.loc[keep_dates].where(km).unstack().swaplevel()
    df = pd.concat({k: to_long(f[k]) for k in FEATURES}, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().astype("float32")
    df["fwd_ret"] = to_long(fwd).reindex(df.index)
    df["target"] = to_long(target).reindex(df.index)
    df.index.names = ["date", "ticker"]
    return df.sort_index()


def make_models():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=42, verbose=-1),
    }


def walk_forward(df, dates, rebal_set):
    lab = df.dropna(subset=["target"])
    lab_d, all_d = lab.index.get_level_values("date"), df.index.get_level_values("date")
    outs = []
    for start in range(MIN_TRAIN, len(dates), RETRAIN_EVERY):
        test_days = [d for d in dates[start:start + RETRAIN_EVERY] if d in rebal_set]
        test = df[all_d.isin(test_days)]
        if test.empty:
            continue
        train = lab[lab_d <= dates[start - H - 1]]                 # gap = holding period: no leakage
        out = test[["fwd_ret", "target", "ret_60d"]].copy()
        for name, m in make_models().items():
            m.fit(train[FEATURES], train["target"])
            out[name] = m.predict_proba(test[FEATURES])[:, 1]
        outs.append(out)
        log(f"walk-forward: trained to {dates[start - H - 1].date()}")
    p = pd.concat(outs)
    p["Ensemble"] = p[["Logistic regression", "LightGBM"]].mean(axis=1)
    p["Momentum rule"] = p["ret_60d"]
    return p


def pick_with_caps(ranked, n, cap, sector_of):
    picks, count = [], {}
    for t in ranked:
        s = sector_of.get(t, "Unknown")
        if cap and s != "Unknown" and count.get(s, 0) >= cap:
            continue
        picks.append(t); count[s] = count.get(s, 0) + 1
        if len(picks) == n:
            break
    return picks


def run_backtest(preds, spy, fx, sector_of):
    gbp_mult = (fx / fx.shift(-H)).fillna(1)
    spy_fwd = spy.shift(-H) / spy - 1
    curves = {}
    for s in SCORES + ["Equal-weight (all)"]:
        rets, prev = {}, pd.Series(dtype=float)
        for d, day in preds.groupby(level="date"):
            day = day.droplevel("date").dropna(subset=["fwd_ret"])
            if day.empty:
                continue
            if s == "Equal-weight (all)":
                w = pd.Series(1 / len(day), index=day.index)
            else:
                picks = pick_with_caps(day[s].sort_values(ascending=False).index, TOP_N, MAX_PER_SECTOR, sector_of)
                w = pd.Series(1 / len(picks), index=picks)
            gross = (1 + (w * day["fwd_ret"].reindex(w.index)).sum()) * gbp_mult.loc[d]
            rets[d] = gross - 1 - w.sub(prev, fill_value=0).abs().sum() * COST_BPS / 10_000
            prev = w
        curves[s] = pd.Series(rets)
    idx = curves["Equal-weight (all)"].index
    curves["S&P 500 tracker (£)"] = pd.Series({d: (1 + spy_fwd.loc[d]) * gbp_mult.loc[d] - 1 for d in idx})

    ppy = 252 / H
    def stats(r):
        eq = (1 + r).cumprod()
        return {"CAGR": eq.iloc[-1] ** (ppy / len(r)) - 1, "Volatility": r.std() * np.sqrt(ppy),
                "Sharpe": r.mean() / r.std() * np.sqrt(ppy), "Max drawdown": float((eq / eq.cummax() - 1).min()),
                "Win rate": float((r > 0).mean()), "£100 grew to": float(eq.iloc[-1] * 100)}
    table = {s: {k: round(float(v), 4) for k, v in stats(r).items()} for s, r in curves.items()}
    from sklearn.metrics import roc_auc_score
    sc = preds.dropna(subset=["target"])
    auc = {m: round(float(roc_auc_score(sc["target"], sc[m])), 4) for m in SCORES}
    yearly = {s: {str(y): round(float(v), 4) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}
              for s, r in curves.items()}
    growth = {s: {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in ((1 + r).cumprod() * 100).items()}
              for s, r in curves.items()}
    return {"table": table, "auc": auc, "yearly": yearly, "growth": growth,
            "period": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")]}


# ---------------------------------------------------------------- paper portfolio
def read_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def write_json(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str))


def rebalance(state, today, close_today, fx_today, spy_today, picks, model, sector_of, scores):
    """Sell everything not wanted, buy new picks so each is 1/N of current portfolio value (paper)."""
    pos = {t: p for t, p in state.get("positions", {}).items()}
    value = state.get("cash_gbp", START_CAPITAL) + sum(
        p["shares"] * close_today.get(t, p["last_price_usd"]) / fx_today for t, p in pos.items())
    target_gbp = value / len(picks)
    trades = []
    old = {t: p["shares"] * close_today.get(t, p["last_price_usd"]) / fx_today for t, p in pos.items()}
    new_pos = {}
    for t in picks:
        px = float(close_today[t])
        new_pos[t] = {"shares": target_gbp * fx_today / px, "entry_date": pos.get(t, {}).get("entry_date", today),
                      "entry_price_usd": pos.get(t, {}).get("entry_price_usd", px), "last_price_usd": px,
                      "sector": sector_of.get(t, "Unknown"), "score": round(float(scores.get(t, np.nan)), 4)}
    for t in sorted(set(old) | set(new_pos)):
        before, after = old.get(t, 0.0), (target_gbp if t in new_pos else 0.0)
        change = after - before
        if before == 0:
            action, why = "BUY", "New in the top 10"
        elif after == 0:
            action, why = "SELL", "Dropped out of the top 10"
        elif abs(change) >= max(2, 0.02 * value):
            action, why = ("RESIZE (top up)" if change > 0 else "RESIZE (trim)"), "Still in, size changed"
        else:
            action, why = "KEEP", "Still in the top 10"
        trades.append({"date": today, "ticker": t, "action": action, "why": why,
                       "before_gbp": round(before, 2), "after_gbp": round(after, 2), "change_gbp": round(change, 2)})
    cost = sum(abs(x["change_gbp"]) for x in trades) * COST_BPS / 10_000
    state.update(positions=new_pos, cash_gbp=-cost, last_rebalance=today, model=model)
    if "spy_units" not in state:  # benchmark: £100 in an S&P 500 tracker on day one
        state.update(spy_units=START_CAPITAL * fx_today / spy_today, start_date=today)
    order = {"SELL": 0, "BUY": 1, "RESIZE (top up)": 2, "RESIZE (trim)": 2, "KEEP": 3}
    return sorted(trades, key=lambda x: (order[x["action"]], x["ticker"]))


# ---------------------------------------------------------------- main
def main():
    DATA.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    state = read_json(DATA / "state.json", {})
    dates, spy, fx = market_data()
    today = dates[-1].strftime("%Y-%m-%d")
    log(f"latest market date: {today}")

    days_since = (len(dates[dates > pd.Timestamp(state["last_rebalance"])]) if state.get("last_rebalance") else None)
    need_rebalance = FORCE_REBALANCE or not state.get("positions") or (days_since is not None and days_since >= H)
    backtest = read_json(DATA / "backtest.json", None)
    warnings_list = []

    if need_rebalance:
        log("REBALANCE DAY: full download + backtest + retrain")
        member_rows, all_tickers, sector_of = load_membership()
        raw_c, raw_v = download_all(all_tickers)
        raw_c, raw_v = raw_c.reindex(dates), raw_v.reindex(dates)
        have = [t for t in all_tickers if t in raw_c.columns and raw_c[t].notna().sum() > 300]
        close, volume = raw_c[have].ffill(), raw_v[have].fillna(0)
        mask_all = member_rows.reindex(dates, method="ffill").fillna(False).astype(bool)
        fresh = raw_c[have].notna().astype(int).rolling(5, min_periods=1).max().astype(bool)
        mask = mask_all.reindex(columns=have).fillna(False).astype(bool) & fresh
        coverage = ((mask_all[have].sum(axis=1) / mask_all.sum(axis=1)).groupby(dates.year).mean()).round(3)

        rebal_dates = dates[::H]
        keep_idx = sorted(set(range(0, len(dates), TRAIN_STEP)) | set(range(0, len(dates), H)) | {len(dates) - 1})
        df = build_dataset(close, volume, spy, mask, sector_of, dates[keep_idx])
        log(f"dataset: {len(df):,} rows")
        preds = walk_forward(df, dates, set(rebal_dates))
        backtest = run_backtest(preds, spy, fx, sector_of)
        backtest["coverage"] = {str(k): float(v) for k, v in coverage.items()}
        backtest["generated"] = today
        ml_sharpes = {m: backtest["table"][m]["Sharpe"] for m in ML_MODELS}
        model = max(ml_sharpes, key=ml_sharpes.get) if SIGNAL_MODEL == "auto" else SIGNAL_MODEL
        backtest["model"] = model
        write_json(DATA / "backtest.json", backtest)

        lab = df.dropna(subset=["target"])
        latest = df.index.get_level_values("date").max()
        tdf = df.xs(latest, level="date").copy()
        if model in ML_MODELS:
            for name, m in make_models().items():
                m.fit(lab[FEATURES], lab["target"]); tdf[name] = m.predict_proba(tdf[FEATURES])[:, 1]
            tdf["Ensemble"] = tdf[["Logistic regression", "LightGBM"]].mean(axis=1)
        tdf["Momentum rule"] = tdf["ret_60d"]
        picks = pick_with_caps(tdf[model].sort_values(ascending=False).index, TOP_N, MAX_PER_SECTOR, sector_of)
        trades = rebalance(state, today, close.loc[dates[-1]], float(fx.iloc[-1]), float(spy.iloc[-1]),
                           picks, model, sector_of, tdf[model].to_dict())
        tr_path = DATA / "trades.csv"
        tr = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
        tr = tr[tr.get("date", pd.Series(dtype=str)) != today] if len(tr) else tr
        pd.concat([tr, pd.DataFrame(trades)], ignore_index=True).to_csv(tr_path, index=False)
        state["last_trades"] = trades
        log(f"rebalanced into: {', '.join(picks)}")
    else:
        log(f"mark-to-market only ({days_since} trading days since last rebalance)")

    # --- mark to market (holdings + SPY + FX only on normal days: fast) ---
    held = list(state["positions"])
    try:
        pc, _ = _download(held, start=(dates[-1] - pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
        pc = pc.reindex(dates[dates >= dates[-1] - pd.Timedelta(days=15)]).ffill()
        last_px = pc.iloc[-1]
    except Exception as e:  # noqa: BLE001
        last_px = pd.Series(dtype=float); warnings_list.append(f"price download failed: {e}")
    fx_now, spy_now = float(fx.iloc[-1]), float(spy.iloc[-1])
    holdings = []
    for t, p in state["positions"].items():
        px = float(last_px.get(t, np.nan))
        if np.isnan(px):
            px = p["last_price_usd"]; warnings_list.append(f"no fresh price for {t}; using last known")
        p["last_price_usd"] = px
        holdings.append({"ticker": t, "sector": p["sector"], "shares": round(p["shares"], 6), "price_usd": round(px, 2),
                         "value_gbp": round(p["shares"] * px / fx_now, 2), "entry_date": p["entry_date"],
                         "return_since_entry": round(px / p["entry_price_usd"] - 1, 4), "score": p.get("score")})
    value = sum(h["value_gbp"] for h in holdings) + state.get("cash_gbp", 0)
    sp_value = state["spy_units"] * spy_now / fx_now
    nav_path = DATA / "nav.csv"
    nav = pd.read_csv(nav_path) if nav_path.exists() else pd.DataFrame(columns=["date", "portfolio_gbp", "sp500_gbp", "gbpusd"])
    nav = nav[nav["date"] != today]
    nav = pd.concat([nav, pd.DataFrame([{"date": today, "portfolio_gbp": round(value, 4), "sp500_gbp": round(sp_value, 4),
                                         "gbpusd": round(fx_now, 5)}])], ignore_index=True).sort_values("date")
    nav.to_csv(nav_path, index=False)
    write_json(DATA / "state.json", state)

    prev = nav.iloc[-2] if len(nav) > 1 else None
    remaining = H - len(dates[dates > pd.Timestamp(state["last_rebalance"])])
    bt = backtest or {}
    table = bt.get("table", {})
    model = state.get("model")
    results = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "market_date": today, "gbpusd": round(fx_now, 4), "model": model,
        "paper": {"start_date": state["start_date"], "start_value_gbp": START_CAPITAL,
                  "value_gbp": round(value, 2), "sp500_value_gbp": round(sp_value, 2),
                  "return_since_start": round(value / START_CAPITAL - 1, 4),
                  "sp500_return_since_start": round(sp_value / START_CAPITAL - 1, 4),
                  "day_change": round(value / prev["portfolio_gbp"] - 1, 4) if prev is not None else None,
                  "sp500_day_change": round(sp_value / prev["sp500_gbp"] - 1, 4) if prev is not None else None},
        "holdings": sorted(holdings, key=lambda h: -h["value_gbp"]),
        "last_rebalance": {"date": state["last_rebalance"], "trades": state.get("last_trades", [])},
        "next_rebalance_in_trading_days": max(remaining, 0),
        "rebalanced_today": bool(need_rebalance),
        "backtest": {"generated": bt.get("generated"), "period": bt.get("period"), "auc": bt.get("auc"),
                     "table": table, "coverage": bt.get("coverage")},
        "verdict": ({"model_sharpe": table[model]["Sharpe"], "equal_weight_sharpe": table["Equal-weight (all)"]["Sharpe"],
                     "sp500_sharpe": table["S&P 500 tracker (£)"]["Sharpe"],
                     "beats_equal_weight": table[model]["Sharpe"] > table["Equal-weight (all)"]["Sharpe"],
                     "beats_sp500": table[model]["Sharpe"] > table["S&P 500 tracker (£)"]["Sharpe"]} if model in table else None),
        "warnings": warnings_list,
        "disclaimer": "Paper trading / research only. Not financial advice.",
    }
    write_json(DATA / "results.json", results)
    write_json(DOCS / "results.json", results)   # also published on the web page for Claude/phone
    from dashboard import build_dashboard
    (DOCS / "index.html").write_text(build_dashboard(results, bt, nav), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    log(f"done: paper £{value:.2f} vs S&P £{sp_value:.2f}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
