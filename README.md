# 📈 Trading System (paper trading, £100)

A machine-learning stock-picking system that runs **every weekday** on GitHub Actions and publishes a **live dashboard**.

- **Dashboard:** `https://tznvyvb2zd-dot.github.io/Trading-system/`
- **Data for Claude / scripts:** `docs/results.json` (also at `.../Trading-system/results.json`)

## What it does
| When | What |
|---|---|
| Every weekday, 22:30 UTC | Updates prices, values the £100 paper portfolio vs an S&P 500 tracker, rebuilds the dashboard |
| Every ~21 trading days (monthly) | Downloads the **real point-in-time S&P 500** (~830 stocks, incl. ones later removed), re-runs a walk-forward backtest, retrains the best model (logistic regression / LightGBM / ensemble), and rebalances into the top 10 (max 2 per sector) |

## Files
| Path | Contents |
|---|---|
| `src/trading_system.py` | The whole pipeline |
| `src/dashboard.py` | Builds `docs/index.html` |
| `data/state.json` | Current paper positions |
| `data/nav.csv` | Daily portfolio value vs S&P 500 (£) |
| `data/trades.csv` | Every paper trade (BUY / SELL / RESIZE / KEEP) |
| `data/backtest.json` | Latest backtest results |
| `docs/results.json` | Today's summary (machine-readable) |

## Manual run
**Actions → Daily trading run → Run workflow** (tick *force rebalance* to re-pick stocks today).

## Honest results so far
On the real S&P 500 (no hindsight), none of the models beat a plain S&P 500 tracker in the 2014-2026 backtest (AUC ≈ 0.50). The earlier "wins" on 30 hand-picked stocks were survivorship bias (worth up to +24%/yr of fake returns). This repo runs the system live as a **paper-trading experiment**.

Membership data: [fja05680/sp500](https://github.com/fja05680/sp500). Prices: Yahoo Finance via `yfinance`.

⚠️ Research / learning project. Not financial advice.
