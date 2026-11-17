# 🇬🇧 UK Trading System (FTSE 350, paper trading, £100)

A machine-learning stock-picking system for the **FTSE 350** that runs **every weekday** on GitHub Actions and publishes a **live dashboard**.

- **Dashboard:** `https://tznvyvb2zd-dot.github.io/Trading-system/`
- **Data for Claude / scripts:** `docs/results.json`

## How it trades
| When | What |
|---|---|
| **Monthly** (every ~21 trading days) | Re-runs an honest walk-forward backtest on the **point-in-time FTSE 350** (incl. companies that later left the index), picks the best model, and re-picks the **top 10** (max 2 per sector) |
| **Every weekday** (22:30 UTC) | Re-scores all members. **Daily check:** if a holding's 10-day smoothed rank falls below **50**, sell it and buy the best-ranked replacement (max 2 swaps/day). Values the portfolio vs FTSE 100 and FTSE 250 trackers |

Costs on every trade: **0.5% stamp duty + ~0.1% spread on UK buys**, ~0.1% spread on sells.

Models: logistic regression, LightGBM, ensemble (each with/without earnings-surprise features) and a momentum rule; each tested "monthly only" vs "monthly + daily check".

## Honest results (backtest 2012-2026, first run 17 Nov 2026)
No strategy beat simply owning the FTSE 350 equally (Sharpe 0.50) or the FTSE 250 / FTSE 100 trackers (0.43 / 0.38). Best model: logistic regression, monthly (0.35). The daily check didn't help after stamp duty. AUC ~0.51. This repo runs the system live as a **paper-trading experiment**.

## Data
- Membership: rebuilt from FTSE Russell's official *FTSE 100/250 Historic Additions and Deletions* (to Oct 2026) plus current Wikipedia constituent lists (`data/uk/`). Names matched to Yahoo tickers; companies Yahoo no longer has (mostly takeovers) are missing: ~64% of members covered in 2011, rising to ~98% in 2026.
- Prices & earnings: Yahoo Finance via `yfinance` (London prices cleaned for pence/pound glitches). UK earnings data on Yahoo is sparse (~15% of stock-days).
- Benchmarks: iShares Core FTSE 100 (ISF.L) and iShares FTSE 250 (MIDD.L).

## Files
| Path | Contents |
|---|---|
| `src/trading_system.py` | Pipeline |
| `src/earnings.py` | Earnings-surprise features |
| `src/dashboard.py` | Builds `docs/index.html` |
| `data/uk_state.json`, `uk_nav.csv`, `uk_trades.csv`, `uk_backtest.json` | Paper portfolio, daily values, trades, backtest |
| `data/us_archive/` | The earlier US (S&P 500) version's paper record |
| `research/` | Source files used to rebuild the FTSE 350 history |

Manual run: **Actions → Daily trading run → Run workflow** (tick *force rebalance* to re-pick today).

⚠️ Research / learning project. Not financial advice.
