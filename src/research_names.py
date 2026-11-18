"""Fetch company names for every FTSE 350 ticker we've ever held (for news searches)."""
import time

import pandas as pd
import yfinance as yf

from trading_system import ROOT

mem = pd.read_csv(ROOT / "data/uk/ftse350_membership.csv")
tick = sorted({t for s in mem.tickers for t in s.replace(";", ",").split(",") if t.strip()})
rows = []
for t in tick:
    info = {}
    for _ in range(3):
        try:
            info = yf.Ticker(t).info or {}
            break
        except Exception:  # noqa: BLE001
            time.sleep(3)
    rows.append({"ticker": t, "long": info.get("longName"), "short": info.get("shortName")})
pd.DataFrame(rows).to_csv(ROOT / "research" / "company_names.csv", index=False)
print(pd.DataFrame(rows).isna().sum())
