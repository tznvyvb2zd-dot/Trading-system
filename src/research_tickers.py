"""Helper (GitHub Actions): look up Yahoo tickers for historical FTSE company names."""
import json, time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
names = json.loads((ROOT / "research" / "names_to_lookup.json").read_text())
rows = []
for i, n in enumerate(names):
    q = n.replace("(", " ").replace(")", " ").replace("*", " ")
    for attempt in range(3):
        try:
            res = yf.Search(q + " plc", max_results=8, news_count=0).quotes or yf.Search(q, max_results=8, news_count=0).quotes
            break
        except Exception:  # noqa: BLE001
            res = []; time.sleep(2 * (attempt + 1))
    if not res:
        rows.append({"name": n, "rank": 0})
    for r, x in enumerate(res):
        rows.append({"name": n, "rank": r + 1, "symbol": x.get("symbol"), "shortname": x.get("shortname"),
                     "longname": x.get("longname"), "exchange": x.get("exchange"), "type": x.get("quoteType")})
    time.sleep(0.3)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(names)}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "research" / "ticker_lookup.csv", index=False)
print("done", len(rows))
