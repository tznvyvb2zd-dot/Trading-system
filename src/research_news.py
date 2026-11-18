"""Collect daily GDELT news tone + article counts (2017 to today) for each company in research/news_queries.csv.
Sharded so several GitHub jobs can run in parallel: research_news_<k>.py calls collect(k, N_SHARDS)."""
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.gdeltproject.org/api/v2/doc/doc"
N_SHARDS = 4
OUT = ROOT / "research" / "news"


def fetch(query, mode, start="20170101000000"):
    end = pd.Timestamp.utcnow().strftime("%Y%m%d000000")
    wait = 20
    for attempt in range(7):
        try:
            r = requests.get(API, params={"query": query, "mode": mode, "format": "json", "startdatetime": start,
                                          "enddatetime": end, "timelinesmooth": 0}, timeout=120)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                tl = r.json().get("timeline") or []
                if not tl:
                    return pd.Series(dtype=float)
                d = pd.DataFrame(tl[0]["data"])
                return pd.Series(d["value"].values, index=pd.to_datetime(d["date"].str[:8]), dtype=float)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(wait); wait = min(wait * 2, 300)
    return None


def collect(shard, n=N_SHARDS):
    OUT.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(ROOT / "research" / "news_queries.csv")
    q = q.iloc[shard::n]
    rows, failed = [], []
    for i, (t, query) in enumerate(zip(q.ticker, q["query"])):
        tone = fetch(query, "timelinetone"); time.sleep(8)
        vol = fetch(query, "timelinevolraw"); time.sleep(8)
        if tone is None or vol is None:
            failed.append(t); continue
        df = pd.DataFrame({"tone": tone, "articles": vol}).fillna(0)
        df = df[(df.tone != 0) | (df.articles != 0)]
        df["ticker"] = t
        rows.append(df.rename_axis("date").reset_index())
        print(f"[{shard}] {i + 1}/{len(q)} {t} {query}: {len(df)} days", flush=True)
    out = pd.concat(rows) if rows else pd.DataFrame(columns=["date", "tone", "articles", "ticker"])
    out.to_csv(OUT / f"news_{shard}.csv.gz", index=False)
    pd.Series(failed, dtype=str).to_csv(OUT / f"failed_{shard}.csv", index=False)
    print(f"[{shard}] done, failed: {failed}")
