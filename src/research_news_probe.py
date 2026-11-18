"""Probe: does GDELT's DOC API give daily news tone + volume back to 2017 for UK company names?"""
import json
import time

import requests

from trading_system import ROOT

API = "https://api.gdeltproject.org/api/v2/doc/doc"
out = {}
for name in ['"Tesco"', '"Rolls-Royce"', '"Rotork"', '"Mitie"', '"Tate & Lyle"', '"Playtech"']:
    for mode in ["timelinetone", "timelinevolraw"]:
        q = f'{name} sourcelang:english'
        for attempt in range(4):
            r = requests.get(API, params={"query": q, "mode": mode, "format": "json", "startdatetime": "20170101000000",
                                          "enddatetime": "20261117000000", "timelinesmooth": 0}, timeout=120)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                break
            time.sleep(10)
        try:
            data = r.json()["timeline"][0]["data"]
            vals = [d["value"] for d in data]
            out[f"{name} {mode}"] = {"n_points": len(data), "first": data[0]["date"], "last": data[-1]["date"],
                                     "nonzero": sum(1 for v in vals if v), "sample": data[:2] + data[-2:]}
        except Exception as e:  # noqa: BLE001
            out[f"{name} {mode}"] = {"error": str(e), "status": r.status_code, "text": r.text[:300]}
        time.sleep(6)
json.dump(out, open(ROOT / "research" / "news_probe.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
