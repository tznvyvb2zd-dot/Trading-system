"""
Hourly intraday check (London market hours) of the current paper holdings.
Sends a WATCH push if a holding is down DROP_DAY today or DROP_TOTAL since bought.
The SELL decision is made at the evening run (it re-scores all ~350 stocks).
Prices from Yahoo are ~15 minutes delayed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from notify import send  # noqa: E402
from trading_system import BUFFER_RANK, DATA, DROP_DAY, DROP_TOTAL  # noqa: E402


def main():
    state_path = DATA / "uk_state.json"
    if not state_path.exists():
        print("no portfolio yet"); return
    state = json.loads(state_path.read_text())
    sent_path = DATA / "uk_alerts_sent.json"
    today = datetime.now(timezone.utc).date().isoformat()
    sent = json.loads(sent_path.read_text()) if sent_path.exists() else {}
    sent = {today: sent.get(today, [])}
    hits = []
    for t, p in state.get("positions", {}).items():
        try:
            px = float(yf.Ticker(t).fast_info["last_price"])
        except Exception as e:  # noqa: BLE001
            print(f"{t}: no price ({e})"); continue
        day, tot = px / p["last_px"] - 1, px / p["entry_px"] - 1
        print(f"{t}: {day:+.2%} today, {tot:+.2%} since bought")
        if abs(day) > 0.6:                       # almost certainly a pence/pound data glitch
            continue
        if (day <= DROP_DAY or tot <= DROP_TOTAL) and t not in sent[today]:
            hits.append((t, day, tot)); sent[today].append(t)
    if hits:
        nm = lambda t: t.replace(".L", "")
        title = "⚠️ Sudden drop: " + ", ".join(f"{nm(t)} {d:+.1%}" for t, d, _ in hits)
        lines = [f"{nm(t)}: {d:+.1%} today, {tt:+.1%} since bought" for t, d, tt in hits]
        lines.append(f"Don't panic-sell yet: tonight's run (22:30) re-scores everything and sends a 🔴 SELL only if the "
                     f"model also drops it out of the top {BUFFER_RANK}. Prices ~15 min delayed.")
        send(title, "\n".join(lines), priority=4, tags=["warning"])
    sent_path.write_text(json.dumps(sent))


if __name__ == "__main__":
    main()
