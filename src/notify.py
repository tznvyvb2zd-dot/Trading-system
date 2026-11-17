"""Phone notifications via ntfy (https://ntfy.sh). Topic comes from the NTFY_TOPIC secret; no topic = no-op."""
from __future__ import annotations

import os
import sys

import requests

DASHBOARD = "https://tznvyvb2zd-dot.github.io/Trading-system/"


def send(title: str, message: str, priority: int = 3, tags=None, click: str = DASHBOARD) -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print(f"[notify] (no NTFY_TOPIC set) {title}: {message}")
        return False
    try:
        r = requests.post("https://ntfy.sh/", json={"topic": topic, "title": title, "message": message,
                                                   "priority": priority, "tags": tags or [], "click": click}, timeout=20)
        print(f"[notify] sent ({r.status_code}): {title}")
        return r.ok
    except Exception as e:  # noqa: BLE001
        print(f"[notify] failed: {e}")
        return False


if __name__ == "__main__":            # test: python src/notify.py
    ok = send("Trading system connected ✅", "You'll get alerts here for sudden drops, SELL signals and trade days.",
              tags=["white_check_mark"])
    sys.exit(0 if ok or not os.environ.get("NTFY_TOPIC") else 1)
