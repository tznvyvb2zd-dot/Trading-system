"""Helper: download the files listed in research/urls.txt into research/raw/ (run on GitHub Actions)."""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "research" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept-Language": "en-GB,en;q=0.9"}
status = []
for line in (ROOT / "research" / "urls.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    name, url = line.split(maxsplit=1)
    try:
        r = requests.get(url, headers=UA, timeout=60)
        (RAW / name).write_bytes(r.content)
        status.append(f"{name}\t{r.status_code}\t{len(r.content)}")
    except Exception as e:  # noqa: BLE001
        status.append(f"{name}\tERROR\t{e}")
(RAW / "_status.tsv").write_text("\n".join(status) + "\n")
print("\n".join(status))
sys.exit(0)
