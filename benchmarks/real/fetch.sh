#!/usr/bin/env bash
# Fetch the held-out real exports into lm/data/real (git-ignored). Online Retail II is downloaded separately
# (UCI, about 45 MB) and its first 500 lines saved as online_retail.csv.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p lm/data/real
python3 - <<'PY'
import json, subprocess
labels = json.load(open("benchmarks/real/labels.json"))
for name, d in labels.items():
    if name.startswith("_") or not d["source"].startswith("https://data."):
        continue
    subprocess.run(["curl", "-sSL", "--max-time", "60", "-o", f"lm/data/real/{name}.csv", d["source"]], check=True)
    print("fetched", name)
PY
