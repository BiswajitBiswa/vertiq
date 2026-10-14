"""Score mapping proposals on held-out real exports (benchmarks/real/labels.json): the rule-based proposer
(vertiq.tables.propose) always, and any Vertiq LM run saved by lm/evaluate_real.py. Every proposal goes
through vertiq.mapping.repair first, as it does in use. A role is correct when the final column is one of the
accepted answers; a table is exact when all five roles are.

Run:  python benchmarks/real/score.py [--runs lm/runs/real-*.json]   writes benchmarks/results/real.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vertiq.mapping import ROLES, Mapping, repair  # noqa: E402
from vertiq.tables import propose  # noqa: E402

LABELS = json.loads((ROOT / "benchmarks/real/labels.json").read_text())
DATA = ROOT / "lm/data/real"


def rows(name):
    with open(DATA / f"{name}.csv", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def grade(m: Mapping | None, label: dict) -> dict:
    roles = {r: (getattr(m, r) if m else None) in label[r] for r in ROLES}
    return {"roles": roles, "correct": sum(roles.values()), "exact": all(roles.values()),
            "mapping": {r: getattr(m, r) for r in ROLES} if m else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "lm/runs/real-*.json"))
    a = ap.parse_args()
    tables = [n for n in LABELS if not n.startswith("_") and (DATA / f"{n}.csv").exists()]
    out = {"tables": tables, "methods": {}}
    res = {}
    for n in tables:
        rs = rows(n)
        m, _ = propose(rs)
        if m:
            m, _ = repair(m, rs)
        res[n] = grade(m, LABELS[n])
    out["methods"]["rules"] = res
    for path in sorted(glob.glob(a.runs)):
        run = json.loads(Path(path).read_text())
        res = {}
        for n in tables:
            raw = run["outputs"].get(n)
            try:
                m, _ = repair(Mapping.parse(raw), rows(n))
            except (ValueError, TypeError, json.JSONDecodeError):
                m = None
            res[n] = grade(m, LABELS[n])
        out["methods"][run["name"]] = res
    for method, res in out["methods"].items():
        roles = sum(r["correct"] for r in res.values()) / (5 * len(res))
        exact = sum(r["exact"] for r in res.values())
        out.setdefault("summary", {})[method] = {"roles_correct": round(roles, 3), "exact": f"{exact}/{len(res)}"}
        print(f"{method:24} roles correct {roles:.0%}  exact {exact}/{len(res)}  " +
              "  ".join(f"{n}:{r['correct']}/5" for n, r in res.items()))
    (ROOT / "benchmarks/results/real.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
