"""Summarise Vertiq LM's scored runs (lm/runs/*.json, written by lm/evaluate.py) into
benchmarks/results/lm.json: overall, seen and never-seen domains, raw and after the data repair
(vertiq.mapping.repair), recomputed here from each run's saved answers so every model is repaired by
the same code. Model outputs themselves are not copied, only the scores.

Run:  python benchmarks/collect_lm.py [--lm lm]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vertiq.mapping import Mapping, apply, repair, score  # noqa: E402

# (run name, model, training, test set, prompt)
RUNS = [
    ("base", "Qwen 3.5 4B, no training", "none", "v1", "v1"),
    ("tuned", "+ LoRA fine-tune (round 1)", "990 generated tables, 500 steps", "v1", "v1"),
    ("pref", "+ ORPO on its own mistakes", "256 preference pairs, 250 steps", "v1", "v1"),
    ("v2-sft-on-v1", "Round 3 fine-tune", "990 realistic tables + 22 teacher tables, 1000 steps, column kinds in prompt", "v1", "v2"),
    ("pref-on-v2", "+ ORPO on its own mistakes (round 1 model)", "as above", "v2", "v1"),
    ("v2-sft", "Round 3 fine-tune", "as above", "v2", "v2"),
    ("v2-pref-on-v1", "Round 3 + ORPO on its own mistakes", "245 preference pairs from round 3's errors (examples over 460 tokens "
     "left out), 980 single-example steps", "v1", "v2"),
    ("v2-pref", "Round 3 + ORPO on its own mistakes", "as above", "v2", "v2"),
]


def summarise(run: dict, tests: list[dict]) -> dict:
    by = {"overall": [], "seen": [], "unseen": []}
    for ex, rec in zip(tests, run["examples"]):
        truth = Mapping.parse(ex["messages"][1]["content"])
        try:
            pred = Mapping.parse(rec["output"])
            fixed = repair(pred, ex["rows"])[0]
            r = {"json": 1, "loads": int(not apply(pred, ex["rows"])[1]), "exact": int(score(pred, truth)["exact"]),
                 "loads_repaired": int(not apply(fixed, ex["rows"])[1]), "exact_repaired": int(score(fixed, truth)["exact"])}
        except (ValueError, json.JSONDecodeError):
            r = {"json": 0, "loads": 0, "exact": 0, "loads_repaired": 0, "exact_repaired": 0}
        for k in ("overall", "seen" if ex["seen"] else "unseen"):
            by[k].append(r)
    return {k: {"n": len(v), **{m: round(sum(x[m] for x in v) / len(v), 3) for m in v[0]}} for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default=str(ROOT / "lm"))
    a = ap.parse_args()
    lm = Path(a.lm)
    sets = {"v1": [json.loads(l) for l in open(lm / "data" / "test.jsonl")],
            "v2": [json.loads(l) for l in open(lm / "data" / "v2" / "test.jsonl")]}
    out = []
    for name, model, training, test, prompt in RUNS:
        f = lm / "runs" / f"{name}.json"
        if not f.exists():
            continue
        out.append({"run": name, "model": model, "training": training, "test_set": test, "prompt": prompt,
                    **summarise(json.loads(f.read_text()), sets[test])})
    doc = {"generated": datetime.utcnow().strftime("%Y-%m-%d"),
           "hardware": "Apple M1 Pro, 16 GB, MLX 0.32 (4-bit weights, LoRA rank 8 on 4 layers)",
           "base_model": "mlx-community/Qwen3.5-4B-MLX-4bit (Apache-2.0)",
           "test_sets": {"v1": "114 tables, 15 domains, 4 never seen in training (energy, insurance, manufacturing, retail_sales)",
                         "v2": "114 more realistic tables (real audit columns, locale amounts), same split"},
           "metrics": {"json": "answer is valid JSON", "loads": "the mapping loads every sample row",
                       "exact": "every field matches the known answer",
                       "_repaired": "the same after vertiq.mapping.repair (names snapped to real columns, roles held to their value types, date format decided by the rows)"},
           "runs": out}
    dest = ROOT / "benchmarks" / "results" / "lm.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2))
    for r in out:
        o, u = r["overall"], r["unseen"]
        print(f"{r['run']:14} {r['test_set']} loads {o['loads']:.3f} exact {o['exact']:.3f} | repaired {o['loads_repaired']:.3f} / {o['exact_repaired']:.3f} | unseen exact {u['exact']:.3f} / {u['exact_repaired']:.3f}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
