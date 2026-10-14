"""Score a model (optionally with a LoRA adapter) on the held-out mapping examples.

For every test table the model's answer is parsed and then checked two ways:
  * against the known-correct mapping (field by field, exact match)
  * against the data itself: does `vertiq.mapping.apply` load all sample rows with it?
Results are reported overall, for seen vs never-seen domains, and per domain, so no domain's
result hides inside an average.

Run: lm/.venv/bin/python lm/evaluate.py --model lm/models/qwen3.5-4b-4bit [--adapter lm/adapters/mapping] --name base
     [--data lm/data/v2/test.jsonl] [--prompt 1|2]  (score an older model on newer tables in its own prompt format)
     [--shard 2/4]      score every 4th table starting at the 3rd, so several machines can split one test set
     [--progress]       print `MESH {"step": n, "metrics": {...}}` after each table, for the GPU mesh dashboard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from vertiq.mapping import Mapping, apply, repair, score  # noqa: E402


def prompt_text(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    ap.add_argument("--prompt", type=int, default=0, help="rebuild prompts as v1 or v2 from the rows (0: as stored)")
    ap.add_argument("--shard", default="", help="i/k: score tables i, i+k, i+2k, ... (0-based)")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    from mlx_lm import generate, load
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    tests = [json.loads(l) for l in open(a.data)]
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/"))
        tests = tests[i::k]
    if a.limit:
        tests = tests[:a.limit]
    rows, started = [], time.time()
    for ex in tests:
        user, truth = ex["messages"][0]["content"], Mapping.parse(ex["messages"][1]["content"])
        if a.prompt:
            from make_data import render
            user = render(ex["rows"], a.prompt)
        out = generate(model, tok, prompt=prompt_text(tok, user), max_tokens=220, verbose=False)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        rec = {"domain": ex["domain"], "seen": ex["seen"], "output": out[:400]}
        try:
            pred = Mapping.parse(out)
            s = score(pred, truth)
            _ds, problems = apply(pred, ex["rows"])
            fixed, _notes = repair(pred, ex["rows"])                  # what the data can settle on its own
            rec.update(parsed=True, exact=s["exact"], roles=s["roles_correct"], attrs=s["attrs_overlap"],
                       loads=not problems, loads_repaired=not apply(fixed, ex["rows"])[1],
                       exact_repaired=score(fixed, truth)["exact"])
        except (ValueError, json.JSONDecodeError):
            rec.update(parsed=False, exact=False, roles=0.0, attrs=0.0, loads=False, loads_repaired=False, exact_repaired=False)
        rows.append(rec)
        if a.progress:
            n = len(rows)
            print("MESH " + json.dumps({"step": n, "metrics": {"eval_exact": round(sum(r["exact"] for r in rows) / n, 4),
                                                                "eval_roles": round(sum(r["roles"] for r in rows) / n, 4)}}), flush=True)

    def summary(rs):
        n = len(rs) or 1
        return {"n": len(rs), "valid_json": round(sum(r["parsed"] for r in rs) / n, 3),
                "loads_data": round(sum(r["loads"] for r in rs) / n, 3), "exact": round(sum(r["exact"] for r in rs) / n, 3),
                "roles_correct": round(sum(r["roles"] for r in rs) / n, 3), "attrs_overlap": round(sum(r["attrs"] for r in rs) / n, 3),
                "loads_repaired": round(sum(r["loads_repaired"] for r in rs) / n, 3),
                "exact_repaired": round(sum(r["exact_repaired"] for r in rs) / n, 3)}
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    report = {"name": a.name, "model": a.model, "adapter": a.adapter, "data": a.data, "prompt": a.prompt, "seconds": round(time.time() - started),
              "overall": summary(rows), "seen_domains": summary([r for r in rows if r["seen"]]),
              "unseen_domains": summary([r for r in rows if not r["seen"]]),
              "per_domain": {d: summary(v) | {"seen": v[0]["seen"]} for d, v in sorted(by_dom.items())}}
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{a.name}.json").write_text(json.dumps(report | {"examples": rows}, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "per_domain"}, indent=2))


if __name__ == "__main__":
    main()
