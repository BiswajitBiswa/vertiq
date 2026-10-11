"""Reward and punish, from the model's own mistakes: generate fresh tables (a new seed, only domains
the model trained on, the same number per domain so no domain is favoured), let the fine-tuned model
map them, and keep every answer that isn't exactly right as a preference pair:

  chosen   = the correct mapping (rewarded)
  rejected = what the model actually wrote (punished)

The four never-seen test domains are not touched, so the final test stays honest.
Writes lm/data/dpo/{train,valid}.jsonl and lm/runs/failures.json (per-domain counts).

Run: lm/.venv/bin/python lm/collect_failures.py --per-domain 36
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from make_data import SEED, D, example  # noqa: E402
from vertiq.mapping import Mapping, score  # noqa: E402
from evaluate import prompt_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=36)
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3.5-4b-4bit"))
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "mapping"), help="'' for a fused model")
    ap.add_argument("--meta", default=str(ROOT / "data" / "meta.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "dpo"))
    ap.add_argument("--seed-offset", type=int, default=1)
    a = ap.parse_args()
    from mlx_lm import generate, load
    meta = json.loads(Path(a.meta).read_text())
    seen = meta["seen_domains"]
    r = random.Random(SEED + a.seed_offset)          # fresh tables, never the test set
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    pairs, tried, failed, started = [], Counter(), Counter(), time.time()
    progress = ROOT / "runs" / "failures.json"
    for dom in seen:
        for _ in range(a.per_domain):
            prompt, truth = example(dom, r)
            answer = json.dumps({k: v for k, v in truth.items() if not k.startswith("_")}, ensure_ascii=False)
            out = generate(model, tok, prompt=prompt_text(tok, prompt), max_tokens=220, verbose=False)
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            tried[dom] += 1
            try:
                ok = score(Mapping.parse(out), Mapping.parse(answer))["exact"]
            except (ValueError, json.JSONDecodeError):
                ok = False
            if not ok:
                failed[dom] += 1
                pairs.append({"prompt": prompt, "chosen": answer, "rejected": out[:600], "domain": dom})
        progress.write_text(json.dumps({"done": sum(tried.values()), "total": a.per_domain * len(seen),
                                        "tried": tried, "failed": failed, "seconds": round(time.time() - started)}))
    r.shuffle(pairs)
    cut = max(1, len(pairs) // 10)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", pairs[:cut]), ("train", pairs[cut:])):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for p in rows:
                f.write(json.dumps({k: p[k] for k in ("prompt", "chosen", "rejected")}, ensure_ascii=False) + "\n")
    summary = {"done": sum(tried.values()), "total": a.per_domain * len(seen), "tried": tried, "failed": failed,
               "pairs": len(pairs), "train": len(pairs) - cut, "valid": cut, "seconds": round(time.time() - started)}
    progress.write_text(json.dumps(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
