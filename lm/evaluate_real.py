"""Run Vertiq LM on the held-out real exports (benchmarks/real/labels.json) and save its raw proposals for
benchmarks/real/score.py. Inference only; the tables are never used for training.

Run:  lm/.venv/bin/python lm/evaluate_real.py --model lm/models/qwen-mapping-sft-v2 --adapter lm/adapters/mapping-pref-v2 --name v2-pref
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lm"))
from make_data import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    from mlx_lm import generate, load
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    labels = json.loads((ROOT / "benchmarks/real/labels.json").read_text())
    outputs = {}
    for name in labels:
        path = ROOT / "lm/data/real" / f"{name}.csv"
        if name.startswith("_") or not path.exists():
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for _, r in zip(range(3), csv.DictReader(f))]
        msgs = [{"role": "user", "content": render(rows)}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        out = generate(model, tok, prompt=prompt, max_tokens=320, verbose=False)
        outputs[name] = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        print(name, outputs[name][:160].replace("\n", " "))
    (ROOT / "lm/runs" / f"real-{a.name}.json").write_text(json.dumps({"name": a.name, "model": a.model, "adapter": a.adapter,
                                                                     "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
