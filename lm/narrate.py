"""Rewrite Vertiq's template descriptions with a local model and measure how often the rewrite is accepted.

The base model is used without the mapping adapter: the adapter adds the table-mapping skill, and the base
model's general writing is what a description needs. The model sees only the template; a rewrite with a
number that is not in it, or longer than it, is rejected and the template is kept (vertiq.narrate.describe).

Run:  lm/.venv/bin/python lm/narrate.py [--model ...] [--adapter lm/adapters/narrate] [--tests lm/data/narrate/test.jsonl]
Prints each template and rewrite, then how many rewrites were accepted and how many of those were near-copies of the
template (an accepted copy adds nothing). --name writes the scored run to lm/runs/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from vertiq import narrate  # noqa: E402
from vertiq.contract import Dataset, Entity, Event  # noqa: E402
from vertiq.tasks import predict_event  # noqa: E402


def examples(n: int) -> list[tuple[str, dict]]:
    """(template, result) pairs from synthetic datasets: gate decisions and individual predictions."""
    from run import NOW
    out = []
    for seed in range(n):
        rnd = random.Random(seed)
        ents, evs = [], []
        for i in range(rnd.choice([150, 400, 900])):
            risk = 0.03 if seed % 3 == 0 else rnd.choice([0.01, 0.05, 0.12])     # every third dataset has no signal
            ents.append(Entity(f"A{i}"))
            d = NOW - timedelta(days=700)
            while d < NOW:
                if rnd.random() < risk:
                    evs.append(Event(f"A{i}", d, "failure"))
                d += timedelta(days=7)
        t = predict_event(Dataset(ents, evs), target="failure", window_days=30)
        r = t.report()
        out.append((narrate.gate(r), r))
        s = t.score(ents[seed].id) if t.model else None
        if s:
            out.append((narrate.prediction(s, target="failure", window_days=30), s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--adapter", default="", help="a narration adapter (lm/adapters/narrate) to score")
    ap.add_argument("--tests", default="", help="held-out descriptions ({\"template\": ...} per line) instead of --n examples")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--name", default="", help="write the scored run to lm/runs/narrate-<name>.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    from mlx_lm import generate, load
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)

    def write(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=text, max_tokens=200, verbose=False)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    import difflib
    templates = ([json.loads(line)["template"] for line in open(a.tests)] if a.tests else [t for t, _r in examples(a.n)])
    accepted = copies = 0
    rows = []
    for template in templates:
        d = narrate.describe(template, generate=write)
        text = d["text"] if d["source"] == "model" else d.get("rejected", "")
        copy = difflib.SequenceMatcher(None, text.lower(), template.lower()).ratio() > 0.85
        ok = d["source"] == "model"
        accepted += ok
        copies += ok and copy
        rows.append({"template": template, "output": text, "accepted": ok, "copy": copy, "problems": d["problems"]})
        if not a.quiet:
            print("TEMPLATE:", template)
            print("MODEL:   ", text if ok else f"{text}  [rejected: {'; '.join(d['problems'])}]")
            print()
    n = len(templates)
    summary = {"n": n, "accepted": accepted, "accepted_rewrites_not_copies": accepted - copies, "copies": copies,
               "model": a.model, "adapter": a.adapter}
    print(json.dumps(summary))
    if a.name:
        Path(ROOT / "lm" / "runs" / f"narrate-{a.name}.json").write_text(json.dumps({"summary": summary, "examples": rows}, indent=2))


if __name__ == "__main__":
    main()
