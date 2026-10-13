"""Rewrite Vertiq's template descriptions with a local model and measure how often the rewrite is accepted.

The base model is used without the mapping adapter: the adapter adds the table-mapping skill, and the base
model's general writing is what a description needs. The model sees only the template; a rewrite with a
number that is not in it, or longer than it, is rejected and the template is kept (vertiq.narrate.describe).

Run:  lm/.venv/bin/python lm/narrate.py [--model mlx-community/Qwen3.5-4B-MLX-4bit] [--n 12]
Prints each template and rewrite, then the acceptance rate. No training; nothing is written.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    from mlx_lm import generate, load
    model, tok = load(a.model)

    def write(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=text, max_tokens=200, verbose=False)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    accepted = 0
    pairs = examples(a.n)
    for template, result in pairs:
        d = narrate.describe(template, generate=write)
        accepted += d["source"] == "model"
        print("TEMPLATE:", template)
        print("MODEL:   ", d["text"] if d["source"] == "model" else f"{d.get('rejected', '')}  [rejected: {'; '.join(d['problems'])}]")
        print()
    print(f"accepted {accepted} of {len(pairs)} rewrites")


if __name__ == "__main__":
    main()
