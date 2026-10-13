"""Training data for plain-language rewrites: varied template descriptions, rewritten by a local teacher
model, kept only when every check passes.

1. Results are synthesised directly as report dictionaries (gate decisions, predictions, effect estimates)
   across many targets, windows and sizes, and written as template descriptions by vertiq.narrate.
2. The teacher (default: local Qwen 3.5 9B) writes several rewrites of each description with sampling.
3. A rewrite is kept only if vertiq.narrate accepts it (same numbers, each still next to what it
   described, no longer than the template) and it is not a near-copy of the template.
4. Descriptions are split by seed into train, valid and test; the test descriptions are never used in
   training. Each example is the student prompt (vertiq.narrate.PROMPT) and the kept rewrite.

Run:  lm/.venv/bin/python lm/narrate_data.py --teacher lm/models/qwen3.5-9b-4bit --n 600 --out lm/data/narrate
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vertiq import narrate  # noqa: E402

TARGETS = [("failure", "machines"), ("order", "customers"), ("readmission", "patients"), ("renewal", "accounts"),
           ("late payment", "invoices"), ("breakdown", "vehicles"), ("cancellation", "subscribers"), ("return", "orders")]
FACTORS = ["time since the last event", "overdue against its own rhythm", "events this quarter vs last", "number of past events",
           "only one event so far", "average value", "value trend", "irregular timing", "time since the first event",
           "quiet vs the group", "being in group north", "skipped maintenance in the last 60 days", "open complaints"]
ACTIONS = ["the maintenance programme", "the reminder call", "the discount offer", "the follow-up visit", "the new carrier"]

TEACHER_PROMPT = """Rewrite this description of a model result for a manager who is not a statistician. Use your own
sentence structure; do not reuse the original's phrasing. Keep every number exactly as written and attached to the
same thing. Add no numbers, causes, advice or certainty. Keep the entity name if there is one. One short paragraph,
no longer than the original.

Description:
{text}

Rewrite:"""


def synth(rnd: random.Random) -> str:
    """One template description from a random but plausible result."""
    target, things = rnd.choice(TARGETS)
    window = rnd.choice([7, 14, 30, 60, 90])
    kind = rnd.random()
    if kind < 0.45:                                           # gate decision
        held = rnd.choice([36, 44, 77, 120, 168, 412, 1176])
        base = round(rnd.uniform(0.52, 0.8), 4)
        auc = round(min(0.97, max(0.4, base + rnd.uniform(-0.06, 0.12))), 4)
        lo = round(auc - base - rnd.uniform(0.02, 0.09), 4)
        hi = round(auc - base + rnd.uniform(0.02, 0.09), 4)
        trained = lo > 0 and rnd.random() < 0.9
        m = {"auc": auc, "baseline_auc": base, "entities_held_out": held, "min_gain": 0.0,
             "intervals": {"level": 0.9, "gain_ci": [lo, hi]}}
        if rnd.random() < 0.3:                                # value task
            mae = round(rnd.uniform(20, 900), 2)
            m = {"mae": mae, "baseline_mae": round(mae * rnd.uniform(0.9, 1.4), 2), "entities_held_out": held,
                 "min_gain": 0.0, "intervals": {"level": 0.9, "gain_ci": [round(rnd.uniform(-0.05, 0.1), 4), round(rnd.uniform(0.1, 0.3), 4)]}}
            trained = m["intervals"]["gain_ci"][0] > 0
        report = {"status": "trained" if trained else "kept", "task": "event" if "auc" in m else "value",
                  "target": target, "window_days": window, "metrics": m, "reason": "not clear"}
        if rnd.random() < 0.08:
            report = {"status": "learning", "task": "event", "target": target, "window_days": window,
                      "rows": rnd.randint(20, 180), "need_rows": 200}
        return narrate.gate(report)
    if kind < 0.8:                                            # one prediction
        eid = f"{rnd.choice(['M', 'C', 'P', 'A', 'V'])}-{rnd.randint(100, 9999)}"
        fs = rnd.sample(FACTORS, rnd.randint(1, 5))
        k = rnd.randint(0, len(fs))
        score = {"entity": eid, "raises": fs[:k][:3], "lowers": fs[k:][:3]}
        if rnd.random() < 0.75:
            score["chance"] = round(rnd.uniform(0.02, 0.9), 4)
        else:
            score["value"] = round(rnd.uniform(20, 5000), 2)
        return narrate.prediction(score, target=target, window_days=window)
    treated, control = rnd.randint(300, 4000), rnd.randint(100, 1500)
    cr = rnd.uniform(0.05, 0.5)
    eff = rnd.uniform(-0.15, 0.1)
    p = rnd.choice([0.0004, 0.003, 0.02, 0.04, 0.12, 0.35, 0.6])
    nxt = ({"decision": "effect detected"} if p < 0.05 else
           {"decision": "run a larger experiment", "needed": {"effect": round(eff, 4), "treated": rnd.randint(5000, 40000),
                                                              "control": rnd.randint(2000, 15000)}})
    report = {"status": rnd.choice(["trained", "kept"]), "target": target, "window_days": window,
              "metrics": {"average_effect": {"effect": round(eff, 4), "p_value": p, "treated_rate": round(cr + eff, 4),
                                             "control_rate": round(cr, 4)}, "next_step": nxt}}
    return narrate.effect(report, action=rnd.choice(ACTIONS))


def is_copy(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="lm/models/qwen3.5-9b-4bit")
    ap.add_argument("--n", type=int, default=600, help="descriptions to synthesise")
    ap.add_argument("--samples", type=int, default=3, help="teacher rewrites tried per description")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default="lm/data/narrate")
    a = ap.parse_args()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.teacher)
    sampler = make_sampler(temp=a.temp, top_p=0.95)

    def ask(text: str) -> str:
        msgs = [{"role": "user", "content": TEACHER_PROMPT.format(text=text)}]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=prompt, max_tokens=220, sampler=sampler, verbose=False)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "valid", "test")}
    stats = {"descriptions": 0, "tried": 0, "kept": 0, "rejected": {}}
    for i in range(a.n):
        text = synth(random.Random(i))
        split = "test" if i % 10 == 0 else "valid" if i % 10 == 1 else "train"
        stats["descriptions"] += 1
        if split == "test":                                   # test descriptions are kept unrewritten
            files["test"].write(json.dumps({"template": text}, ensure_ascii=False) + "\n")
            continue
        for _ in range(a.samples):
            stats["tried"] += 1
            r = ask(text)
            d = narrate.describe(text, generate=lambda _p, r=r: r)
            why = d["problems"][0].split(" (")[0] if d["problems"] else ("near-copy" if is_copy(r, text) else None)
            if why:
                key = re.sub(r"[-\d.]+", "N", why)
                stats["rejected"][key] = stats["rejected"].get(key, 0) + 1
                continue
            stats["kept"] += 1
            files[split].write(json.dumps({"messages": [{"role": "user", "content": narrate.PROMPT.format(text=text)},
                                                        {"role": "assistant", "content": r}]}, ensure_ascii=False) + "\n")
            break                                             # one accepted rewrite per description
        if i % 50 == 0:
            print(i, stats, flush=True)
    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("done", stats)


if __name__ == "__main__":
    main()
