"""Drop preference examples longer than a token cap, so preference training fits in GPU memory.

ORPO's peak memory grows with sequence length (the 248k-word vocabulary makes the output layer large):
on a 16 GB M1 Pro, 459-token examples peaked at 12.3 GB against a 12.7 GB limit, and longer ones failed.
Cutting long examples short would remove the end of the answer, so they are left out instead.

Run:  lm/.venv/bin/python lm/fit_length.py --data lm/data/dpo2 --out lm/data/dpo2-fit --cap 460
"""
import argparse
import json
import os

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=460)
    ap.add_argument("--model", default="lm/models/qwen-mapping-sft-v2")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)

    def length(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        text = lambda x: x if isinstance(x, str) else json.dumps(x)    # noqa: E731
        return max(len(tok(p + text(ex["chosen"])).input_ids), len(tok(p + text(ex["rejected"])).input_ids))

    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "valid"):
        rows = [json.loads(line) for line in open(f"{a.data}/{split}.jsonl")]
        keep = [r for r in rows if length(r) <= a.cap]
        with open(f"{a.out}/{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
        print(f"{split}: kept {len(keep)} of {len(rows)} at {a.cap} tokens")


if __name__ == "__main__":
    main()
