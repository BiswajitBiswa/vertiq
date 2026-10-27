"""One Vertiq LM job, run by a GPU mesh worker inside the job's bundle (see lm/mesh.py, which builds the bundle).

  {python} lm/mesh_job.py train --data data/train --iters 200 --lr 1e-4 --layers 4 --maxlen 768 \
                                --test data/test.jsonl --eval-n 30
  {python} lm/mesh_job.py eval --adapter adapter --test data/test.jsonl --shard 0/3

train: LoRA fine-tuning of the base model on data/train (train.jsonl, valid.jsonl), then scoring on eval-n held-out
       tables. Examples longer than --maxlen tokens are left out rather than cut (cutting removes the answer).
       Training runs through lm/mesh_train.py, which reports loss and memory to the dashboard. Every 25 steps the
       adapter is saved to out/adapter with the step reached in out/progress.json; the worker sends those files
       back if it has to stop, and a resumed job continues from them, its curves counted on from the same step.
eval:  scores an adapter on one shard of a test set (lm/evaluate.py --shard), reporting after every table.

The base model is --base, else VERTIQ_BASE, else mlx-community/Qwen3.5-4B-MLX-4bit (downloaded once per Mac by mlx-lm;
once it is cached, jobs load it without asking Hugging Face, so they also run offline).
Results land in out/ (eval.json, progress.json, adapter/), which lm/mesh.py collect brings back.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = os.environ.get("VERTIQ_BASE", "mlx-community/Qwen3.5-4B-MLX-4bit")
START = ""          # --start-adapter: an adapter to continue training (its settings are read from it)
OUT = Path("out")
SAVE_EVERY = 25


def _offline_if_cached() -> None:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(BASE, local_files_only=True)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    except Exception:                                   # not cached (or a local path): let mlx-lm fetch it
        pass


def mesh(step: int, **metrics) -> None:
    print("MESH " + json.dumps({"step": step, "metrics": metrics}), flush=True)


def fit(src: Path, dst: Path, maxlen: int) -> dict:
    """Copy train/valid to dst, leaving out examples longer than maxlen tokens (chat template included)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    dst.mkdir(parents=True, exist_ok=True)
    kept = {}
    for name in ("train.jsonl", "valid.jsonl"):
        rows = [json.loads(l) for l in open(src / name)]
        keep = []
        for ex in rows:
            try:
                text = tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(ex["messages"], tokenize=False)
            if len(tok(text)["input_ids"]) <= maxlen:
                keep.append(ex)
        (dst / name).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in keep))
        kept[name] = (len(keep), len(rows))
    return kept


def train(a) -> int:
    OUT.mkdir(exist_ok=True)
    _offline_if_cached()
    state_file = OUT / "progress.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"saved": 0}
    done = state["saved"]
    adapter = OUT / "adapter"
    data = Path("work-data")
    kept = fit(Path(a.data), data, a.maxlen)
    print(f"[vertiq] examples within {a.maxlen} tokens: train {kept['train.jsonl'][0]}/{kept['train.jsonl'][1]}, "
          f"valid {kept['valid.jsonl'][0]}/{kept['valid.jsonl'][1]}", flush=True)
    mesh(done, kept_share=round(kept["train.jsonl"][0] / max(1, kept["train.jsonl"][1]), 3))
    if done < a.iters:
        left = a.iters - done
        cmd = [sys.executable, "lm/mesh_train.py", "--model", BASE, "--train", "--data", str(data),
               "--fine-tune-type", "lora", "--num-layers", str(a.layers), "--batch-size", "1",
               "--grad-accumulation-steps", str(a.accum), "--iters", str(left), "--learning-rate", str(a.lr),
               "--max-seq-length", str(a.maxlen), "--mask-prompt", "--grad-checkpoint",
               "--steps-per-report", "5", "--steps-per-eval", str(a.eval_every), "--val-batches", "10",
               "--save-every", str(SAVE_EVERY), "--adapter-path", str(adapter)]
        if START:
            # continue a trained adapter: its own layers, rank and scale, and its weights as the starting point
            ac = json.loads((Path(START) / "adapter_config.json").read_text())
            lp = ac.get("lora_parameters") or {}
            (OUT / "lora.yaml").write_text("lora_parameters:\n" + "".join(f"  {k}: {v}\n" for k, v in lp.items()
                                                                         if k in ("rank", "dropout", "scale")))
            cmd[cmd.index("--num-layers") + 1] = str(ac.get("num_layers", a.layers))
            cmd += ["--config", str(OUT / "lora.yaml")]
        if done and (adapter / "adapters.safetensors").exists():
            cmd += ["--resume-adapter-file", str(adapter / "adapters.safetensors")]
            print(f"[vertiq] resuming at step {done} of {a.iters}", flush=True)
        elif START:
            cmd += ["--resume-adapter-file", str(Path(START) / "adapters.safetensors")]
            print(f"[vertiq] continuing from {START}", flush=True)
        env = dict(os.environ, MESH_STEP_OFFSET=str(done), MESH_SAVE_EVERY=str(SAVE_EVERY),
                   MESH_PROGRESS_FILE=str(state_file), NO_COLOR="1", TERM="dumb")
        code = subprocess.call(cmd, env=env)
        if code != 0:
            return code
        state_file.write_text(json.dumps({"saved": a.iters}))
    for f in adapter.glob("0*_adapters.safetensors"):                # numbered checkpoints stay on the worker
        f.unlink()
    if a.test and a.eval_n:
        return evaluate(str(adapter), a.test, a.eval_n, "", a.iters)
    return 0


def evaluate(adapter: str, test: str, limit: int, shard: str, step0: int = 0) -> int:
    OUT.mkdir(exist_ok=True)
    _offline_if_cached()
    name = f"mesh-{os.environ.get('MESH_JOB_ID', 'local')}"
    cmd = [sys.executable, "lm/evaluate.py", "--model", BASE, "--adapter", adapter, "--data", test, "--name", name, "--progress"]
    if limit:
        cmd += ["--limit", str(limit)]
    if shard:
        cmd += ["--shard", shard]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if line.startswith("MESH ") and step0:                         # after training, eval points sit at the last step
            d = json.loads(line[5:])
            line = "MESH " + json.dumps({"step": step0, "metrics": d["metrics"]}) + "\n"
        print(line, end="", flush=True)
    if proc.wait() != 0:
        return proc.returncode
    report = json.loads((Path("lm/runs") / f"{name}.json").read_text())
    shutil.copy(Path("lm/runs") / f"{name}.json", OUT / "eval.json")
    o = report["overall"]
    mesh(step0 or o["n"], eval_exact=o["exact"], eval_roles=o["roles_correct"], eval_loads=o["loads_data"],
         eval_exact_repaired=o["exact_repaired"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--iters", type=int, default=200)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--maxlen", type=int, default=768)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--test", default="")
    t.add_argument("--eval-n", type=int, default=30)
    e = sub.add_parser("eval")
    e.add_argument("--adapter", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--shard", default="")
    e.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", default="", help="model to start from: a local folder or a Hugging Face id")
    ap.add_argument("--start-adapter", default="", help="a trained adapter on --base to continue (folder with adapters.safetensors)")
    a = ap.parse_args()
    global BASE, START
    if a.base:
        BASE = a.base
    START = a.start_adapter
    if a.cmd == "train":
        return train(a)
    return evaluate(a.adapter, a.test, a.limit, a.shard)


if __name__ == "__main__":
    sys.exit(main())
