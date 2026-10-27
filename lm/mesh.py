"""Run Vertiq LM sweeps and evaluations on the GPU mesh (any Mac running `hybrid_gpu_mesh.cli worker`), and bring
the results back. Watch them live at http://<head>/dashboard.

  python3 lm/mesh.py sweep --name v3-sweep --data lm/data/v3 --grid lr=1e-4,2e-4 --grid layers=4,8 --iters 150
  python3 lm/mesh.py eval  --name v2-eval --adapter lm/adapters/mapping-v2 --test lm/data/v2/test.jsonl --shards 3
  python3 lm/mesh.py collect --name v3-sweep

Each command packs one bundle (the vertiq package, the lm scripts a job needs and only the data named) and submits
one job per grid point or shard. Only what is named goes into the bundle: Vertiq LM's training data is generated
or public, and nothing else from the repository leaves this Mac. `collect` downloads each job's out/ folder to
lm/runs/mesh/<name>/<job>/ and prints the comparison; for an evaluation it merges the shards into one score.

The head is MESH_HEAD (default localhost:9400); the token is MESH_TOKEN or the head's own token file.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = os.environ.get("MESH_HEAD", "localhost:9400")
HEAD = HEAD if HEAD.startswith("http") else f"http://{HEAD}"
LM_FILES = ["lm/evaluate.py", "lm/mlx_limits.py", "lm/make_data.py", "lm/mesh_job.py", "lm/mesh_train.py"]
OUT_GLOBS = ["out/eval.json", "out/progress.json", "out/adapter/adapters.safetensors", "out/adapter/adapter_config.json"]


def token() -> str:
    t = os.environ.get("MESH_TOKEN")
    if t:
        return t
    p = Path(os.environ.get("MESH_STATE_DIR", Path.home() / ".hybrid-gpu-mesh")) / "token"
    if p.exists():
        return p.read_text().strip()
    raise SystemExit("no cluster token: set MESH_TOKEN (run `python3 -m hybrid_gpu_mesh.cli token` on the head)")


def call(method: str, path: str, body: bytes | None = None, ctype: str = "application/json") -> bytes:
    req = urllib.request.Request(HEAD + path, data=body, method=method, headers={"X-Mesh-Token": token(), "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def bundle(extra: dict[str, Path]) -> str:
    """Upload a bundle of the vertiq package, the job scripts and `extra` ({path in bundle: local file})."""
    files = {str(p.relative_to(ROOT)): p for p in (ROOT / "vertiq").rglob("*.py")}
    files |= {f: ROOT / f for f in LM_FILES}
    files |= extra
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, p in sorted(files.items()):
            tar.add(p, arcname=name)
    size = buf.tell()
    sha = json.loads(call("POST", "/api/bundles", buf.getvalue(), "application/gzip"))["bundle"]
    print(f"bundle {sha[:12]}: {len(files)} files, {size / 1e6:.1f} MB")
    return sha


def submit(jobs: list[dict]) -> list[str]:
    ids = json.loads(call("POST", "/api/jobs", json.dumps(jobs).encode()))["jobs"]
    for j, jid in zip(jobs, ids):
        print(f"  {jid}  {j['label']}")
    print(f"watch: {HEAD}/dashboard")
    return ids


def cmd_sweep(a) -> None:
    data = ROOT / a.data
    extra = {f"data/train/{n}": data / n for n in ("train.jsonl", "valid.jsonl")}
    test = data / "test.jsonl"
    if a.eval_n and test.exists():
        extra["data/test.jsonl"] = test
    sha = bundle(extra)
    grid = [g.split("=", 1) for g in a.grid]
    keys = [k for k, _ in grid]
    jobs = []
    for values in itertools.product(*[v.split(",") for _, v in grid]):
        p = dict(zip(keys, values))
        lr, layers, maxlen = p.get("lr", "1e-4"), p.get("layers", "4"), p.get("maxlen", str(a.maxlen))
        base = (f"--base {a.base} " if a.base else "") + (f"--start-adapter {a.start_adapter} " if a.start_adapter else "")
        cmd = (f"{{python}} lm/mesh_job.py {base}train --data data/train --iters {a.iters} --lr {lr} --layers {layers} "
               f"--maxlen {maxlen} --eval-n {a.eval_n if 'data/test.jsonl' in extra else 0} --test data/test.jsonl")
        jobs.append({"command": cmd, "bundle": sha, "label": " · ".join(f"{k} {v}" for k, v in p.items()),
                     "kind": "train", "group": a.name, "params": p, "vram_gb": a.vram,
                     "total_steps": a.iters, "artifacts": OUT_GLOBS})
    submit(jobs)


def cmd_eval(a) -> None:
    adapter = ROOT / a.adapter
    extra = {"adapter/adapters.safetensors": adapter / "adapters.safetensors",
             "adapter/adapter_config.json": adapter / "adapter_config.json", "data/test.jsonl": ROOT / a.test}
    sha = bundle(extra)
    n = sum(1 for _ in open(ROOT / a.test))
    jobs = [{"command": f"{{python}} lm/mesh_job.py eval --adapter adapter --test data/test.jsonl --shard {i}/{a.shards}",
             "bundle": sha, "label": f"shard {i + 1} of {a.shards}", "kind": "eval", "group": a.name,
             "params": {"shard": f"{i}/{a.shards}"}, "vram_gb": a.vram, "total_steps": len(range(i, n, a.shards)),
             "artifacts": ["out/eval.json"]} for i in range(a.shards)]
    submit(jobs)


def cmd_collect(a) -> None:
    jobs = [j for j in json.loads(call("GET", "/api/jobs"))["jobs"] if j["group"] == a.name]
    if not jobs:
        raise SystemExit(f"no jobs in group {a.name!r}")
    dest = ROOT / "lm" / "runs" / "mesh" / a.name
    rows, examples = [], []
    for j in jobs:
        d = dest / j["job_id"]
        if j["has_result"]:
            d.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(call("GET", f"/api/jobs/{j['job_id']}/results")), mode="r:gz") as tar:
                tar.extractall(d)
            ev = d / "out" / "eval.json"
            if ev.exists():
                examples += json.loads(ev.read_text()).get("examples", [])
        lat = j["latest"]
        rows.append((j["label"], j["state"], lat.get("val_loss"), lat.get("train_loss"), lat.get("peak_mem_gb"),
                     lat.get("eval_exact"), lat.get("eval_roles"), j["nodes"]))
    f = lambda v: "-" if v is None else f"{v:.3f}"                     # noqa: E731
    print(f"{'job':<28} {'state':<10} {'val loss':>9} {'train':>7} {'mem GB':>7} {'exact':>6} {'roles':>6}  machines")
    for r in rows:
        print(f"{r[0][:28]:<28} {r[1]:<10} {f(r[2]):>9} {f(r[3]):>7} {f(r[4]):>7} {f(r[5]):>6} {f(r[6]):>6}  {', '.join(sorted(set(r[7])))}")
    if examples and all(j["kind"] == "eval" for j in jobs):
        n = len(examples)
        by = defaultdict(list)
        for e in examples:
            by["seen" if e["seen"] else "unseen"].append(e)
        print(f"\nmerged {n} tables: exact {sum(e['exact'] for e in examples) / n:.3f}, roles {sum(e['roles'] for e in examples) / n:.3f}, "
              f"loads {sum(e['loads'] for e in examples) / n:.3f}"
              + "".join(f"; {k} domains exact {sum(e['exact'] for e in v) / len(v):.3f} ({len(v)})" for k, v in sorted(by.items())))
    print(f"\nfiles: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--name", required=True)
    s.add_argument("--data", required=True, help="a folder with train.jsonl, valid.jsonl and optionally test.jsonl")
    s.add_argument("--grid", action="append", default=[], help="name=v1,v2 (lr, layers, maxlen); repeat for more")
    s.add_argument("--iters", type=int, default=150)
    s.add_argument("--maxlen", type=int, default=768)
    s.add_argument("--base", default="", help="model the jobs start from (a path on the workers, or a Hugging Face id); "
                                              "default: the worker's VERTIQ_BASE")
    s.add_argument("--start-adapter", default="", help="an adapter on --base to continue training (a path on the workers); "
                                                       "its layers, rank and scale are used")
    s.add_argument("--eval-n", type=int, default=30)
    s.add_argument("--vram", type=float, default=8.0)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--adapter", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--shards", type=int, default=2)
    e.add_argument("--vram", type=float, default=6.0)
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    a = ap.parse_args()
    {"sweep": cmd_sweep, "eval": cmd_eval, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
