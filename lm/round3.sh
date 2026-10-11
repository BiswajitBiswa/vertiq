#!/usr/bin/env bash
# Round 3: the model sees what each column holds (v2 prompt), learns from realistic tables plus the
# teacher's verified ones, then a denser reward/punish round on its own mistakes. Every model is scored
# on the same held-out tables, older ones in their own prompt format, so the comparison is fair.
#   1. v2 data (realistic audit columns and amounts; column kinds in the prompt)
#   2. SFT from the base model on v2 + the teacher's verified tables (about one pass over the data)
#   3. score it, and score the best earlier model (ORPO v1) on the same new tables
#   4. collect the new model's mistakes on fresh tables (same count per domain, test domains untouched)
#   5. ORPO on them, then score
set -euo pipefail
cd "$(dirname "$0")/.."
PY=lm/.venv/bin/python
V2=lm/data/v2

while pgrep -f "lm/teacher.py" >/dev/null; do sleep 30; done     # the teacher shares the GPU; let it finish

[ -f $V2/test.jsonl ] || $PY lm/make_data.py --out $V2
mkdir -p $V2/sft
cp $V2/train.jsonl $V2/valid.jsonl $V2/sft/
$PY - <<'EOF'
# the teacher's verified tables, re-rendered in the v2 prompt (its answers are unchanged)
import json, sys
sys.path.insert(0, "lm"); sys.path.insert(0, ".")
from make_data import render
n = 0
with open("lm/data/v2/sft/train.jsonl", "a") as out:
    for line in open("lm/data/teacher/accepted.jsonl"):
        ex = json.loads(line)
        lines = ex["messages"][0]["content"].split("(CSV):\n")[1].split("\n\n")[0].split("\n")
        cols, rows = lines[0].split(","), [x.split(",") for x in lines[1:]]
        if any(len(r) != len(cols) for r in rows):
            continue
        table = [dict(zip(cols, r)) for r in rows]
        out.write(json.dumps({"messages": [{"role": "user", "content": render(table)}, ex["messages"][1]]}, ensure_ascii=False) + "\n")
        n += 1
print("teacher tables added:", n)
EOF

DATA=$V2/sft ADAPTER=lm/adapters/mapping-v2 ITERS="${ITERS:-1000}" MAXLEN=640 lm/train.sh > lm/runs/train-v2.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-v2 --data $V2/test.jsonl --name v2-sft > lm/runs/v2-sft-eval.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft --adapter lm/adapters/mapping-pref --data $V2/test.jsonl --prompt 1 --name pref-on-v2 > lm/runs/pref-on-v2-eval.log 2>&1

[ -d lm/models/qwen-mapping-sft-v2 ] || $PY -m mlx_lm fuse --model lm/models/qwen3.5-4b-4bit --adapter-path lm/adapters/mapping-v2 \
  --save-path lm/models/qwen-mapping-sft-v2 > lm/runs/fuse-v2.log 2>&1
$PY lm/collect_failures.py --model lm/models/qwen-mapping-sft-v2 --adapter "" --meta $V2/meta.json --out lm/data/dpo2 \
  --per-domain "${PER_DOMAIN:-60}" --seed-offset 2 > lm/runs/collect-v2.log 2>&1
cp lm/runs/failures.json lm/runs/failures-v2.json

MODEL=lm/models/qwen-mapping-sft-v2 DATA=lm/data/dpo2 ADAPTER=lm/adapters/mapping-pref-v2 ITERS="${PREF_ITERS:-300}" \
  lm/pref.sh > lm/runs/pref-v2.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter lm/adapters/mapping-pref-v2 --data $V2/test.jsonl --name v2-pref > lm/runs/v2-pref-eval.log 2>&1
# cross-check on the original v1 tables, in the v2 prompt
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter lm/adapters/mapping-pref-v2 --prompt 2 --name v2-pref-on-v1 > lm/runs/v2-pref-on-v1-eval.log 2>&1
echo "round 3 done"
