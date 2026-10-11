#!/usr/bin/env bash
# Round 2: the teacher writes verified tables, then the same fine-tuning runs on the program's tables
# plus the teacher's, and is scored on the same held-out test. Run after preference training has been scored.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${N:-120}"
lm/.venv/bin/python lm/teacher.py --n "$N" > lm/runs/teacher.log 2>&1
mkdir -p lm/data/sft2
cp lm/data/train.jsonl lm/data/sft2/train.jsonl
lm/.venv/bin/python -c "
import json
with open('lm/data/sft2/train.jsonl', 'a') as f:
    for line in open('lm/data/teacher/accepted.jsonl'):
        f.write(json.dumps({'messages': json.loads(line)['messages']}, ensure_ascii=False) + '\n')"
cp lm/data/valid.jsonl lm/data/sft2/valid.jsonl
DATA=lm/data/sft2 ADAPTER=lm/adapters/mapping-teacher ITERS=500 lm/train.sh > lm/runs/train-teacher.log 2>&1
lm/.venv/bin/python lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-teacher --name teacher-sft > lm/runs/teacher-sft-eval.log 2>&1
tail -20 lm/runs/teacher-sft-eval.log
