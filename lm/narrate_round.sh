#!/usr/bin/env bash
# Plain-language rewrites: teacher data, LoRA fine-tune of the 4B model, and scoring on held-out descriptions.
#   1. lm/narrate_data.py: the local 9B teacher rewrites synthesised descriptions; only rewrites that pass every
#      vertiq.narrate check and are not near-copies are kept (test descriptions are never rewritten)
#   2. LoRA fine-tune of Qwen 3.5 4B on them (answers only)
#   3. score base 4B and the fine-tuned model on the same held-out descriptions
set -euo pipefail
cd "$(dirname "$0")/.."
PY=lm/.venv/bin/python
D=lm/data/narrate
while pgrep -f "mlx_lm_lora.train|lm/evaluate.py|finish3.sh" >/dev/null; do sleep 60; done     # one GPU job at a time
[ -f $D/train.jsonl ] || $PY lm/narrate_data.py --teacher lm/models/qwen3.5-9b-4bit --n "${N:-500}" --out $D > lm/runs/narrate-data.log 2>&1
$PY lm/mlx_limits.py mlx_lm.lora --model lm/models/qwen3.5-4b-4bit --train --data $D \
  --fine-tune-type lora --num-layers 4 --batch-size 1 --grad-accumulation-steps 4 --iters "${ITERS:-300}" \
  --learning-rate 1e-4 --max-seq-length 512 --mask-prompt --grad-checkpoint \
  --steps-per-report 25 --steps-per-eval 100 --val-batches 10 --save-every 100 \
  --adapter-path lm/adapters/narrate > lm/runs/narrate-train.log 2>&1
$PY lm/narrate.py --model lm/models/qwen3.5-4b-4bit --tests $D/heldout.jsonl --name base --quiet > lm/runs/narrate-base-eval.log 2>&1
$PY lm/narrate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/narrate --tests $D/heldout.jsonl --name tuned --quiet > lm/runs/narrate-tuned-eval.log 2>&1
echo "narrate round done"
