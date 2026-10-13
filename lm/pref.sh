#!/usr/bin/env bash
# Reward / punish round: preference training on the fine-tuned model's own mistakes
# (lm/collect_failures.py: chosen = the correct mapping, rejected = what the model wrote).
# ORPO, because DPO keeps a second copy of the model as its reference: on a 16 GB Mac it ran out of
# GPU memory and took ~4 minutes a step. ORPO needs no reference model.
# Starts from the SFT model fused into lm/models/qwen-mapping-sft; kept only if it beats SFT on the
# held-out test (lm/evaluate.py). lm/mlx_limits.py caps MLX's buffer cache so ORPO fits in 16 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
lm/.venv/bin/python lm/mlx_limits.py mlx_lm_lora.train \
  --model "${MODEL:-lm/models/qwen-mapping-sft}" --train --train-mode "${MODE:-orpo}" --data "${DATA:-lm/data/dpo}" \
  --train-type lora --num-layers 4 --batch-size 1 --gradient-accumulation-steps "${GRAD_ACCUM:-4}" \
  --iters "${ITERS:-300}" --learning-rate "${LR:-1e-5}" --beta "${BETA:-0.1}" \
  --max-seq-length "${MAXLEN:-768}" --grad-checkpoint ${RESUME:+--resume-adapter-file "$RESUME"} \
  --steps-per-report 10 --steps-per-eval 50 --val-batches "${VAL_BATCHES:-10}" --save-every "${SAVE_EVERY:-50}" \
  --adapter-path "${ADAPTER:-lm/adapters/mapping-pref}"
