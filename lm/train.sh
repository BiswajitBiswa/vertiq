#!/usr/bin/env bash
# Fine-tune Vertiq LM's first job (table -> contract mapping) on this Mac with LoRA.
# Base: Qwen 3.5 4B (Apache-2.0), 4-bit MLX. Only the answers are learned (--mask-prompt).
# Sized for an M1 Pro with 16 GB (batch 2 ran out of GPU memory): batch 1 with 4-step gradient
# accumulation, 4 LoRA layers, gradient checkpointing, sequences up to 640 tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-lm/data/sft}"
ADAPTER="${ADAPTER:-lm/adapters/mapping}"
if [ "$DATA" = "lm/data/sft" ]; then
  mkdir -p lm/data/sft
  cp lm/data/train.jsonl lm/data/valid.jsonl lm/data/sft/
fi
lm/.venv/bin/python -m mlx_lm lora \
  --model lm/models/qwen3.5-4b-4bit --train --data "$DATA" \
  --fine-tune-type lora --num-layers 4 --batch-size 1 --grad-accumulation-steps 4 --iters "${ITERS:-500}" \
  --learning-rate 1e-4 --max-seq-length "${MAXLEN:-640}" --mask-prompt --grad-checkpoint \
  --steps-per-report 25 --steps-per-eval 125 --val-batches 10 --save-every 100 \
  --adapter-path "$ADAPTER"
