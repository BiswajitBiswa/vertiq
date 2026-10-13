#!/usr/bin/env bash
# Finish round 3 after the ORPO step ran out of GPU memory: retry up to 3 times, resuming from the
# last saved adapter, then score. On the v2 prompts, accumulating gradients over 4 examples ran out of
# GPU memory, so this updates after every example (GRAD_ACCUM=1) for 4x the steps at half the learning
# rate, and leaves out the 8% of examples longer than 460 tokens (lm/fit_length.py): peak memory grows
# with length and the longest ones exceeded the GPU's 12.7 GB limit.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=lm/.venv/bin/python
AD=lm/adapters/mapping-pref-v2
[ -f lm/data/dpo2-fit/train.jsonl ] || $PY lm/fit_length.py --data lm/data/dpo2 --out lm/data/dpo2-fit --cap 460
for attempt in 1 2 3; do
  last=$(ls $AD/0*_adapters.safetensors 2>/dev/null | tail -1)
  done_steps=$([ -n "$last" ] && basename "$last" | sed 's/^0*//; s/_adapters.safetensors//' || echo 0)
  left=$(( ${PREF_ITERS:-980} - ${done_steps:-0} ))
  [ "$left" -le 0 ] && break
  echo "attempt $attempt: $left steps left (resume: ${last:-none})"
  if MODEL=lm/models/qwen-mapping-sft-v2 DATA=lm/data/dpo2-fit ADAPTER=$AD ITERS=$left MAXLEN=576 RESUME="$last" VAL_BATCHES="${VAL_BATCHES:-2}" \
       GRAD_ACCUM=1 LR="${LR:-5e-6}" SAVE_EVERY=100 \
       lm/pref.sh >> lm/runs/pref-v2.log 2>&1; then
    break
  fi
  sleep 60
done
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter $AD --data lm/data/v2/test.jsonl --name v2-pref > lm/runs/v2-pref-eval.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter $AD --prompt 2 --name v2-pref-on-v1 > lm/runs/v2-pref-on-v1-eval.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-v2 --prompt 2 --name v2-sft-on-v1 > lm/runs/v2-sft-on-v1-eval.log 2>&1
echo "round 3 done"
