#!/usr/bin/env bash
# Finish round 3 after the ORPO step ran out of GPU memory: retry up to 3 times, resuming from the
# last saved adapter (saved every 50 steps), then score. Same settings as round3.sh otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=lm/.venv/bin/python
AD=lm/adapters/mapping-pref-v2
for attempt in 1 2 3; do
  last=$(ls $AD/0*_adapters.safetensors 2>/dev/null | tail -1)
  done_steps=$([ -n "$last" ] && basename "$last" | sed 's/^0*//; s/_adapters.safetensors//' || echo 0)
  left=$(( ${PREF_ITERS:-300} - ${done_steps:-0} ))
  [ "$left" -le 0 ] && break
  echo "attempt $attempt: $left steps left (resume: ${last:-none})"
  if MODEL=lm/models/qwen-mapping-sft-v2 DATA=lm/data/dpo2 ADAPTER=$AD ITERS=$left MAXLEN=576 RESUME="$last" \
       lm/pref.sh >> lm/runs/pref-v2.log 2>&1; then
    break
  fi
  sleep 60
done
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter $AD --data lm/data/v2/test.jsonl --name v2-pref > lm/runs/v2-pref-eval.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen-mapping-sft-v2 --adapter $AD --prompt 2 --name v2-pref-on-v1 > lm/runs/v2-pref-on-v1-eval.log 2>&1
$PY lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-v2 --prompt 2 --name v2-sft-on-v1 > lm/runs/v2-sft-on-v1-eval.log 2>&1
echo "round 3 done"
