# Vertiq LM

Vertiq LM's first job is mapping an unfamiliar table to Vertiq's contract: which column is the entity, the time, the
event type, the value and the group, which columns to keep, and the date format. It is an open model fine-tuned on a
Mac, checked by the data itself, and punished for its own mistakes.

**Base model:** Qwen 3.5 4B, `mlx-community/Qwen3.5-4B-MLX-4bit` (Apache-2.0).
**Hardware used:** Apple M1 Pro, 16 GB. Everything here runs on Apple Silicon with MLX.

## Setup

```bash
python3.12 -m venv lm/.venv
lm/.venv/bin/pip install mlx-lm==0.32.0 mlx-lm-lora==5.8.6 huggingface_hub
lm/.venv/bin/huggingface-cli download mlx-community/Qwen3.5-4B-MLX-4bit --local-dir lm/models/qwen3.5-4b-4bit
# the optional local teacher (5.6 GB)
lm/.venv/bin/huggingface-cli download mlx-community/Qwen3.5-9B-MLX-4bit --local-dir lm/models/qwen3.5-9b-4bit
```

Models, adapters, data and runs are git-ignored (`lm/.gitignore`). Run every command from the repository root.

## The pipeline

| Step | Command | What it does |
|---|---|---|
| 1. Generate | `lm/.venv/bin/python lm/make_data.py --out lm/data/v2` | 15 domains × 6 naming styles × 5 languages, every date format; 4 domains held out entirely by a seeded random draw |
| 2. Fine-tune | `DATA=lm/data/v2/sft ADAPTER=lm/adapters/mapping-v2 ITERS=1000 lm/train.sh` | LoRA on 4 layers, batch 1 with 4-step gradient accumulation, prompt masked: only answers are learned |
| 3. Score | `lm/.venv/bin/python lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-v2 --data lm/data/v2/test.jsonl --name v2-sft` | raw and repaired, overall, seen and never-seen domains, per domain |
| 4. Collect mistakes | `lm/.venv/bin/python lm/collect_failures.py --model <fused model> --adapter "" --meta lm/data/v2/meta.json --out lm/data/dpo2 --per-domain 60` | fresh tables from trained domains only, the same count per domain; wrong answers become rejected pairs |
| 5. Reward / punish | `MODEL=<fused model> DATA=lm/data/dpo2 ADAPTER=lm/adapters/mapping-pref-v2 lm/pref.sh` | ORPO: rewards the right mapping, punishes the model's own wrong one |
| 6. Teach | `lm/.venv/bin/python lm/teacher.py --n 120` | a stronger open model writes new tables; each is kept only if it loads and the teacher, mapping it blind, agrees with itself |
| All of it | `lm/round3.sh` | steps 1–5 with the teacher's tables, then scores everything; `lm/finish3.sh` resumes ORPO after running out of memory |

`lm/report.py --watch` writes a live page (`lm/runs/report.html`) while training runs.
`python benchmarks/collect_lm.py` turns scored runs into `benchmarks/results/lm.json`.

## Teachers

The teacher chain tries any OpenAI-compatible API teachers you configure (`TEACHER_API_KEY`, `TEACHER_BASE_URL`,
`TEACHER_MODEL`, then `TEACHER2_*` …), and falls back to the local Qwen 3.5 9B. Closed providers whose terms forbid
training on their output (OpenAI, Anthropic, Google) are refused unless `TEACHER_ALLOW_CLOSED=1`. Every accepted example
records which teacher wrote it.

## Memory on a 16 GB Mac

- Batch 2 with 8 LoRA layers ran out of GPU memory; batch 1, 4 layers, gradient checkpointing and 640-token sequences fit.
- DPO keeps a reference copy of the model and ran at about 4 minutes a step before running out of memory; ORPO needs no
  reference and runs at 15–30 s a step.
- Preference training needs most of the machine: close browsers, IDEs and dev servers first.

## What the data repair does at use time

`vertiq.mapping.repair` runs after the model answers: column names are snapped to real columns (never invented), roles are
held to their value types (time holds dates, the entity is not a date, the value is a number, row counters are not
attributes) and the date format is decided by the rows. It moves a role only when the data leaves exactly one candidate,
flags weaker evidence for a person, and never changed a correct mapping in 2,100 checks per seed.
