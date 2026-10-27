# Vertiq LM

Vertiq LM maps an unfamiliar table to Vertiq's data contract. It identifies the entity, timestamp, event type, value and
grouping columns, the attributes to retain and the date format. It is an open-weight model fine-tuned locally, validated
against the source rows, and trained on its own errors through preference optimisation.

**Base model:** Qwen 3.5 4B, `mlx-community/Qwen3.5-4B-MLX-4bit` (Apache-2.0).
**Reference hardware:** Apple M1 Pro with 16 GB of memory. All tooling runs on Apple Silicon with MLX.

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
| 3. Evaluate | `lm/.venv/bin/python lm/evaluate.py --model lm/models/qwen3.5-4b-4bit --adapter lm/adapters/mapping-v2 --data lm/data/v2/test.jsonl --name v2-sft` | scores before and after validation, overall, for seen and unseen domains, and per domain |
| 4. Collect errors | `lm/.venv/bin/python lm/collect_failures.py --model <fused model> --adapter "" --meta lm/data/v2/meta.json --out lm/data/dpo2 --per-domain 60` | new tables from training domains only, an equal number per domain; incorrect outputs become rejected examples |
| 5. Preference training | `MODEL=<fused model> DATA=lm/data/dpo2 ADAPTER=lm/adapters/mapping-pref-v2 lm/pref.sh` | ORPO: raises the likelihood of the correct mapping and lowers that of the model's incorrect output |
| 6. Teacher generation | `lm/.venv/bin/python lm/teacher.py --n 120` | a larger open-weight model generates new tables; each is retained only if it loads and the teacher reproduces the mapping without seeing it |
| Full round | `lm/round3.sh` | steps 1–5 including teacher tables, followed by evaluation; `lm/finish3.sh` resumes ORPO after an out-of-memory error |

### Memory on a 16 GB Mac

Preference training's peak memory grows with sequence length, because the output layer covers a 248k-word vocabulary: a
459-token example peaked at 12.3 GB against the GPU's 12.7 GB limit. `lm/finish3.sh` therefore drops the 8% of preference
examples longer than 460 tokens (`lm/fit_length.py`; cutting them short would remove the end of the answer), updates after
every example rather than accumulating four, and runs through `lm/mlx_limits.py`, which caps MLX's buffer cache. Close
memory-heavy applications before training.

`lm/report.py --watch` writes a live page (`lm/runs/report.html`) while training runs.
`python benchmarks/collect_lm.py` turns scored runs into `benchmarks/results/lm.json`.

## Several Macs: sweeps and split evaluations

`lm/mesh.py` sends work to a GPU mesh: a head node that keeps a job queue, and workers on other Apple Silicon
Macs that run jobs while those Macs are idle. Progress, loss curves and logs show on the head's dashboard
(`http://<head>:9400/dashboard`), including when the only worker is this Mac.

```bash
python3 lm/mesh.py sweep --name v3-sweep --data lm/data/v3 --grid lr=1e-4,2e-4 --grid layers=4,8 --iters 150
python3 lm/mesh.py eval  --name v2-eval --adapter lm/adapters/mapping-v2 --test lm/data/v2/test.jsonl --shards 3
python3 lm/mesh.py collect --name v3-sweep      # results to lm/runs/mesh/<name>/, with a comparison table
```

- A sweep runs one LoRA fine-tuning job per grid point (`lm/mesh_job.py train`), each scored on held-out tables
  afterwards. Examples longer than `--maxlen` tokens are left out, not cut. Each job saves its adapter every 25
  steps, so a job moved to another Mac continues from there.
- An evaluation splits a test set into shards (`lm/evaluate.py --shard i/k`), one job each; `collect` merges them.
- Only the vertiq package, the job scripts and the named data and adapter go into a job's bundle. Vertiq LM's
  training data is generated or public.
- Workers load the base model from Hugging Face (`VERTIQ_BASE`, default `mlx-community/Qwen3.5-4B-MLX-4bit`)
  and need the same environment as above (`--python lm/.venv/bin/python` when starting the worker).
- The head is `MESH_HEAD` (default `localhost:9400`); the token is `MESH_TOKEN` or the head's own token file.

## Rounds that did not ship (October 2026)

Prompt v3 lets wide exports fit the 640-token training length. Each round continued the current model (round 3 + ORPO)
for 300 steps on 990 tables, half of them messy, on an M4 Pro. Exact is after `vertiq.mapping.repair`, as in use.

| Model | Messy tables (114) | Production-like tables (114) | Unseen domains, production-like | Real exports: roles, exact |
|---|---|---|---|---|
| Current: round 3 + ORPO, prompt v2 | 30.7% | **53.5%** | **41.7%** | 83%, 2 of 6 |
| Prompt v3, two values per text column, lr 5e-5 (30 messy tables) | 23.3% | | | 77%, 1 of 6 |
| Prompt v3, three values per text column, lr 1e-4 | **40.4%** | 43.0% | 31.2% | **90%, 3 of 6** |
| Prompt v3, three values with repeat counts, lr 5e-5 (30 messy tables) | 33.3% | 36.0% | 29.2% | |

The compact prompt helps on messy and real exports and costs on cleaner tables, mostly in telling the event type and
site columns apart; repeat counts did not recover it. A model that wins only some checks does not ship. Next to try: one
model trained on both prompts, v2 when the table fits and v3 when it does not.

## Teachers

The teacher chain tries any OpenAI-compatible API teachers you configure (`TEACHER_API_KEY`, `TEACHER_BASE_URL`,
`TEACHER_MODEL`, then `TEACHER2_*` …), and falls back to the local Qwen 3.5 9B. Closed providers whose terms forbid
training on their output (OpenAI, Anthropic, Google) are refused unless `TEACHER_ALLOW_CLOSED=1`. Every accepted example
records which teacher generated it.

## Memory requirements on a 16 GB machine

- Batch 2 with 8 LoRA layers ran out of GPU memory; batch 1, 4 layers, gradient checkpointing and 640-token sequences fit.
  On a 24 GB M4 Pro, 640 tokens still is the longest that fits (704 and 768 ran out of memory): the 248k-word output
  layer makes memory grow with sequence length, so wide tables need a shorter prompt rather than a longer limit.
- DPO keeps a reference copy of the model and ran at about 4 minutes a step before running out of memory; ORPO needs no
  reference and runs at 15–30 s a step.
- Preference training requires most of the available memory. Close browsers, IDEs and development servers before starting.

## Validation at inference

`vertiq.mapping.repair` runs after the model responds. Column names are matched to existing columns and never invented. Roles
are checked against value types: timestamps must contain dates, the entity cannot be a date, the value must be numeric, and row
counters are excluded from attributes. The date format is inferred from the rows. A role is reassigned only when the data leaves
exactly one valid candidate; weaker evidence is flagged for review. In 2,100 checks per run, validation altered no correct mapping.

## Other rounds and checks

| Script | What it does |
|---|---|
| `lm/make_data.py --messy 0.5` | Training tables with real-export clutter: wide tables, several amount columns, a second date, free text, empty and placeholder columns, ISO timestamps |
| `lm/make_data.py --messy 0.5 --prompt 3 --out lm/data/v5` (`--repeats`: v6) | The same tables in prompt v3: one line per column (name, kind, up to three distinct values cut to 14 characters) instead of header, rows and kinds; 82% of the messy tables fit 640 tokens, against 54% (`--text-values 2`: 88%, but more attributes missed) |
| `lm/evaluate_real.py` + `benchmarks/real/score.py` | Vertiq LM on six held-out public exports (`benchmarks/real/fetch.sh`); never trained on |
| `lm/narrate_round.sh` | Plain-language rewrites: a local 9B teacher (`lm/narrate_data.py`), a LoRA on the 4B, scoring on held-out descriptions (`lm/narrate.py`) |
| `lm/fit_length.py`, `lm/mlx_limits.py` | Fit preference training in 16 GB: drop over-long examples, cap MLX's buffer cache |
