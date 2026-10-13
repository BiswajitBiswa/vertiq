# Changelog

All notable changes are recorded here. Versions follow semantic versioning; before 1.0 the API may change between minor versions.

## Unreleased

- The evaluation gate now requires the gain over the baseline to be clear at 90% confidence by default (`confidence=0.9` on
  `predict_event`, `predict_value` and `harness.evaluate`; `confidence=None` restores the previous behaviour). On 25 random
  datasets this cut promotions from 11 to 2.
- `narrate`: plain-language descriptions of predictions, gate decisions and effect estimates, and `describe(text, generate=...)`,
  which accepts a model's rewrite of the template only if every number in it appears in the template and it is no longer.
  `lm/narrate.py` measures this with a local model: Qwen 3.5 4B and 9B added little to the templates, which remain the default.
- Learners: gradient-boosted trees (`boosted`, `boosted_regression`) with path-based contributing factors, and `learner="auto"`,
  which chooses on a validation split of the training entities so the held-out score stays unbiased.
- Evaluation gate: entity-bootstrap confidence intervals in every report; `confidence` requires the gain to be clear; `evidence`
  from a sequential test can block a promotion.
- `tasks.predict_uplift`: effects of any action in `Dataset.actions`, with two-model or boosted estimators and cross-fitted evaluation.
- New modules: `survival` (time-to-event), `causal` (matched-cohort ATT), `design` (experiment sizing), `sequential` (SPRT),
  `drift` (PSI and accuracy), `sequences` (lagged patterns), `profiles` (serving). `learn.benjamini_yekutieli`.
- `Task.rows_at` for drift checks; `survival=True` on `predict_event`.
- Benchmarks: seven new planted-truth scenarios. `benchmarks/public/online_retail.py` for the first public-data benchmark.
- First public-data result: Online Retail II repeat purchase (5,881 customers), `benchmarks/results/public.json`, shown on the
  benchmarks page and in the README.
- The random-data benchmark now scores 25 datasets with and without the confidence check, and the site computes the figure
  from the results instead of stating it.
- `learner="auto"` now chooses by 3-fold cross-validation over the training entities instead of one validation split. On
  24 planted-truth runs from 100 to 6,400 entities, the largest shortfall against the better learner fell from 0.059 to 0.037 AUC;
  on Online Retail II it now picks the boosted model.
- `min_gain` on `predict_event`, `predict_value` and `harness.evaluate`: the margin a candidate must exceed over the
  baseline and the production model. Rejections report the reason.

## 0.1.0 (2026-10-11)

- Data contract (`Entity`, `Event`, `Action`, `Dataset`) and table mapping with `repair()`: names snapped to real columns, roles
  held to their value types, the date format decided by the rows.
- General tasks `predict_event` and `predict_value` with plugins, explanations and the evaluation gate (`vertiq.harness`).
- Customer models: churn, uplift from control groups, bill size and campaign value, affinity, offers, tested insights, inventory.
- Declarative rules: `repeat`, `overdue`, `exceeds`, `too_short`.
- Custom learners need only `fit`, `predict`, `to_dict` and `from_dict`; per-group offsets and reasons are optional.
- Benchmarks: seven planted-truth scenarios over five seeds; every scored Vertiq LM run.
- Vertiq LM tooling: data generation, LoRA fine-tuning on Apple Silicon, scoring, ORPO on the model's own mistakes, a verified teacher.
- Docs portal with 14 sector guides, generated from the results.
