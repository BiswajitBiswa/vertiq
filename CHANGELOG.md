# Changelog

All notable changes are recorded here. Versions follow semantic versioning; before 1.0 the API may change between minor versions.

## Unreleased

- Learners: gradient-boosted trees (`boosted`, `boosted_regression`) with path-based contributing factors, and `learner="auto"`,
  which chooses on a validation split of the training entities so the held-out score stays unbiased.
- Evaluation gate: entity-bootstrap confidence intervals in every report; `confidence` requires the gain to be clear; `evidence`
  from a sequential test can block a promotion.
- `tasks.predict_uplift`: effects of any action in `Dataset.actions`, with two-model or boosted estimators and cross-fitted evaluation.
- New modules: `survival` (time-to-event), `causal` (matched-cohort ATT), `design` (experiment sizing), `sequential` (SPRT),
  `drift` (PSI and accuracy), `sequences` (lagged patterns), `profiles` (serving). `learn.benjamini_yekutieli`.
- `Task.rows_at` for drift checks; `survival=True` on `predict_event`.
- Benchmarks: seven new planted-truth scenarios. `benchmarks/public/online_retail.py` for the first public-data benchmark.
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
