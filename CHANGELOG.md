# Changelog

All notable changes are recorded here. Versions follow semantic versioning; before 1.0 the API may change between minor versions.

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
