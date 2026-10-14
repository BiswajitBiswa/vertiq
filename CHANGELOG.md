# Changelog

All notable changes are recorded here. Versions follow semantic versioning; before 1.0 the API may change between minor versions.

## 0.2.0 (2026-10-14)

- Held-out real-export mapping benchmark (`benchmarks/real`): six public exports from six sectors, accepted answers per
  role, never trained on. Vertiq LM 83% of roles, the untrained 4B 63%, the rule-based proposer 57%.
- Entity tables (`tables.entity_table`, `attach_entities`): one row per entity, joined to events by key; fixed labels feed
  pattern search; totals and dates computed at export describe only. A table alone is read as each entity's latest activity.
- Reading exports: format by content (an .xls that is really .xlsx; Excel 97-2003 with xlrd; CSV delimiters), header below
  titles, single line-item tables turned into records, fixed-length id numbers, repeating names as entities, full totals rows.
- Large exports in linear time (a 15,494-row file: from over 30 minutes to under a second for the proposals).
- Plain-language rewrites: a 4B adapter trained on 208 checked rewrites by a local 9B teacher; on 50 held-out descriptions
  27 real rewrites against 17 for the untrained model (`benchmarks/results/narrate.json`).
- `patterns.find`: tested patterns across all features of a dataset (attributes, groups, weekday, part of day, item
  groups, event type) against the value and each other, with false-discovery control; planted patterns found 2 of 2 and
  nothing reported in random data.
- `python -m vertiq.report config.json`: exported tables (CSV, or Excel with openpyxl) to a written report.
- `tables.propose` and `tables.find_join`: a mapping and a line-item join proposed from the values alone, then checked.
- `mapping`: a separate time-of-day column (`clock`), and `attach_items` for line items joined by a shared key.
- Mapping validation for real exports: summary and mostly-empty rows are named and skipped; placeholder ids ("-", "NA",
  a default number on many rows) become one-off entities instead of merging into one; a role whose column is empty, a
  placeholder or the wrong kind moves to the clearly best-fitting column or is flagged; a value column that is 0 where
  another column records the same amount (or that amount times a quantity) gives way to the more complete one. Judgements
  that need volume wait for 30 rows. No correct mapping in the 10,500-mapping check is changed, and Vertiq LM scores are
  unchanged.
- `narrate.summary`: "In the last day" and "seen for the first time in this data".
- `menu`: candidate features as reviewable specs (count, since, share, sum, group rate, attribute), proposed from the data and
  judged by `autotune`; `menu.parse` rejects anything outside the menu, so a language model can suggest items without code.
- `category.predict_category`: which of several outcomes comes next, gated on macro AUC against each entity's past mix.
- `plan`: action per entity under a budget from measured effects, costs and outcome values.
- `guard`: allow, hold or block for proposed actions, from measured effects, intervals, rule flags and limits.
- `diagnose.change`: rate and mix effects of a change between two periods, with false-discovery control.
- `summary.summarise` and `narrate.summary`: the last period against the previous one, in figures and sentences.
- Vertiq LM round 3 preference training (ORPO on the round 3 model's own mistakes): on the production-like test set, exact
  matches after validation rose from 43.9% to 53.5% (unseen domains 31.2% to 41.7%) and every table loads; on the original test
  set, 63.2%. The README previously described the overall figure as the unseen-domain figure; both are now stated.
- `autotune.search`: a gated search over task settings and candidate features. Each change is scored by 5-fold cross-validation
  on training entities and kept only if the mean improves and most folds agree; the final configuration must beat the baseline
  rule and the untuned configuration on held-out entities. Tasks accept `holdout=` for custom splits.
- `sequential.TwoSampleTest`: a mixture SPRT with an always-valid p-value for comparing two live arms.
- `causal.doubly_robust`: cross-fitted augmented inverse-probability weighting (AIPW) for the average treatment effect.
- Benchmarks for all three; the suite runs in parallel.
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
