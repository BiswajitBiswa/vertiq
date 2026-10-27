# Changelog

All notable changes are recorded here. Versions follow semantic versioning; before 1.0 the API may change between minor versions.

## 0.2.4 (2026-10-27)

- A mapping whose date format no date can match (a field written twice, such as `%m/%d/%m`, which Vertiq LM can
  propose) is now reported row by row as not matching, instead of stopping the report with a regular-expression error.
- Vertiq LM: prompt v3 (`lm/make_data.py --prompt 3`), one line per column instead of header, rows and column kinds, so
  88% of messy training tables fit 640 tokens (54% before). Three runs continued from the current model on it got more
  roles right on held-out tables (73–82% against 68%) and one did better on the real exports (4 of 6 exact against 2),
  but none beat the current model's 30% exact on held-out tables, so the current model stays.

## 0.2.3 (2026-10-27)

- Sector benchmark (`benchmarks/sectors`): `vertiq-report` on public data from five sectors (retail, food safety, animal
  services, manufacturing, transport), nothing tuned per sector; results in `benchmarks/results/sectors.json`, full reports
  in `benchmarks/sectors/reports/`, and a section on the benchmarks page.
- Several event files about the same entities in one report: `"records": [{"file": ..., "event": ...}, ...]` (a machine's
  error log and failure log), merged on the entity, with an entity table joined once.
- `"target"` may name an outcome as written in a column ("Fail" in a results column); that column becomes the event type
  and is no longer also an attribute.
- Identifiers found by name or by join: a column named as an identifier (`machineID`, `customer_id`, `Customer ID`) holding
  whole numbers is an entity with 20 or more values, so machine 7 or store 12 are not read as counts; the column whose values
  are an entity table's key is the entity. A numeric column named as a code (`license_`, `Invoice No`, `zip_code`) is never
  the value; "number of ..." stays a count.
- Patterns skip comparisons built into the export: an attribute recorded on one event type only says nothing new about type.
- Vertiq LM: sweeps can start from any model (`lm/mesh.py sweep --base ...`), so a round continues from the production model.

## 0.2.2 (2026-10-27)

Not published on PyPI before this release: 0.2.1 (below) ships with it.

- Faster, with identical results (the benchmark suite and a 14,800-bill report compared byte for byte):
  - the report with the input search on a 14,800-bill export: 21 minutes of compute to 70 seconds; benchmark suite
    164 to 70 seconds on 4 cores.
  - autotune's validation trials skip what they never read (bootstrap intervals, segment checks, the final refit),
    share one cache of feature rows, and run their folds in parallel worker processes (VERTIQ_JOBS; serial where
    fork is unavailable).
  - `Logistic`/`Linear` fit and score with the same arithmetic in the same order, without per-value method calls and
    dictionary lookups; weights are bit-identical.
  - `Task.score` computes the cohort cadence and the data's "today" once, not once per entity.
- The report reads item lists written in one cell, searches candidate inputs (outlet, payment, items) under the gate,
  tries return windows of 30, 60 and 90 days, and writes per-entity scores (`"scores"`); entities that may not be
  used for training stay out of every model (`"trainable"`, `"train": false`).

- Segment check in the evaluation gate (`harness.segment_check`): a candidate that beats the baseline on average but is
  clearly worse for a segment of the held-out entities is not promoted, and `narrate.gate` names the segment. Segments
  come from the data, never from a list per sector: entity groups and entity-table labels with 2 to 12 values, and
  length of history (one record, 2 to 5, more). Each segment with at least 30 held-out entities gets its own bootstrap
  interval, at a confidence level corrected for the number of segments checked. On Online Retail II the next-order-value
  model, already kept, is also clearly worse for customers with more than 5 orders; the purchase model passes all 3 checks.

## 0.2.1 (2026-10-14)

- Reports finish on mid-size exports: UCI Online Retail II (525,461 invoice lines, 28,816 invoices, 4,600 products) in about
  a minute, where the report did not finish before.
  - `patterns`: each comparison is a bit count instead of a pass over every record; at most the 40 most common items or item
    groups are features (chosen by frequency, never by outcome, so false-discovery control holds); item-with-item pairs are
    left to `affinity`; value patterns on items say that part of the gap is basket size.
  - `mapping`: fixed-length whole numbers of 4 or more digits count as identifiers when judging fitness, and `repair` moves
    the entity only to a column with many values, so a country or store column never replaces the customer.
  - `tables`: a product code beside the product name is the item again, not an item group.
  - `affinity`: pairs need support on 1 in 1,000 bills in large data, and combos rank by frequency times log2 lift, so
    frequent strong pairs lead instead of the rarest.
- Vertiq LM across several Macs: `lm/mesh.py` submits LoRA sweeps and sharded evaluations to a job-queue head node and
  collects the results; `lm/mesh_job.py` runs on the worker (leaves out over-long examples, saves every 25 steps, resumes
  on another Mac); `lm/mesh_train.py` reports loss, validation loss, speed and peak memory through mlx-lm's training
  callback; `lm/evaluate.py` gains `--shard` and `--progress`.

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
