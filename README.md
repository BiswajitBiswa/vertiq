<p align="center"><img src="docs/assets/logo.svg" alt="Vertiq" width="290"></p>

<p align="center"><b>Decision models for any business, trained on its own outcomes.</b><br>
<a href="https://biswajitbiswa.github.io/vertiq/">Docs portal</a> · <a href="https://biswajitbiswa.github.io/vertiq/benchmarks.html">Benchmarks</a> · <a href="https://biswajitbiswa.github.io/vertiq/guides/">Sector guides</a></p>

Vertiq predicts what happens next to a customer, a machine, a patient or an account, measures what actually worked
against a control group, and only ships a model that beats the obvious rule on records it never saw. It is pure Python
with no dependencies, so it runs on a laptop, a server or a till. Vertiq LM, an open 4B model fine-tuned on a Mac, maps
any sector's export to Vertiq's contract.

- **Map once.** Any table becomes entities, events and actions (`vertiq.mapping`, `vertiq.contract`).
- **Ask general questions.** Will it happen within N days, how much will the next one be worth, who does a message bring
  back (`vertiq.tasks`, `uplift`, `affinity`, `insights`, `rules`).
- **Earn its place.** Every model is scored on held-out entities against the baseline and the current model
  (`vertiq.harness`); if it doesn't win, you keep what you had.

## Benchmarks

Synthetic scenarios with a planted truth, 5 seeds each, mean ± sd
(`python benchmarks/run.py`, 28 s on arm64, Python 3.10.10).

| Scenario | Vertiq | Reference |
|---|---|---|
| Customer return | AUC 0.976 ± 0.005, Brier 0.043 ± 0.004 | recency formula: AUC 0.969 ± 0.005, Brier 0.061 ± 0.006 |
| Machine breakdowns | AUC 0.735 ± 0.059; with one plugin 0.767 ± 0.043 | recency: AUC 0.674 ± 0.050 |
| Length of stay | error 1.61 ± 0.06 days | same as last time: 2.18 ± 0.14 days |
| Pure noise | models shipped: 0 of 5 | baseline kept every time |
| Who a message brings back | top-ranked half +23.6 ± 5.9 pts, bottom half -1.8 ± 5.2 pts | planted: +30 pts for the overdue only |
| Offer effect on the bill | -20.2 ± 0.9% | planted: −20% |
| Tested insights | planted pattern found 5/5; false patterns in noise 0 | Benjamini-Hochberg at 10% |
| Mapping repair safety | correct mappings changed: 0 of 2100 per seed | must be 0 |

These prove the machinery finds real signal, ignores noise and refuses to ship a worse model. They don't prove accuracy on
your data; the evaluation gate measures that on your own held-out records every time it trains. Public datasets (Online
Retail II, Hillstrom email uplift, NASA C-MAPSS) are next.

### Vertiq LM: mapping unseen tables

Qwen 3.5 4B (Apache-2.0, 4-bit MLX), LoRA on an Apple M1 Pro with 16 GB. 114 held-out tables over 15 domains; 4 domains
(energy, insurance, manufacturing, retail sales) never appear in training. "Repaired" = after `vertiq.mapping.repair`,
which lets the rows settle column names, value types and the date format.

Original test set:

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
| Qwen 3.5 4B, no training | 57.9% | 3.5% | 8.8% | 4.2% |
| + LoRA fine-tune (round 1) | 66.7% | 20.2% | 30.7% | 12.5% |
| + ORPO on its own mistakes | 78.1% | 28.1% | 38.6% | 22.9% |
| Round 3 fine-tune | 93.9% | 35.1% | 39.5% | 29.2% |

Realistic test set (real audit columns, amounts in local styles):

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
| + ORPO on its own mistakes (round 1 model) | 80.7% | 29.8% | 34.2% | 14.6% |
| Round 3 fine-tune | 91.2% | 41.2% | 43.9% | 31.2% |

## Quick start

```bash
pip install git+https://github.com/BiswajitBiswa/vertiq
```

```python
from vertiq.mapping import Mapping, apply, repair
from vertiq.tasks import predict_event

rows = [...]   # your export as a list of dicts, e.g. list(csv.DictReader(open("work_orders.csv")))
m, notes = repair(Mapping.parse({"entity": "Equipment No", "time": "Work Order Date", "type": "Order Type",
                                 "value": "Cost", "group": "Plant", "attrs": [], "date_format": "%Y-%m-%d"}), rows)
ds, problems = apply(m, rows)
task = predict_event(ds, target="breakdown", window_days=30)
print(task.status, task.report()["metrics"])   # trained, kept (the baseline wins) or learning (too little data)
print(task.score("EQ-602"))                    # the chance, and what raises or lowers it
```

## Sector guides

Vertiq has no sector-specific code. Each guide shows how a sector's records map, which questions apply, the domain
knowledge worth adding as a plugin, and what has been tested. Every guide's example runs in `tests/test_guides.py`.

- [Restaurants and cafés](https://biswajitbiswa.github.io/vertiq/guides/restaurants-cafes.html): Bring regulars back, size offers to each guest's usual bill, and stop promoting what's out of stock.
- [Retail and e-commerce](https://biswajitbiswa.github.io/vertiq/guides/retail-ecommerce.html): Spot shoppers going quiet, recommend the next product from real baskets, and reorder before stock runs out.
- [Salons, gyms and appointment services](https://biswajitbiswa.github.io/vertiq/guides/salons-gyms-services.html): Learn each client's rhythm, catch the ones slipping, and time reminders to when they are due.
- [B2B distribution and invoicing](https://biswajitbiswa.github.io/vertiq/guides/b2b-distribution.html): See which accounts are ordering less, which invoices will be paid late, and what each account buys next.
- [SaaS and subscriptions](https://biswajitbiswa.github.io/vertiq/guides/saas-subscriptions.html): Predict which accounts will cancel or downgrade, and what the next renewal will be worth.
- [Healthcare and clinics](https://biswajitbiswa.github.io/vertiq/guides/healthcare-clinics.html): Flag likely readmissions and missed follow-ups, and estimate the length of the next stay.
- [Manufacturing and maintenance](https://biswajitbiswa.github.io/vertiq/guides/manufacturing-maintenance.html): Predict breakdowns, catch skipped preventive maintenance and parts used beyond the bill of materials.
- [Supply chain and inventory](https://biswajitbiswa.github.io/vertiq/guides/supply-chain-inventory.html): Reorder on time, find slow movers tying up cash, and forecast the next order per item.
- [Finance and portfolio operations](https://biswajitbiswa.github.io/vertiq/guides/finance-portfolios.html): Find dormant clients and accounts that need a call, from their own activity. Not trading signals.
- [Tax and accounting practices](https://biswajitbiswa.github.io/vertiq/guides/tax-accounting.html): See which clients will file late, which notices repeat, and where work is overdue.
- [Insurance](https://biswajitbiswa.github.io/vertiq/guides/insurance.html): Predict renewals and lapses, flag repeat or duplicate claims for review.
- [Logistics and delivery](https://biswajitbiswa.github.io/vertiq/guides/logistics.html): Predict which shipments run late and which customers ship less, from scan and delivery events.
- [Telecom and utilities](https://biswajitbiswa.github.io/vertiq/guides/telecom-utilities.html): Predict churn and unusual usage per subscriber or meter, and size the next bill.
- [Education](https://biswajitbiswa.github.io/vertiq/guides/education.html): See which students are disengaging from attendance and assessments, early enough to help.

## Docs portal

Live at **https://biswajitbiswa.github.io/vertiq/**: architecture, how it learns, Vertiq LM, benchmarks and the sector guides. `docs/` is a static site
built by `python docs/build.py` from the results files, and GitHub Pages serves it from `main`.

## Layout

| Path | What |
|---|---|
| `vertiq/` | the package: contract, mapping, tasks, harness, plugins, rules, churn, uplift, value, affinity, insights, inventory, offers |
| `tests/` | unit tests, the boundary test (no framework, ORM or HTTP imports) and the guide examples |
| `benchmarks/` | `run.py` (planted-truth scenarios), `collect_lm.py`, `results/*.json` |
| `lm/` | Vertiq LM: data generation, LoRA fine-tuning, scoring, preference training, the verified teacher ([lm/README.md](lm/README.md)) |
| `docs/` | the portal and its builder |

## Status

Version 0.1.0.

## Licence

Apache License 2.0, see [LICENSE](LICENSE). The Qwen 3.5 model weights Vertiq LM fine-tunes are not in this repository;
they are downloaded separately and carry their own Apache-2.0 licence.
