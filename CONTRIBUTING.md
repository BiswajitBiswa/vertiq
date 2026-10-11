# Contributing to Vertiq

Thanks for helping. Vertiq is small on purpose, so a few rules keep it that way.

## Ground rules

- **No dependencies in `vertiq/`.** The core is pure Python. `tests/test_boundary.py` fails on framework, ORM or HTTP imports.
- **No sector-specific code in the core.** Domain knowledge belongs in a plugin (`@plugins.feature`, `@plugins.rule`) or a guide.
- **No look-ahead.** Anything that builds features must use only events before the cutoff.
- **Claims need evidence.** A change that affects model quality should come with a benchmark result or a test with a planted truth.

## Setup

```bash
git clone https://github.com/BiswajitBiswa/vertiq && cd vertiq
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

## Before you open a pull request

```bash
python -m pytest -q                       # all tests, including every sector guide's example
python benchmarks/run.py --seeds 5        # if you changed modelling code; commit results/core.json
python docs/build.py                      # the docs and README are generated; commit the output
```

CI runs the tests on Python 3.10 to 3.12 and fails if `docs/` or `README.md` don't match `docs/build.py`.

## Adding a sector guide

Add an entry to `docs/guides.py`. `tests/test_guides.py` runs its example against generated rows in its own column names and
date format, so the guide can't drift from the code. Mark evidence honestly: `bench`, `map`, `unit` or `none`.

## Commit messages

Say what changed and why, in plain sentences. One logical change per pull request.
