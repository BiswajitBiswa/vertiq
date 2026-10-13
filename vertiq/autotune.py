"""A gated search over task settings and candidate features: try one change at a time, keep it only if
it improves the validation score, and let the evaluation gate judge the final configuration.

Every trial is scored on validation splits carved from the training entities only (entities the gate
holds out are removed before the search starts), so the gate's held-out score stays unbiased. The final
configuration is trained by the normal task and must beat the baseline rule and the untuned
configuration on the held-out entities, at the usual confidence; otherwise the untuned one is kept.

  result = autotune.search(ds, target="breakdown", window_days=30,
                           options={"learner": ["logistic", "boosted"], "survival": [False, True]},
                           features={"skipped_pm_60": skipped_pm_60, "shift_pattern": shift_pattern})
  result["task"], result["config"], result["trials"]

This is the loop sometimes described as automated research (propose a change, measure, keep or revert),
applied with a held-out gate so that repeated trials cannot overfit the score that decides promotion.
"""
from __future__ import annotations

import math

from . import harness, plugins
from .contract import Dataset
from .tasks import predict_event, predict_value

PACK = "__autotune__"


def _subset(ds: Dataset, keep: set[str]) -> Dataset:
    return Dataset([e for e in ds.entities if e.id in keep], [e for e in ds.events if e.entity in keep])


def _fold(key: str, folds: int) -> int:
    return int(harness.hashlib.md5((key + "#tune").encode(), usedforsecurity=False).hexdigest(), 16) % folds


def _run(ds: Dataset, kind: str, target: str, window_days: int, config: dict, chosen: list[str], features: dict,
         pack: str | None, previous=None, confidence=harness.CONFIDENCE, holdout=None):
    plugins.clear(PACK)
    for fn in plugins.features_for(pack):
        plugins.feature(PACK)(fn)
    for name in chosen:
        plugins.feature(PACK)(features[name])
    kw = {k: v for k, v in config.items() if not (kind == "value" and k == "survival")}
    task_fn = predict_event if kind == "event" else predict_value
    try:
        return task_fn(ds, target=target, window_days=window_days, pack=PACK, previous=previous,
                       confidence=confidence, holdout=holdout, **kw)
    finally:
        plugins.clear(PACK)


def _score(task, kind: str) -> float | None:
    m = task.report().get("metrics") or {}
    if kind == "event":
        return m.get("auc")
    return -m["mae"] if m.get("mae") is not None else None


def search(ds: Dataset, *, target: str, window_days: int, kind: str = "event", pack: str | None = None,
           base: dict | None = None, options: dict | None = None, features: dict | None = None,
           folds: int = 5, min_step: float = 0.003, agree: int | None = None, rounds: int = 2) -> dict:
    """Greedy search from `base` (task keyword arguments, e.g. {"learner": "logistic"}) over `options`
    ({argument: [values]}) and candidate plugin `features` ({name: fn(entity, before, cut)}), plus the
    features already registered under `pack`. The training entities are split into `folds` folds, and
    each fold in turn is the validation set. A change is kept when the mean validation score rises by at
    least `min_step` and the score rises in at least `agree` folds (default: all but one), so a gain
    that comes from one lucky fold is not kept (AUC for event tasks; minus the mean absolute error for
    value tasks). Returns {"task" (the tuned task if the gate
    promoted it over the untuned one, else the untuned task), "tuned_promoted", "tuned", "untuned",
    "config", "features", "validation_score", "trials" (every change tried, with its score)}."""
    base = dict(base or {"learner": "logistic" if kind == "event" else "linear"})
    options, features = options or {}, features or {}
    agree = folds - 1 if agree is None else agree
    train = _subset(ds, {e.id for e in ds.entities if not harness.held_out(e.id)})

    def evaluate(config, chosen):
        scores = [_score(_run(train, kind, target, window_days, config, chosen, features, pack, confidence=None,
                              holdout=lambda key, k=k: _fold(key, folds) == k), kind) for k in range(folds)]
        return None if any(x is None for x in scores) else scores

    def mean(scores):
        return sum(scores) / len(scores) if scores else -math.inf

    config, chosen = dict(base), []
    best_folds = evaluate(config, chosen)
    best = mean(best_folds)
    trials = [{"change": "start", "score": _r(best), "kept": True}]
    for _round in range(rounds):
        improved = False
        moves = [(k, v) for k, vals in options.items() for v in vals]
        moves += [("+feature", n) for n in features] + [("-feature", n) for n in features]
        for key, val in moves:
            if key == "+feature" and val in chosen or key == "-feature" and val not in chosen:
                continue
            if key not in ("+feature", "-feature") and config.get(key) == val:
                continue
            cfg, ch = dict(config), list(chosen)
            if key == "+feature":
                ch.append(val)
            elif key == "-feature":
                ch.remove(val)
            else:
                cfg[key] = val
            f = evaluate(cfg, ch)
            s = mean(f)
            wins = sum(a > b for a, b in zip(f, best_folds)) if f and best_folds else 0
            keep = s >= best + min_step and wins >= agree
            trials.append({"change": f"{key} {val}" if key.endswith("feature") else f"{key}={val}",
                           "score": _r(s), "folds_improved": f"{wins}/{folds}", "kept": keep})
            if keep:
                config, chosen, best, best_folds, improved = cfg, ch, s, f, True
        if not improved:
            break
    untuned = _run(ds, kind, target, window_days, base, [], features, pack)
    final = (_run(ds, kind, target, window_days, config, chosen, features, pack, previous=untuned.model)
             if (config, chosen) != (base, []) else untuned)
    use = final if final is not untuned and final.status == "trained" else untuned
    return {"task": use, "tuned_promoted": use is final and final is not untuned, "tuned": final, "untuned": untuned,
            "config": config, "features": chosen, "validation_score": _r(best), "trials": trials}


def _r(x):
    return round(x, 4) if x not in (None, -math.inf) else None
