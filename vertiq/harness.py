"""The evaluation gate every task passes through: hold out a share of entities completely, train on
the rest, and compare against a simple baseline (and the previous model, if any) on the held-out ones.
A model that doesn't beat the baseline is reported as "kept" (the baseline stays) and never shipped.

  classify  metric: AUC (ranking) and Brier (probability error), plus calibration bins
  regress   metric: mean absolute error in the target's own units, and R²
"""
from __future__ import annotations

import hashlib
import math
import random

from .learn import auc, brier, calibration, r2

TEST_PCT = 20
MIN_ROWS = 200
MIN_CLASS = 20
CONFIDENCE = 0.9          # default: promote only when the gain over the baseline is clear at 90%


def held_out(key: str, pct: int = TEST_PCT) -> bool:
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % 100 < pct


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * (len(xs) - 1) + 0.5)))]


def bootstrap(keys: list[str], pred: list[float], base: list[float], y: list, kind: str,
              draws: int = 200, level: float = 0.9, seed: int = 11) -> dict:
    """Confidence intervals for the model's score and its gain over the baseline, from a bootstrap that
    resamples whole entities (an entity appears at several cutoffs, so rows are not independent).
    Gain is AUC minus the baseline's AUC for classification, and the relative reduction in mean absolute
    error for regression."""
    by: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    ents = list(by)
    rnd = random.Random(seed)
    score, gain = [], []
    for _ in range(draws):
        idx = [i for _e in ents for i in by[rnd.choice(ents)]]
        yy = [y[i] for i in idx]
        if kind == "classify":
            a, b = auc([pred[i] for i in idx], yy), auc([base[i] for i in idx], yy)
            if a is None or b is None:
                continue
            score.append(a)
            gain.append(a - b)
        else:
            mae = lambda p: sum(abs(math.expm1(p[i]) - math.expm1(y[i])) for i in idx) / len(idx)  # noqa: E731
            m, mb = mae(pred), mae(base)
            score.append(m)
            gain.append((mb - m) / mb if mb else 0.0)
    if not score:
        return {}
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {"level": level, "score_ci": [round(_quantile(score, lo), 4), round(_quantile(score, hi), 4)],
            "gain_ci": [round(_quantile(gain, lo), 4), round(_quantile(gain, hi), 4)]}


SELECT_FOLDS = 3


def _fold(key: str, folds: int = SELECT_FOLDS) -> int:
    return int(hashlib.md5((key + "#select").encode(), usedforsecurity=False).hexdigest(), 16) % folds


def _select(rows, y, keys, tr, candidates, kind, fit_kw) -> tuple[type, dict]:
    """Choose among candidate learners by cross-validation over the training entities only (3 folds, split
    by entity), so the held-out entities still give an unbiased score for the learner that is chosen. One
    validation split was too noisy: it sometimes chose a learner several AUC points worse."""
    scores = {c.__name__: [] for c in candidates}
    for f in range(SELECT_FOLDS):
        val = [i for i in tr if _fold(keys[i]) == f]
        fit = [i for i in tr if _fold(keys[i]) != f]
        truth = [y[i] for i in val]
        if not val or not fit or (kind == "classify" and len(set(truth)) < 2):
            continue
        for cls in candidates:
            m = cls().fit([rows[i] for i in fit], [y[i] for i in fit], **fit_kw)
            pred = [m.predict(rows[i]) for i in val]
            if kind == "classify":
                scores[cls.__name__].append(auc(pred, truth) or 0.0)
            else:
                scores[cls.__name__].append(-sum(abs(math.expm1(p) - math.expm1(t)) for p, t in zip(pred, truth)) / len(truth))
    avg = {k: (sum(v) / len(v) if v else float("-inf")) for k, v in scores.items()}
    best = max(candidates, key=lambda c: avg[c.__name__])
    return best, {k: round(v, 4) for k, v in avg.items()}


def _r(x):
    return round(x, 4) if x is not None else None


def evaluate(rows: list[dict], y: list, keys: list[str], groups: list[str], baseline: list[float], learner_cls,
             *, kind: str = "classify", previous=None, min_rows: int = MIN_ROWS, fit_kw: dict | None = None,
             min_gain: float = 0.0, confidence: float | None = CONFIDENCE, evidence: dict | None = None) -> dict:
    """{"status": learning | kept | trained, "metrics", "model" (when trained), "reason" (when kept)}.
    `keys` decide the held-out split (use the entity id, so no entity is in both halves).
    `min_gain`: the margin a candidate must win by. For classification, an absolute AUC margin over the
    baseline and the previous model (0.02 = two points); for regression, a relative reduction in mean
    absolute error (0.05 = 5% lower). 0 means it must simply not be worse.
    Every report carries bootstrap confidence intervals (`intervals`). With `confidence` set (0.9 by
    default), a candidate is promoted only if the lower end of its gain interval over the baseline also
    exceeds `min_gain`: the improvement must be clear, not just observed. On random data this cuts
    promotions from about half of all candidates to under one in ten. `confidence=None` turns it off.
    `evidence`: a sequential test's state (vertiq.sequential.SequentialTest.state()); it is recorded, and
    a decision for H0 (no improvement) blocks promotion. It never forces one."""
    n = len(y)
    fit_kw = fit_kw or {}
    if kind == "classify":
        pos = sum(y)
        if n < min_rows or pos < MIN_CLASS or n - pos < MIN_CLASS:
            return {"status": "learning", "rows": n, "positives": pos, "need_rows": min_rows}
    elif n < min_rows:
        return {"status": "learning", "rows": n, "need_rows": min_rows}
    test = [held_out(k) for k in keys]
    tr = [i for i in range(n) if not test[i]]
    te = [i for i in range(n) if test[i]]
    pick = lambda xs, idx: [xs[i] for i in idx]      # noqa: E731
    selection = None
    if isinstance(learner_cls, (list, tuple)):
        learner_cls, selection = _select(rows, y, keys, tr, list(learner_cls), kind, fit_kw)
    model = learner_cls().fit(pick(rows, tr), pick(y, tr), **fit_kw)
    held_rows, held_y, held_base = pick(rows, te), pick(y, te), pick(baseline, te)
    pred = [model.predict(r) for r in held_rows]
    m = {"rows": n, "held_out": len(te), "entities_held_out": len({keys[i] for i in te}), "learner": learner_cls.__name__}
    if selection:
        m["selection"] = selection
    if kind == "classify":
        m.update(auc=_r(auc(pred, held_y)), brier=_r(brier(pred, held_y)), calibration=calibration(pred, held_y),
                 baseline_auc=_r(auc(held_base, held_y)), rate=round(sum(y) / n, 4))
        m["min_gain"] = min_gain
        reasons = []
        if m["auc"] is None or (m["baseline_auc"] is not None and m["auc"] < m["baseline_auc"] + min_gain):
            reasons.append(f"AUC {m['auc']} does not exceed the baseline's {m['baseline_auc']} by {min_gain}")
        if previous is not None:
            m["previous_auc"] = _r(auc([previous.predict(r) for r in held_rows], held_y))
            if m["auc"] is not None and m["auc"] < (m["previous_auc"] or 0) + (min_gain if min_gain > 0 else -0.005):
                reasons.append(f"AUC {m['auc']} does not exceed the production model's {m['previous_auc']} by {min_gain}")
    else:
        mae = lambda p: sum(abs(math.expm1(a) - math.expm1(b)) for a, b in zip(p, held_y)) / len(held_y)  # noqa: E731
        m.update(mae=_r(mae(pred)), baseline_mae=_r(mae(held_base)), r2=_r(r2(pred, held_y)))
        m["min_gain"] = min_gain
        reasons = []
        if m["mae"] > m["baseline_mae"] * (1 - min_gain):
            reasons.append(f"MAE {m['mae']} is not {min_gain:.0%} below the baseline's {m['baseline_mae']}")
        if previous is not None:
            m["previous_mae"] = _r(mae([previous.predict(r) for r in held_rows]))
            if m["mae"] > m["previous_mae"] * ((1 - min_gain) if min_gain > 0 else 1.005):
                reasons.append(f"MAE {m['mae']} is not {min_gain:.0%} below the production model's {m['previous_mae']}")
    held_keys = pick(keys, te)
    m["intervals"] = bootstrap(held_keys, pred, held_base, held_y, kind, level=confidence or 0.9)
    if confidence and m["intervals"]:
        lo = m["intervals"]["gain_ci"][0]
        if lo <= min_gain:
            reasons.append(f"the gain over the baseline is not clear at {confidence:.0%} confidence "
                           f"(interval lower bound {lo} is not above {min_gain})")
    if evidence:
        m["evidence"] = evidence
        if evidence.get("decision") == "accept_h0":
            reasons.append(f"the sequential test found no improvement after {evidence.get('n')} observations")
    better = not reasons
    if not better:
        return {"status": "kept", "reason": "; ".join(reasons), "metrics": m}
    final = (learner_cls(model.names) if hasattr(model, "names") else learner_cls()).fit(rows, y, **fit_kw)
    if hasattr(final, "fit_offsets"):            # optional: a learned correction per group
        final.fit_offsets(rows, y, groups)
    return {"status": "trained", "metrics": m, "model": final}
