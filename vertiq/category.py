"""Which of several outcomes comes next: for each entity, the probability that the first of `types` to
happen within the window is each type, or that none happens (a repair, a replacement or nothing; an
upgrade, a downgrade or a cancellation; a readmission to cardiology, to general medicine or none).

One logistic model per outcome (one against the rest), normalised to probabilities that sum to one,
on the same point-in-time features as tasks.predict_event. The baseline is the entity's own past mix of
these types, smoothed toward the overall mix. The gate is the same idea as for one event: held-out
entities, the macro-average AUC (each outcome against the rest) must beat the baseline's, and with
`confidence` set the lower end of an entity-bootstrap interval for the gain must be above zero.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from . import harness
from .contract import Dataset
from .learn import Logistic, auc

NONE = "none"


def _macro_auc(probs: list[dict], labels: list[str], classes: list[str]) -> float | None:
    scores = []
    for c in classes:
        y = [1 if lab == c else 0 for lab in labels]
        if 0 < sum(y) < len(y):
            a = auc([p[c] for p in probs], y)
            if a is not None:
                scores.append(a)
    return sum(scores) / len(scores) if scores else None


class CategoryTask:
    def __init__(self, builder, window: int, classes: list[str], result: dict):
        self.b, self.window, self.classes, self.result = builder, window, classes, result
        self.models = result.get("models")

    @property
    def status(self) -> str:
        return self.result["status"]

    def report(self) -> dict:
        return {k: v for k, v in self.result.items() if k != "models"} | {"task": "category", "classes": self.classes,
                                                                          "window_days": self.window}

    def _probs(self, row: dict) -> dict:
        raw = {c: m.predict(row) for c, m in self.models.items()}
        total = sum(raw.values()) or 1.0
        return {c: raw[c] / total for c in self.classes}

    def score(self, entity_id: str, at: datetime | None = None) -> dict | None:
        """{"entity", "probabilities": {outcome: p}, "most_likely"} now (or at `at`)."""
        if self.models is None:
            return None
        cut = at or (self.b.ds.reference_now() + timedelta(seconds=1))
        f = self.b.row(entity_id, cut, self.b.cadence(cut, self.window))
        if f is None:
            return None
        p = self._probs(f)
        return {"entity": entity_id, "probabilities": {c: round(v, 4) for c, v in p.items()},
                "most_likely": max(p, key=p.get)}


def predict_category(ds: Dataset, *, types: list[str], window_days: int, history: set | None = None, pack: str | None = None,
                     cutoffs: int = 4, confidence: float | None = harness.CONFIDENCE, min_rows: int = harness.MIN_ROWS,
                     draws: int = 200) -> CategoryTask:
    from .tasks import _Builder
    if len(types) < 2:
        raise ValueError("need at least two outcome types (use tasks.predict_event for one)")
    b = _Builder(ds, types[0], history, pack)
    classes = list(types) + [NONE]
    rows, labels, keys, base = [], [], [], []
    overall = {c: 1.0 for c in classes}
    for cut in b.cutoffs(window_days, cutoffs):
        cad = b.cadence(cut, window_days)
        for eid, evs in b.by.items():
            f = b.row(eid, cut, cad)
            if f is None:
                continue
            nxt = next((e.type for e in sorted(evs, key=lambda e: e.at)
                        if e.type in types and cut <= e.at < cut + timedelta(days=window_days)), NONE)
            past = [e.type for e in evs if e.type in types and e.at < cut]
            rows.append(f)
            labels.append(nxt)
            keys.append(eid)
            base.append(past)
    for lab in labels:
        overall[lab] += 1
    tot = sum(overall.values())
    prior = {c: overall[c] / tot for c in classes}
    base_probs = []
    for past in base:                                    # the entity's own past mix, smoothed toward the overall mix
        k = 3.0
        counts = {c: past.count(c) for c in types}
        n_none = max(0, 1 - len(past))                   # an entity with no such events leans to "none"
        denom = len(past) + n_none + k
        base_probs.append({c: (counts.get(c, 0) + (n_none if c == NONE else 0) + k * prior[c]) / denom for c in classes})
    n = len(labels)
    present = {c for c in labels}
    if n < min_rows or len(present) < 2:
        return CategoryTask(b, window_days, classes, {"status": "learning", "rows": n, "classes_seen": sorted(present), "need_rows": min_rows})
    test = [harness.held_out(k) for k in keys]
    tr = [i for i in range(n) if not test[i]]
    te = [i for i in range(n) if test[i]]

    def fit(idx):
        return {c: Logistic().fit([rows[i] for i in idx], [1 if labels[i] == c else 0 for i in idx]) for c in classes}
    models = fit(tr)
    task = CategoryTask(b, window_days, classes, {"status": "learning", "models": models})
    pred = [task._probs(rows[i]) for i in te]
    held_labels, held_base, held_keys = [labels[i] for i in te], [base_probs[i] for i in te], [keys[i] for i in te]
    m = {"rows": n, "held_out": len(te), "entities_held_out": len(set(held_keys)), "classes": classes,
         "class_rates": {c: round(labels.count(c) / n, 4) for c in classes},
         "macro_auc": _r(_macro_auc(pred, held_labels, classes)), "baseline_macro_auc": _r(_macro_auc(held_base, held_labels, classes)),
         "accuracy": _r(sum(max(p, key=p.get) == lab for p, lab in zip(pred, held_labels)) / max(1, len(te))),
         "baseline_accuracy": _r(sum(max(p, key=p.get) == lab for p, lab in zip(held_base, held_labels)) / max(1, len(te)))}
    by: dict[str, list[int]] = {}
    for j, k in enumerate(held_keys):
        by.setdefault(k, []).append(j)
    ents, rnd, gains = list(by), random.Random(11), []
    for _ in range(draws):
        idx = [j for _e in ents for j in by[rnd.choice(ents)]]
        a = _macro_auc([pred[j] for j in idx], [held_labels[j] for j in idx], classes)
        bb = _macro_auc([held_base[j] for j in idx], [held_labels[j] for j in idx], classes)
        if a is not None and bb is not None:
            gains.append(a - bb)
    gains.sort()
    level = confidence or 0.9
    if gains:
        m["intervals"] = {"level": level, "gain_ci": [round(gains[int((1 - level) / 2 * (len(gains) - 1))], 4),
                                                       round(gains[int((1 + level) / 2 * (len(gains) - 1))], 4)]}
    reasons = []
    if m["macro_auc"] is None or (m["baseline_macro_auc"] is not None and m["macro_auc"] <= m["baseline_macro_auc"]):
        reasons.append(f"macro AUC {m['macro_auc']} does not exceed the baseline's {m['baseline_macro_auc']}")
    if confidence and m.get("intervals") and m["intervals"]["gain_ci"][0] <= 0:
        reasons.append(f"the gain over the baseline is not clear at {confidence:.0%} confidence")
    if reasons:
        return CategoryTask(b, window_days, classes, {"status": "kept", "reason": "; ".join(reasons), "metrics": m})
    return CategoryTask(b, window_days, classes, {"status": "trained", "metrics": m, "models": fit(list(range(n)))})


def _r(x):
    return round(x, 4) if x is not None else None
