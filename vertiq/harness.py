"""The evaluation gate every task passes through: hold out a share of entities completely, train on
the rest, and compare against a simple baseline (and the previous model, if any) on the held-out ones.
A model that doesn't beat the baseline is reported as "kept" (the baseline stays) and never shipped.

  classify  metric: AUC (ranking) and Brier (probability error), plus calibration bins
  regress   metric: mean absolute error in the target's own units, and R²
"""
from __future__ import annotations

import hashlib
import math

from .learn import auc, brier, calibration, r2

TEST_PCT = 20
MIN_ROWS = 200
MIN_CLASS = 20


def held_out(key: str, pct: int = TEST_PCT) -> bool:
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % 100 < pct


def _r(x):
    return round(x, 4) if x is not None else None


def evaluate(rows: list[dict], y: list, keys: list[str], groups: list[str], baseline: list[float], learner_cls,
             *, kind: str = "classify", previous=None, min_rows: int = MIN_ROWS, fit_kw: dict | None = None) -> dict:
    """{"status": learning | kept | trained, "metrics", "model" (when trained), "reason" (when kept)}.
    `keys` decide the held-out split (use the entity id, so no entity is in both halves)."""
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
    model = learner_cls().fit(pick(rows, tr), pick(y, tr), **fit_kw)
    held_rows, held_y, held_base = pick(rows, te), pick(y, te), pick(baseline, te)
    pred = [model.predict(r) for r in held_rows]
    m = {"rows": n, "held_out": len(te), "entities_held_out": len({keys[i] for i in te})}
    if kind == "classify":
        m.update(auc=_r(auc(pred, held_y)), brier=_r(brier(pred, held_y)), calibration=calibration(pred, held_y),
                 baseline_auc=_r(auc(held_base, held_y)), rate=round(sum(y) / n, 4))
        better = m["auc"] is not None and (m["baseline_auc"] is None or m["auc"] >= m["baseline_auc"])
        if previous is not None:
            m["previous_auc"] = _r(auc([previous.predict(r) for r in held_rows], held_y))
            better = better and m["auc"] >= (m["previous_auc"] or 0) - 0.005
    else:
        mae = lambda p: sum(abs(math.expm1(a) - math.expm1(b)) for a, b in zip(p, held_y)) / len(held_y)  # noqa: E731
        m.update(mae=_r(mae(pred)), baseline_mae=_r(mae(held_base)), r2=_r(r2(pred, held_y)))
        better = m["mae"] <= m["baseline_mae"]
        if previous is not None:
            m["previous_mae"] = _r(mae([previous.predict(r) for r in held_rows]))
            better = better and m["mae"] <= m["previous_mae"] * 1.005
    if not better:
        return {"status": "kept", "reason": "the baseline (or the current model) is still better", "metrics": m}
    final = learner_cls(model.names).fit(rows, y, **fit_kw)
    final.fit_offsets(rows, y, groups)
    return {"status": "trained", "metrics": m, "model": final}
