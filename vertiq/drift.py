"""Drift: whether the data a model now sees, or its accuracy, has moved away from what it was evaluated on.

  psi(reference, current)            Population Stability Index of one variable, on the reference's
                                     deciles. Common reading: below 0.1 stable, 0.1 to 0.25 a moderate
                                     shift, above 0.25 a major shift.
  feature_drift(ref_rows, cur_rows)  PSI for every feature, worst first, each with a level
  performance(preds, outcomes, ...)  accuracy and calibration on outcomes that have since arrived,
                                     compared with the evaluation the model was promoted on
  check(...)                         a combined verdict: "stable", "watch" or "retrain"

Drift detection recommends; it never replaces a model. A retrained candidate still has to pass the
evaluation gate against the model in service (tasks.predict_event(previous=...)).
"""
from __future__ import annotations

import math

from .learn import auc, brier

WATCH, ALERT = 0.1, 0.25


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference`."""
    if not reference or not current:
        return 0.0
    ref = sorted(reference)
    cuts = sorted({ref[(i * len(ref)) // bins] for i in range(1, bins)})

    def shares(xs):
        counts = [0] * (len(cuts) + 1)
        for x in xs:
            lo, hi = 0, len(cuts)
            while lo < hi:                     # index of the first cut >= x
                mid = (lo + hi) // 2
                if cuts[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            counts[lo] += 1
        return [(c + 0.5) / (len(xs) + 0.5 * len(counts)) for c in counts]
    r, c = shares(reference), shares(current)
    return round(sum((ci - ri) * math.log(ci / ri) for ri, ci in zip(r, c)), 4)


def _level(v: float) -> str:
    return "major" if v >= ALERT else "moderate" if v >= WATCH else "stable"


def feature_drift(ref_rows: list[dict], cur_rows: list[dict], names: list[str] | None = None) -> list[dict]:
    """PSI per feature, worst first."""
    names = names or sorted({k for r in ref_rows for k in r} & {k for r in cur_rows for k in r})
    out = []
    for k in names:
        v = psi([float(r.get(k, 0.0)) for r in ref_rows], [float(r.get(k, 0.0)) for r in cur_rows])
        out.append({"feature": k, "psi": v, "level": _level(v)})
    return sorted(out, key=lambda d: -d["psi"])


def performance(preds: list[float], outcomes: list[int], *, reference_auc: float | None = None,
                tolerance: float = 0.03) -> dict:
    """Accuracy and calibration on outcomes that have arrived since the model was promoted."""
    a, b = auc(preds, outcomes), brier(preds, outcomes)
    gap = (sum(preds) / len(preds) - sum(outcomes) / len(outcomes)) if preds else None
    out = {"n": len(preds), "auc": round(a, 4) if a is not None else None, "brier": round(b, 4) if b is not None else None,
           "calibration_gap": round(gap, 4) if gap is not None else None, "reference_auc": reference_auc}
    out["degraded"] = bool(reference_auc is not None and a is not None and a < reference_auc - tolerance)
    return out


def check(ref_rows: list[dict], cur_rows: list[dict], *, ref_scores: list[float] | None = None,
          cur_scores: list[float] | None = None, perf: dict | None = None) -> dict:
    """Combined verdict. "retrain" when accuracy has degraded or the scores or a feature shifted
    majorly; "watch" on a moderate shift; otherwise "stable"."""
    feats = feature_drift(ref_rows, cur_rows)
    score_psi = psi(ref_scores, cur_scores) if ref_scores and cur_scores else None
    reasons = []
    if perf and perf.get("degraded"):
        reasons.append(f"AUC fell from {perf['reference_auc']} to {perf['auc']}")
    if score_psi is not None and score_psi >= ALERT:
        reasons.append(f"score distribution shifted (PSI {score_psi})")
    major = [f for f in feats if f["level"] == "major"]
    if major:
        reasons.append("major shift in " + ", ".join(f["feature"] for f in major[:3]))
    moderate = [f for f in feats if f["level"] == "moderate"]
    verdict = "retrain" if reasons else ("watch" if moderate or (score_psi or 0) >= WATCH else "stable")
    return {"verdict": verdict, "reasons": reasons, "score_psi": score_psi, "features": feats[:10]}
