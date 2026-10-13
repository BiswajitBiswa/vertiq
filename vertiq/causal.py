"""Effect estimation without a randomised control group: matched-cohort average treatment effect on the
treated (ATT). Each treated entity is matched to the most similar untreated entity on pre-treatment
covariates, within a caliper; unmatched entities are reported rather than forced into a poor match, and
covariate balance is reported before and after matching.

The estimate is unbiased only if there are no unobserved confounders and treated entities have
comparable controls. Where a randomised control group exists, prefer tasks.predict_uplift.

`doubly_robust` estimates the average treatment effect (ATE) by augmented inverse-probability weighting
(AIPW): a propensity model and two outcome models, cross-fitted so no record is scored by a model that
saw it. The estimate stays consistent if either the propensity model or the outcome models are right,
and its interval comes from the influence function. The same no-unobserved-confounders assumption holds.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist

from .learn import Linear, Logistic


def _sd(records: list[dict], features: list[str]) -> dict[str, float]:
    out = {}
    for f in features:
        vals = [float(r[f]) for r in records]
        mu = sum(vals) / len(vals)
        out[f] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
    return out


def _smd(a: list[dict], b: list[dict], f: str, sd: float) -> float | None:
    if not a or not b:
        return None
    return round((sum(float(r[f]) for r in a) / len(a) - sum(float(r[f]) for r in b) / len(b)) / sd, 4)


def matched_att(records: list[dict], *, treatment: str, outcome: str, features: list[str], caliper: float = 0.25,
                draws: int = 200, seed: int = 13) -> dict:
    """ATT by 1:1 nearest-neighbour matching with replacement on standardised covariates, within
    `caliper` standard deviations per covariate. Returns the estimate, a bootstrap interval over
    matched pairs, matched and unmatched counts, and balance (standardised mean differences)."""
    treated = [r for r in records if r[treatment]]
    control = [r for r in records if not r[treatment]]
    if not treated or not control:
        return {"status": "insufficient_data", "treated": len(treated), "control": len(control)}
    sd = _sd(records, features)
    limit = caliper * math.sqrt(len(features))

    def dist(a, b):
        return math.sqrt(sum(((float(a[f]) - float(b[f])) / sd[f]) ** 2 for f in features))
    pairs, unmatched = [], 0
    for t in treated:
        best = min(control, key=lambda c: dist(t, c))
        if dist(t, best) > limit:
            unmatched += 1
            continue
        pairs.append((t, best))
    if not pairs:
        return {"status": "no_matches", "treated": len(treated), "unmatched": unmatched}
    diffs = [float(t[outcome]) - float(c[outcome]) for t, c in pairs]
    rnd = random.Random(seed)
    boot = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(draws))
    mt, mc = [t for t, _ in pairs], [c for _, c in pairs]
    return {"status": "ok", "att": round(sum(diffs) / len(diffs), 4),
            "interval_90": [round(boot[int(0.05 * (draws - 1))], 4), round(boot[int(0.95 * (draws - 1))], 4)],
            "matched": len(pairs), "unmatched": unmatched, "controls_used": len({id(c) for c in mc}),
            "balance_before": {f: _smd(treated, control, f, sd[f]) for f in features},
            "balance_after": {f: _smd(mt, mc, f, sd[f]) for f in features}}


def doubly_robust(records: list[dict], *, treatment: str, outcome: str, features: list[str], folds: int = 5,
                  clip: float = 0.02, level: float = 0.9, seed: int = 13) -> dict:
    """Average treatment effect by cross-fitted AIPW. Returns the estimate, its interval at `level`, the
    outcome-model-only estimate for comparison, and overlap diagnostics (propensities clipped to
    [`clip`, 1 - `clip`]; many clipped records mean treated and untreated entities are not comparable)."""
    t = [1 if r[treatment] else 0 for r in records]
    y = [float(r[outcome]) for r in records]
    nt = sum(t)
    if nt < 2 * folds or len(t) - nt < 2 * folds:
        return {"status": "insufficient_data", "treated": nt, "control": len(t) - nt}
    rows = [{f: float(r[f]) for f in features} for r in records]
    binary = set(y) <= {0.0, 1.0}
    out_cls = Logistic if binary else Linear
    fit_kw = {} if binary else {"epochs": 30}
    rnd = random.Random(seed)
    fold = [rnd.randrange(folds) for _ in records]
    psi, reg, clipped = [0.0] * len(t), [0.0] * len(t), 0
    for k in range(folds):
        tr = [i for i in range(len(t)) if fold[i] != k]
        te = [i for i in range(len(t)) if fold[i] == k]
        if not te:
            continue
        ps = Logistic().fit([rows[i] for i in tr], [t[i] for i in tr])
        m1 = out_cls().fit([rows[i] for i in tr if t[i]], [y[i] for i in tr if t[i]], **fit_kw)
        m0 = out_cls().fit([rows[i] for i in tr if not t[i]], [y[i] for i in tr if not t[i]], **fit_kw)
        for i in te:
            e = ps.predict(rows[i])
            if e < clip or e > 1 - clip:
                clipped += 1
                e = min(1 - clip, max(clip, e))
            mu1, mu0 = m1.predict(rows[i]), m0.predict(rows[i])
            reg[i] = mu1 - mu0
            psi[i] = mu1 - mu0 + t[i] * (y[i] - mu1) / e - (1 - t[i]) * (y[i] - mu0) / (1 - e)
    n = len(psi)
    ate = sum(psi) / n
    se = math.sqrt(sum((v - ate) ** 2 for v in psi) / (n - 1) / n)
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    return {"status": "ok", "ate": round(ate, 4), "se": round(se, 4), "level": level,
            "interval": [round(ate - z * se, 4), round(ate + z * se, 4)],
            "outcome_model_only": round(sum(reg) / n, 4), "treated": nt, "control": n - nt,
            "clipped_propensities": clipped, "folds": folds}
