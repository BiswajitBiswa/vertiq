"""Effect estimation without a randomised control group: matched-cohort average treatment effect on the
treated (ATT). Each treated entity is matched to the most similar untreated entity on pre-treatment
covariates, within a caliper; unmatched entities are reported rather than forced into a poor match, and
covariate balance is reported before and after matching.

The estimate is unbiased only if there are no unobserved confounders and treated entities have
comparable controls. Where a randomised control group exists, prefer tasks.predict_uplift.
"""
from __future__ import annotations

import math
import random


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
