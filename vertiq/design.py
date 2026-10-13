"""Experiment design: how large a treatment and control group must be to detect an effect, the smallest
effect a given experiment can detect, and what to do next when the evidence is not yet conclusive.

All calculations are for a difference between two proportions (for example, the share of customers
who return, or of machines that fail), using a two-sided test at significance `alpha` with the stated
`power`. They use the normal approximation, which is reliable when each group expects at least ten
events.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_Z = NormalDist()


def sample_size(base_rate: float, effect: float, *, alpha: float = 0.05, power: float = 0.8,
                control_share: float = 0.2) -> dict:
    """Group sizes needed to detect a change of `effect` (absolute, e.g. -0.05 for five points fewer)
    from `base_rate`, with `control_share` of entities held out as the control group."""
    p0, p1 = base_rate, base_rate + effect
    if not (0 < p0 < 1 and 0 < p1 < 1) or effect == 0 or not (0 < control_share < 1):
        raise ValueError("need 0 < base_rate, base_rate + effect < 1, a non-zero effect and 0 < control_share < 1")
    k = (1 - control_share) / control_share                 # treated per control
    za, zb = _Z.inv_cdf(1 - alpha / 2), _Z.inv_cdf(power)
    pbar = (p0 + k * p1) / (1 + k)
    n_control = (za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) / k)) ** 2 / effect ** 2
    n_control = math.ceil(n_control)
    n_treated = math.ceil(n_control * k)
    return {"control": n_control, "treated": n_treated, "total": n_control + n_treated,
            "base_rate": base_rate, "effect": effect, "alpha": alpha, "power": power}


def detectable_effect(n_treated: int, n_control: int, base_rate: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest absolute effect these group sizes detect with the stated power (normal approximation)."""
    if n_treated < 1 or n_control < 1 or not 0 < base_rate < 1:
        raise ValueError("need positive group sizes and 0 < base_rate < 1")
    za, zb = _Z.inv_cdf(1 - alpha / 2), _Z.inv_cdf(power)
    se = math.sqrt(base_rate * (1 - base_rate) * (1 / n_treated + 1 / n_control))
    return round((za + zb) * se, 4)


def next_step(average_effect: dict, treated: int, control: int, *, alpha: float = 0.05, power: float = 0.8,
              target_effect: float | None = None) -> dict:
    """What to do with an observed effect ({"effect", "p_value", "control_rate"}, as reported by
    tasks.predict_uplift): act on it, or run a larger experiment, with the sizes needed."""
    eff, p, base = average_effect["effect"], average_effect["p_value"], average_effect["control_rate"]
    mde = detectable_effect(treated, control, base, alpha=alpha, power=power) if 0 < base < 1 else None
    if p < alpha:
        return {"decision": "effect detected", "effect": eff, "p_value": p, "detectable_effect": mde}
    want = target_effect if target_effect is not None else (eff if abs(eff) >= 0.005 else (mde or 0.05))
    out = {"decision": "run a larger experiment", "effect": eff, "p_value": p, "detectable_effect": mde}
    try:
        out["needed"] = sample_size(base, want, alpha=alpha, power=power, control_share=control / max(1, treated + control))
    except ValueError:
        pass
    return out
