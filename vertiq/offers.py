"""Basket-building offers: the business's own rule, sized per customer from their usual bill.

"Spend 600, get 60 off" works when 600 is a little above what this person usually spends: a stretch
they'll make, not one they'd make anyway (too low) or never reach (too high). The business sets the
reward and the stretch; vertiq writes the per-customer amounts. It never makes up an offer.
"""
from __future__ import annotations

import math

DEFAULT = {"on": False, "reward_pct": 10, "stretch_pct": 20}
LIMITS = (("reward_pct", 5, 30), ("stretch_pct", 10, 50))


def normalise(saved: dict | None) -> dict:
    """A stored rule with defaults filled in and values clamped to their limits."""
    r = saved or {}
    out = dict(DEFAULT)
    out["on"] = bool(r.get("on", False))
    for k, lo, hi in LIMITS:
        try:
            out[k] = max(lo, min(hi, int(r.get(k, DEFAULT[k]))))
        except (TypeError, ValueError):
            pass
    return out


def validate(payload: dict, current: dict) -> dict:
    """The rule after applying `payload` to `current`; ValueError names a value out of range."""
    new = {"on": bool(payload.get("on", current["on"]))}
    for k, lo, hi in LIMITS:
        try:
            v = int(payload.get(k, current[k]))
        except (TypeError, ValueError):
            raise ValueError(f"{k} is a whole number from {lo} to {hi}")
        if not lo <= v <= hi:
            raise ValueError(f"{k} is a whole number from {lo} to {hi}")
        new[k] = v
    return new


def spend_more(avg_bill: float, r: dict, currency: str = "₹", step: int = 50) -> dict | None:
    """{threshold, reward, words} for someone whose usual bill is `avg_bill`; None if too small.
    The threshold rounds up to `step` (in the currency's own units)."""
    if not r.get("on") or not avg_bill or avg_bill < step:
        return None
    threshold = max(2 * step, int(math.ceil(avg_bill * (1 + r["stretch_pct"] / 100) / step) * step))
    reward = max(step // 5, int(round(threshold * r["reward_pct"] / 100 / 10) * 10))
    return {"threshold": threshold, "reward": reward, "words": f"{currency}{reward} off when you spend {currency}{threshold}"}
