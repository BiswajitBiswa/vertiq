"""What happened: the last period against the one before it, for any dataset.

  summarise(ds, days=30)  per event type: count and total value in each period, the change, and whether
                          the change in count is larger than chance (a Poisson comparison); entities active,
                          new and gone quiet; and the groups whose count changed most for the most common
                          event type, filtered for false discoveries
  narrate.summary(...)    the same as sentences

Counts are compared as rates per period of equal length, so "clear" means the change is unlikely if the
underlying rate had stayed the same (p < 0.05, two-sided).
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

from .contract import Dataset
from .learn import benjamini_hochberg


def _poisson_p(a: int, b: int) -> float:
    """Two-sided p-value that counts a (before) and b (after) over equal periods share one rate."""
    if a + b == 0:
        return 1.0
    z = (b - a) / math.sqrt(a + b)
    return math.erfc(abs(z) / math.sqrt(2))


def _pct(a: float, b: float) -> float | None:
    return round((b - a) / a, 4) if a else None


def summarise(ds: Dataset, *, days: int = 30, now: datetime | None = None, top_groups: int = 3, q: float = 0.1) -> dict:
    now = now or (ds.reference_now() + timedelta(seconds=1))
    lo1, lo0 = now - timedelta(days=days), now - timedelta(days=2 * days)
    cur = [e for e in ds.events if lo1 <= e.at < now]
    prev = [e for e in ds.events if lo0 <= e.at < lo1]
    types = []
    c0, c1 = Counter(e.type for e in prev), Counter(e.type for e in cur)
    for t in sorted(set(c0) | set(c1), key=lambda t: -(c0[t] + c1[t])):
        v0 = sum(float(e.value or 0) for e in prev if e.type == t)
        v1 = sum(float(e.value or 0) for e in cur if e.type == t)
        p = _poisson_p(c0[t], c1[t])
        types.append({"type": t, "before": c0[t], "after": c1[t], "change": _pct(c0[t], c1[t]), "p_value": round(p, 5),
                      "clear": p < 0.05, "value_before": round(v0, 2), "value_after": round(v1, 2), "value_change": _pct(v0, v1)})
    first = {}
    for e in ds.events:
        if e.entity not in first or e.at < first[e.entity]:
            first[e.entity] = e.at
    active1, active0 = {e.entity for e in cur}, {e.entity for e in prev}
    groups = ds.groups()
    by_group = []
    if types and len(set(groups.values())) > 1:
        main = types[0]["type"]
        g0 = Counter(groups.get(e.entity, "") for e in prev if e.type == main)
        g1 = Counter(groups.get(e.entity, "") for e in cur if e.type == main)
        rows = [{"group": g, "type": main, "before": g0[g], "after": g1[g], "change": _pct(g0[g], g1[g]),
                 "p_value": _poisson_p(g0[g], g1[g])} for g in sorted(set(g0) | set(g1))]
        flags = benjamini_hochberg([r["p_value"] for r in rows], q) if rows else []
        for r, ok in zip(rows, flags):
            r["clear"], r["p_value"] = bool(ok), round(r["p_value"], 5)
        rows.sort(key=lambda r: -abs(r["after"] - r["before"]))
        by_group = rows[:top_groups]
    return {"days": days, "from": lo1.isoformat(), "to": now.isoformat(), "types": types,
            "active": len(active1), "active_before": len(active0),
            "new": sum(1 for eid in active1 if first[eid] >= lo1), "gone_quiet": len(active0 - active1), "groups": by_group}
