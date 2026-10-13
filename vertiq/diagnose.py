"""Why did a number change between two periods? A rate (failures per machine, repeat customers, late
invoices) is split by group, and the change is divided between:

  rate effect  a group's own rate moved (plant B's failure rate rose)
  mix effect   the share of each group moved (more of the fleet is now in plant B, which always fails more)

The symmetric shift-share split is exact: the rate and mix effects of all groups add up to the overall
change. Each group's rate change gets a p-value (two proportions for yes/no outcomes, a normal
approximation for numbers), and the family is filtered with Benjamini-Hochberg, so a group is called a
driver only when its change is unlikely to be chance.
"""
from __future__ import annotations

import math
from datetime import datetime

from .contract import Dataset
from .learn import benjamini_hochberg, two_proportion_p


def _p_numeric(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 1.0 if ma == mb else 0.0
    return math.erfc(abs(ma - mb) / se / math.sqrt(2))


def change(records: list[dict], *, group: str = "group", period: str = "period", outcome: str = "outcome",
           before="before", after="after", q: float = 0.1, min_group: int = 20) -> dict:
    """`records`: one per entity and period, {group, period, outcome}. Returns the overall rates and change,
    and per group: share and rate in each period, rate and mix effects, p-value and whether it is a
    significant driver after false-discovery control."""
    cells: dict[tuple, list[float]] = {}
    for r in records:
        if r[period] in (before, after):
            cells.setdefault((r[group], r[period]), []).append(float(r[outcome]))
    n0 = sum(len(v) for (g, p), v in cells.items() if p == before)
    n1 = sum(len(v) for (g, p), v in cells.items() if p == after)
    if not n0 or not n1:
        return {"status": "insufficient_data", "before": n0, "after": n1}
    binary = all(x in (0.0, 1.0) for v in cells.values() for x in v)
    groups = sorted({g for g, _p in cells})
    rows = []
    for g in groups:
        a, b = cells.get((g, before), []), cells.get((g, after), [])
        s0, s1 = len(a) / n0, len(b) / n1
        r0 = sum(a) / len(a) if a else 0.0
        r1 = sum(b) / len(b) if b else 0.0
        if len(a) >= min_group and len(b) >= min_group:
            p = two_proportion_p(int(sum(b)), len(b), int(sum(a)), len(a)) if binary else _p_numeric(a, b)
        else:
            p = None
        rows.append({"group": g, "n_before": len(a), "n_after": len(b), "share_before": round(s0, 4), "share_after": round(s1, 4),
                     "rate_before": round(r0, 4), "rate_after": round(r1, 4),
                     "rate_effect": (r1 - r0) * (s0 + s1) / 2, "mix_effect": (s1 - s0) * (r0 + r1) / 2, "p_value": p})
    tested = [r for r in rows if r["p_value"] is not None]
    flags = benjamini_hochberg([r["p_value"] for r in tested], q) if tested else []
    for r, ok in zip(tested, flags):
        r["significant"] = bool(ok)
    for r in rows:
        r.setdefault("significant", False)
        r["rate_effect"], r["mix_effect"] = round(r["rate_effect"], 5), round(r["mix_effect"], 5)
        r["p_value"] = round(r["p_value"], 5) if r["p_value"] is not None else None
    rate0 = sum(sum(v) for (g, p), v in cells.items() if p == before) / n0
    rate1 = sum(sum(v) for (g, p), v in cells.items() if p == after) / n1
    rows.sort(key=lambda r: -(abs(r["rate_effect"]) + abs(r["mix_effect"])))
    return {"status": "ok", "rate_before": round(rate0, 4), "rate_after": round(rate1, 4), "change": round(rate1 - rate0, 5),
            "rate_effects": round(sum(r["rate_effect"] for r in rows), 5), "mix_effects": round(sum(r["mix_effect"] for r in rows), 5),
            "groups": rows, "drivers": [r["group"] for r in rows if r["significant"]], "q": q}


def event_records(ds: Dataset, *, target: str, before: tuple[datetime, datetime], after: tuple[datetime, datetime]) -> list[dict]:
    """Records for change(): for each entity active in a period (any event in it, or before it and never
    ending), whether a `target` event happened in that period. Periods are (start, end) pairs."""
    by = ds.by_entity()
    groups = ds.groups()
    out = []
    for label, (lo, hi) in (("before", before), ("after", after)):
        for eid, evs in by.items():
            if not any(e.at < hi for e in evs):
                continue                                   # not yet present
            hit = any(e.type == target and lo <= e.at < hi for e in evs)
            out.append({"group": groups.get(eid, ""), "period": label, "outcome": 1 if hit else 0})
    return out
