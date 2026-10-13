"""Time-to-next-event estimation: the probability that an entity's next event of a type happens within
a window, given how long it has been since the last one. Gaps that are still open at the cutoff are
right-censored, so entities that have not returned yet still count as "at risk" rather than being
dropped or treated as lost.

  kaplan_meier(durations, observed)   the survival curve S(t) of the gap length
  within(curve, elapsed, window, scale)   P(next event within window | none for `elapsed` days)
  gaps(events_by_entity, cutoff, type)    the durations and censoring flags for every entity at a cutoff
  rhythm(events, cutoff, type, cohort)    an entity's own typical gap, shrunk toward the cohort's

`scale` rescales time by an entity's rhythm relative to the cohort (an accelerated-failure-time
assumption), so one cohort curve serves both fast and slow entities. The estimator assumes censoring is
independent of the gap length, which holds when the only censoring is the cutoff itself.
"""
from __future__ import annotations

from datetime import datetime
from statistics import median


def kaplan_meier(durations: list[float], observed: list[bool]) -> list[tuple[float, float]]:
    """Survival curve as [(t, S(t))] steps, starting at (0, 1)."""
    pts, s, at_risk = [(0.0, 1.0)], 1.0, len(durations)
    counts: dict[float, list[int]] = {}
    for d, o in zip(durations, observed):
        c = counts.setdefault(d, [0, 0])
        c[0] += 1
        c[1] += int(bool(o))
    for t in sorted(counts):
        total, events = counts[t]
        if events and at_risk:
            s *= 1 - events / at_risk
            pts.append((t, s))
        at_risk -= total
    return pts


def survival_at(curve: list[tuple[float, float]], t: float) -> float:
    lo, hi = 0, len(curve) - 1                 # last step at or before t
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if curve[mid][0] <= t:
            lo = mid
        else:
            hi = mid - 1
    return curve[lo][1]


def within(curve: list[tuple[float, float]], elapsed: float, window: float, scale: float = 1.0) -> float:
    """P(next event within `window` days | none for `elapsed` days) = 1 - S(e + w) / S(e)."""
    scale = max(scale, 1e-6)
    s0 = survival_at(curve, elapsed / scale)
    return 0.0 if s0 <= 0 else max(0.0, min(1.0, 1 - survival_at(curve, (elapsed + window) / scale) / s0))


def _times(events, cutoff: datetime, type: str | None) -> list[datetime]:
    return sorted(e.at for e in events if e.at < cutoff and (type is None or e.type == type))


def gaps(events_by_entity: dict, cutoff: datetime, type: str | None = None) -> tuple[list[float], list[bool]]:
    """Every completed gap between consecutive events before `cutoff` (observed), plus each entity's
    open gap from its last event to the cutoff (censored)."""
    durations, observed = [], []
    for evs in events_by_entity.values():
        ts = _times(evs, cutoff, type)
        for a, b in zip(ts, ts[1:]):
            durations.append(float((b - a).days))
            observed.append(True)
        if ts:
            durations.append(float((cutoff - ts[-1]).days))
            observed.append(False)
    return durations, observed


def rhythm(events, cutoff: datetime, type: str | None, cohort: float, shrink: int = 5) -> float:
    """An entity's typical gap: its median gap, shrunk toward the cohort's when it has few gaps."""
    ts = _times(events, cutoff, type)
    g = [(b - a).days for a, b in zip(ts, ts[1:]) if (b - a).days > 0]
    if not g:
        return cohort
    w = len(g) / (len(g) + shrink)
    return w * median(g) + (1 - w) * cohort
