"""What is known about a customer (or account) at a moment in time: features built ONLY from events
before that moment, so a model trained on the past is tested the way it is used (no peeking).

An event is anything with `occurred_at` and `amount` (a bill, an order, an invoice). Events with a
negative amount (refunds) are not visits. A check-in channel (an event that proves presence but is
not a sale) counts as a visit only on a day with no sale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median, pstdev


@dataclass(frozen=True)
class Visit:
    at: datetime
    amount: float


def visits_from(events, checkin_channel: str | tuple[str, ...] | None = None) -> list[Visit]:
    """One customer's visits, oldest first. A check-in (an event with a truthy `is_checkin`, or one on
    a `checkin_channel`, one name or several) proves presence but isn't a sale: it counts as a visit
    only on a day with no sale and at most once that day, so a check-in beside a bill is not a second visit."""
    chans = {checkin_channel} if isinstance(checkin_channel, str) else set(checkin_channel or ())
    is_in = lambda e: bool(getattr(e, "is_checkin", False)) or getattr(e, "channel", None) in chans   # noqa: E731
    sales = [e for e in events if float(e.amount or 0) >= 0 and not is_in(e)]
    sale_days = {e.occurred_at.date() for e in sales}
    lone = {}                                       # at most one check-in visit a day (a rescan isn't a revisit)
    for e in sorted((e for e in events if is_in(e) and e.occurred_at.date() not in sale_days), key=lambda e: e.occurred_at):
        lone.setdefault(e.occurred_at.date(), e)
    lone = list(lone.values())
    return sorted((Visit(e.occurred_at, float(e.amount or 0)) for e in sales + lone), key=lambda v: v.at)


def cohort_cadence(histories: dict[str, list[Visit]], cutoff: datetime, fallback: float) -> float:
    """The business's typical gap between visits, from customers' own median gaps before `cutoff`."""
    meds = []
    for vs in histories.values():
        times = [v.at for v in vs if v.at < cutoff]
        gaps = [(times[i] - times[i - 1]).days for i in range(1, len(times))]
        gaps = [g for g in gaps if g > 0]
        if gaps:
            meds.append(median(gaps))
    return float(median(meds)) if meds else float(fallback)


def snapshot(visits: list[Visit], cutoff: datetime, cadence: float, sector: str = "",
             biz_size: int = 0) -> dict | None:
    """Features for one customer at `cutoff`, or None if they hadn't visited yet."""
    past = [v for v in visits if v.at < cutoff]
    if not past:
        return None
    times = [v.at for v in past]
    recency = max(0, (cutoff - times[-1]).days)
    tenure = max(0, (cutoff - times[0]).days)
    freq = len(past)
    gaps = [g for g in ((times[i] - times[i - 1]).days for i in range(1, len(times))) if g > 0]
    personal = median(gaps) if gaps else None
    w = len(gaps) / (len(gaps) + 5) if gaps else 0.0        # Bayesian shrink toward the business's cadence
    expected = max(1.0, w * (personal or cadence) + (1 - w) * cadence)
    v90 = sum(1 for t in times if t >= cutoff - timedelta(days=90))
    v_prev = sum(1 for t in times if cutoff - timedelta(days=180) <= t < cutoff - timedelta(days=90))
    amounts = [v.amount for v in past if v.amount > 0]
    recent, older = amounts[-3:], amounts[:-3]
    spend_trend = (math.log((sum(recent) / len(recent)) / (sum(older) / len(older)))
                   if recent and older and sum(older) > 0 else 0.0)
    f = {
        "log_recency": math.log1p(recency),
        "log_freq": math.log1p(freq),
        "one_visit": 1.0 if freq == 1 else 0.0,
        "log_tenure": math.log1p(tenure),
        "log_due": math.log1p(recency / expected),
        "rel_recency": min(10.0, recency / max(1.0, cadence)),
        "gap_cv": (pstdev(gaps) / (sum(gaps) / len(gaps))) if len(gaps) >= 2 else 0.0,
        "trend": (v90 - v_prev) / (v90 + v_prev + 1),
        "log_avg_bill": math.log1p(sum(amounts) / len(amounts)) if amounts else 0.0,
        "spend_trend": max(-2.0, min(2.0, spend_trend)),
        "weekend_share": sum(1 for t in times if t.weekday() >= 5) / freq,
        "log_biz_size": math.log1p(biz_size),
    }
    if sector:
        f[f"sector:{sector.lower()}"] = 1.0
    return f


def facts(visits: list[Visit], cutoff: datetime, cadence: float) -> dict:
    """The plain numbers behind a customer's features, for explaining a score in words."""
    past = [v for v in visits if v.at < cutoff]
    times = [v.at for v in past]
    gaps = [g for g in ((times[i] - times[i - 1]).days for i in range(1, len(times))) if g > 0]
    usual = median(gaps) if gaps else cadence
    return {"visits": len(past), "recency_days": max(0, (cutoff - times[-1]).days) if times else None,
            "usual_gap_days": round(usual, 1),
            "visits_90": sum(1 for t in times if t >= cutoff - timedelta(days=90)),
            "visits_prev_90": sum(1 for t in times if cutoff - timedelta(days=180) <= t < cutoff - timedelta(days=90))}


def returned(visits: list[Visit], start: datetime, days: int) -> int:
    """1 if they visited in [start, start + days), else 0: the label the models learn."""
    end = start + timedelta(days=days)
    return 1 if any(start <= v.at < end for v in visits) else 0


def reference_now(histories: dict[str, list[Visit]]) -> datetime:
    """The latest visit in the business: its "today" (a POS may sync late)."""
    last = [vs[-1].at for vs in histories.values() if vs]
    return max(last) if last else datetime.utcnow()
