"""Hidden insights: patterns in a business's own data the owner wouldn't see on a dashboard, each
one tested before it is shown.

vertiq looks for many patterns at once, so each candidate carries a p-value and the whole family is
filtered with Benjamini-Hochberg (learn.benjamini_hochberg, 10% false-discovery rate): the more
places vertiq looks, the stronger a pattern must be to be shown. Groups smaller than MIN_GROUP are
never compared. Descriptive facts (who holds the revenue, regulars slowing down) are shown only past
a size that makes them worth acting on.

Each insight: {key, kind, title, detail, action, evidence}. kind is pattern | risk | opportunity.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import timedelta
from statistics import median

from .features import reference_now
from .learn import benjamini_hochberg, two_proportion_p

MIN_GROUP = 20
REGULAR = 3
_DAYS = ("Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays")


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _times(a: float, b: float) -> str:
    return f"{a / b:.1f}×" if b else ""


def _compare(key, groups: dict[str, tuple[int, int]], total: tuple[int, int], make) -> list[dict]:
    """One candidate per group value: its rate vs everyone else's, with a p-value."""
    out = []
    x_all, n_all = total
    for value, (x, n) in groups.items():
        rest_x, rest_n = x_all - x, n_all - n
        if n < MIN_GROUP or rest_n < MIN_GROUP:
            continue
        p = two_proportion_p(x, n, rest_x, rest_n)
        rate, rest = x / n, rest_x / rest_n
        cand = make(value, rate, rest, n)
        if cand:
            cand["evidence"] = {"group": n, "group_rate": round(rate, 3), "others": rest_n,
                                "others_rate": round(rest, 3), "p_value": round(p, 5)}
            cand["_p"] = p
            cand["_effect"] = abs(math.log((rate + 0.01) / (rest + 0.01)))
            cand.setdefault("key", f"{key}:{value}")
            out.append(cand)
    return out


def find(visits: dict, window_days: int, *, first_items: dict | None = None, combos: list[dict] | None = None,
         currency: str = "₹", limit: int = 8) -> dict:
    """Insights for one business. `visits`: {customer id: [Visit]}; `first_items`: {customer id:
    item names on their first bill}; `combos`: vertiq.affinity.analyse(...)["combos"]."""
    hist = visits
    if len(hist) < 2 * MIN_GROUP:
        return {"status": "learning", "customers": len(hist), "insights": []}
    ref = reference_now(hist)
    tested: list[dict] = []
    tested += _first_visit_drivers(hist, ref, int(window_days), first_items or {}, currency)
    tested += _quick_second_visit(hist)
    keep = benjamini_hochberg([c["_p"] for c in tested]) if tested else []
    found = [c for c, k in zip(tested, keep) if k]
    found.sort(key=lambda c: -c["_effect"])
    described = _slowing_regulars(hist, ref, currency) + _concentration(hist) + _quiet_day(hist) + _combos(combos)
    insights = [{k: v for k, v in c.items() if not k.startswith("_")} for c in found + described]
    return {"status": "ok", "customers": len(hist), "tested": len(tested), "shown_tested": len(found),
            "insights": insights[:limit]}


def _first_visit_drivers(hist, ref, window: int, first_items: dict, cur: str = "₹") -> list[dict]:
    """What about someone's first visit predicts whether they come back: the day, the size of the
    bill, and what they ordered."""
    ready = {cid: vs for cid, vs in hist.items() if vs and (ref - vs[0].at).days >= window}
    if len(ready) < 2 * MIN_GROUP:
        return []
    back = {cid: int(len(vs) >= 2) for cid, vs in ready.items()}
    total = (sum(back.values()), len(back))
    out = []
    # weekend vs weekday first visit
    day = {cid: ("weekend" if vs[0].at.weekday() >= 5 else "weekday") for cid, vs in ready.items()}
    out += _compare("first_day", _tally(day, back), total, lambda v, r, o, n: {
        "kind": "pattern", "title": f"{v.capitalize()} first-timers come back {'more' if r > o else 'less'}",
        "detail": f"{_pct(r)} of people whose first visit was on a {v} came back, against {_pct(o)} of the rest.",
        "action": (f"Make your welcome offer stronger for {'weekday' if v == 'weekend' else 'weekend'} first-timers."
                   if r > o else f"Give {v} first-timers a reason to return, like a second-visit offer.")})
    # first bill size, in thirds
    amounts = sorted(vs[0].amount for vs in ready.values() if vs[0].amount > 0)
    if len(amounts) >= 3 * MIN_GROUP:
        lo, hi = amounts[len(amounts) // 3], amounts[2 * len(amounts) // 3]
        band = {cid: ("small" if vs[0].amount <= lo else "large" if vs[0].amount > hi else "middle")
                for cid, vs in ready.items() if vs[0].amount > 0}
        out += _compare("first_bill", _tally(band, back), total, lambda v, r, o, n: {
            "kind": "pattern",
            "title": f"A {v} first bill means {'more' if r > o else 'fewer'} return visits",
            "detail": (f"{_pct(r)} of people with a {v} first bill (" + (f"{cur}{lo:,.0f} or less" if v == "small" else
                       f"over {cur}{hi:,.0f}" if v == "large" else f"{cur}{lo:,.0f} to {cur}{hi:,.0f}") +
                       f") came back, against {_pct(o)} of the rest."),
            "action": ("Your second-visit nudge matters most for small first bills." if v == "small" and r < o else
                       "Upsell on the first visit: bigger first bills come back more." if v == "large" and r > o else
                       "Watch this group's second visit.")})
    # what they ordered first
    with_items = {cid: first_items[cid] for cid in ready if cid in first_items}
    if len(with_items) >= 2 * MIN_GROUP:
        counts = Counter(i for items in with_items.values() for i in items)
        groups = {}
        for item, _n in counts.most_common(15):
            x = sum(back[cid] for cid, items in with_items.items() if item in items)
            groups[item] = (x, sum(1 for items in with_items.values() if item in items))
        sub_total = (sum(back[cid] for cid in with_items), len(with_items))
        out += _compare("first_item", groups, sub_total, lambda v, r, o, n: {
            "kind": "opportunity" if r > o else "risk",
            "title": (f"First-timers who order {v} come back {_times(r, o)} as often" if r > o
                      else f"First-timers who order {v} rarely come back"),
            "detail": f"{_pct(r)} of {n} people whose first bill had {v} came back, against {_pct(o)} of the rest.",
            "action": (f"Suggest {v} to first-timers at the counter." if r > o
                       else f"Check the {v} experience, and follow up with these guests early.")})
    return out


def _quick_second_visit(hist) -> list[dict]:
    """Whether coming back fast after the first visit predicts becoming a regular."""
    two = {cid: vs for cid, vs in hist.items() if len(vs) >= 2}
    if len(two) < 2 * MIN_GROUP:
        return []
    gap = {cid: (vs[1].at - vs[0].at).days for cid, vs in two.items()}
    cut = median(gap.values())
    if cut < 2:
        return []
    fast = {cid: ("fast" if g <= cut else "slow") for cid, g in gap.items()}
    regular = {cid: int(len(vs) >= REGULAR + 1) for cid, vs in two.items()}
    total = (sum(regular.values()), len(regular))
    return [c for c in _compare("second_visit_speed", _tally(fast, regular), total, lambda v, r, o, n: {
        "kind": "opportunity", "title": f"A second visit within {cut:g} days makes a regular",
        "detail": (f"{_pct(r)} of people who came back within {cut:g} days of their first visit became regulars, "
                   f"against {_pct(o)} of those who took longer."),
        "action": f"Send the second-visit nudge before day {max(1, int(cut) - 1)}, not after."} if v == "fast" and r > o else None)]


def _tally(group_of: dict, outcome: dict) -> dict:
    out: dict = {}
    for cid, g in group_of.items():
        x, n = out.get(g, (0, 0))
        out[g] = (x + outcome[cid], n + 1)
    return out


def _slowing_regulars(hist, ref, cur: str = "₹") -> list[dict]:
    """Regulars whose visits halved this quarter: they haven't gone quiet yet, so the formula
    doesn't flag them, but they are on the way."""
    slowing, at_stake = 0, 0.0
    for vs in hist.values():
        if len(vs) < 4:
            continue
        now = sum(1 for v in vs if v.at >= ref - timedelta(days=90))
        before = sum(1 for v in vs if ref - timedelta(days=180) <= v.at < ref - timedelta(days=90))
        if before >= 2 and now <= before / 2 and (ref - vs[-1].at).days < 60:
            slowing += 1
            at_stake += sum(v.amount for v in vs if v.at >= ref - timedelta(days=180)) / 2
    if slowing < 5:
        return []
    return [{"key": "slowing_regulars", "kind": "risk",
             "title": f"{slowing} regulars are coming half as often",
             "detail": (f"{slowing} regulars visited at least twice last quarter and half as often or less this "
                        f"quarter. They still visit, so they don't show as at risk yet. About {cur}{at_stake:,.0f} "
                        "a quarter rides on them."),
             "action": "Thank them personally now, before they go quiet.",
             "evidence": {"customers": slowing, "quarterly_spend": round(at_stake)}}]


def _concentration(hist) -> list[dict]:
    spend = sorted((sum(v.amount for v in vs) for vs in hist.values()), reverse=True)
    total = sum(spend)
    if total <= 0 or len(spend) < 50:
        return []
    top = max(1, len(spend) // 5)
    share = sum(spend[:top]) / total
    if share < 0.6:
        return []
    return [{"key": "concentration", "kind": "risk",
             "title": f"Your top 20% of customers bring {_pct(share)} of revenue",
             "detail": f"{top} customers bring {_pct(share)} of all spend. Losing a few of them costs more than losing many others.",
             "action": "Give your top customers a perk of their own, like a VIP tier.",
             "evidence": {"top_customers": top, "share": round(share, 3)}}]


def _quiet_day(hist) -> list[dict]:
    by_day = Counter(v.at.weekday() for vs in hist.values() for v in vs)
    n = sum(by_day.values())
    if n < 200 or len(by_day) < 6:
        return []
    avg = n / 7
    day, count = min(((d, by_day.get(d, 0)) for d in range(7)), key=lambda kv: kv[1])
    if count > 0.6 * avg:
        return []
    return [{"key": "quiet_day", "kind": "opportunity",
             "title": f"{_DAYS[day]} are your quietest day",
             "detail": f"{_DAYS[day]} get {_pct(count / avg)} of an average day's visits.",
             "action": f"Run a {_DAYS[day][:-1]}-only offer for the regulars who come on other days.",
             "evidence": {"visits": count, "average_day": round(avg)}}]


def _combos(combos: list[dict]) -> list[dict]:
    strong = [c for c in combos or [] if c["lift"] >= 2 and c["bills_together"] >= 10]
    if not strong:
        return []
    c = strong[0]
    x, y = c["items"]
    return [{"key": f"combo:{x}+{y}", "kind": "opportunity",
             "title": f"{x} and {y} sell together",
             "detail": (f"They're on the same bill {c['lift']:.1f}× more often than chance: {c['a_then_b']:g}% of "
                        f"{x} bills also have {y}."),
             "action": f"Make {x} + {y} a combo, or offer {y} to people who order {x}.",
             "evidence": {"bills_together": c["bills_together"], "lift": c["lift"]}}]
