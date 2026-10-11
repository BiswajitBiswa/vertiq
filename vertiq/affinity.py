"""Customer choices from line items: what sells together (combos), what a customer would likely
try next (cross-sell), and items people buy again on a cycle (refills).

  combos       two items on the same bill far more often than chance (lift), for combo offers
  cross_sell   people who buy A tend to buy B too; the people with A and never B are the audience
  next_items   per customer: what they haven't tried that people like them buy
  refills      items bought again on a steady cycle, and who is due now

Lift = how many times more often two things go together than if choices were independent:
1.0 is chance, 2.0 is twice as often. Only pairs seen at least MIN_PAIR times count, so a single
big order never makes a pattern. Plain data in: [(customer id, bill id, when, item name)].
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from statistics import median

MIN_PAIR = 5
MIN_BASKETS = 30
MAX_ITEMS_PER_BILL = 30


def clean_item(name) -> str:
    """An item name as it should be compared: spaces collapsed, placeholders ("item") dropped."""
    n = " ".join(str(name or "").split())[:60]
    return "" if n.lower() in ("", "item") else n


def analyse(lines, min_pair: int = MIN_PAIR) -> dict:
    """Everything affinity knows from [(customer id, bill id, when, item name)]."""
    bills: dict[str, set] = defaultdict(set)
    owner: dict[str, str] = {}
    bought: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))   # cid -> item -> [dates]
    for cid, tid, at, item in lines:
        bills[tid].add(item)
        owner[tid] = cid
        bought[cid][item].append(at)
    n_bills = len(bills)
    result = {"bills": n_bills, "customers": len(bought), "status": "ok", "combos": [], "cross_sell": [],
              "refills": [], "top_items": [], "_lifts": {}, "_bought": {}, "_cycles": {}}
    if n_bills < MIN_BASKETS:
        result["status"] = "learning"
        return result
    item_bills: Counter = Counter()
    pair_bills: Counter = Counter()
    for items in bills.values():
        items = sorted(items)[:MAX_ITEMS_PER_BILL]
        item_bills.update(items)
        pair_bills.update(combinations(items, 2))
    n_cust = len(bought)
    item_cust = Counter(i for items in bought.values() for i in items)
    pair_cust: Counter = Counter()
    for items in bought.values():
        pair_cust.update(combinations(sorted(items)[:MAX_ITEMS_PER_BILL * 3], 2))
    result["top_items"] = [{"item": i, "customers": c, "share": round(100 * c / n_cust, 1)}
                           for i, c in item_cust.most_common(8)]
    result["combos"] = _combos(pair_bills, item_bills, n_bills, min_pair)
    lifts, cross = _cross_sell(pair_cust, item_cust, n_cust, bought, min_pair)
    result["cross_sell"], result["_lifts"] = cross, lifts
    result["_bought"] = {cid: {i: max(ds) for i, ds in items.items()} for cid, items in bought.items()}
    result["refills"], result["_cycles"] = _refills(bought)
    return result


def _combos(pair_bills: Counter, item_bills: Counter, n: int, min_pair: int = 5) -> list[dict]:
    out = []
    for (a, b), together in pair_bills.items():
        if together < min_pair:
            continue
        lift = together * n / (item_bills[a] * item_bills[b])
        if lift < 1.3:
            continue
        out.append({"items": [a, b], "bills_together": together, "lift": round(lift, 2),
                    "a_then_b": round(100 * together / item_bills[a], 1),
                    "b_then_a": round(100 * together / item_bills[b], 1)})
    out.sort(key=lambda c: (-c["lift"] * min(c["bills_together"], 50), c["items"]))
    return out[:10]


def _cross_sell(pair_cust: Counter, item_cust: Counter, n: int, bought: dict, min_pair: int = 5):
    lifts: dict[str, dict[str, float]] = defaultdict(dict)
    pairs = []
    for (a, b), both in pair_cust.items():
        if both < min_pair:
            continue
        lift = both * n / (item_cust[a] * item_cust[b])
        if lift < 1.2:
            continue
        lifts[a][b] = lifts[b][a] = round(lift, 3)
        for x, y in ((a, b), (b, a)):
            pairs.append({"from": x, "to": y, "lift": round(lift, 2),
                          "also_buy_pct": round(100 * both / item_cust[x], 1),
                          "audience": item_cust[x] - both})
    pairs.sort(key=lambda p: (-(p["lift"] * min(p["audience"], 100)), p["from"], p["to"]))
    return {k: dict(v) for k, v in lifts.items()}, pairs[:10]


def _refills(bought: dict):
    """Items people buy again on a steady cycle: each repeat buyer's median gap, then the median
    across buyers. Needs 5 repeat buyers. Returns (summary, {item: cycle days})."""
    gaps_by_item: dict[str, list] = defaultdict(list)
    for items in bought.values():
        for item, dates in items.items():
            days = sorted({d.date() for d in dates})
            gaps = [(days[i] - days[i - 1]).days for i in range(1, len(days))]
            if gaps:
                gaps_by_item[item].append(median(gaps))
    cycles = {i: float(median(g)) for i, g in gaps_by_item.items() if len(g) >= 5 and median(g) >= 3}
    summary = sorted(({"item": i, "cycle_days": round(c), "repeat_buyers": len(gaps_by_item[i])}
                      for i, c in cycles.items()), key=lambda r: -r["repeat_buyers"])
    return summary[:10], cycles


def next_items(a: dict, customer_id: str, limit: int = 3) -> list[dict]:
    """What this customer hasn't tried, ranked by how strongly their own items point to it."""
    mine = a["_bought"].get(customer_id) or {}
    scores: dict[str, float] = {}
    because: dict[str, str] = {}
    for item in mine:
        for other, lift in (a["_lifts"].get(item) or {}).items():
            if other in mine:
                continue
            if lift > scores.get(other, 0):
                scores[other], because[other] = lift, item
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"item": i, "lift": round(s, 2), "because": because[i], "confident": s >= 1.5} for i, s in ranked]


def due_refills(a: dict, customer_id: str, today: datetime, early: int = 3, late: int = 10) -> list[dict]:
    """Items on a cycle this customer is due to buy again (from `early` days before to `late` after)."""
    out = []
    for item, last in (a["_bought"].get(customer_id) or {}).items():
        cycle = a["_cycles"].get(item)
        if not cycle:
            continue
        due = last + timedelta(days=cycle)
        off = (today - due).days
        if -early <= off <= late:
            out.append({"item": item, "cycle_days": round(cycle), "days_from_due": off})
    return sorted(out, key=lambda r: abs(r["days_from_due"]))


def public(a: dict) -> dict:
    """The analysis without its private lookup tables, for the API."""
    return {k: v for k, v in a.items() if not k.startswith("_")}
