"""Revenue, not just returns: what each campaign added in money against its control group, a model of
how much people spend when they come back (so targeting can favour messages that grow the basket),
and the regulars most worth asking for a referral (more customers).

  campaign_value  per campaign: return rate, bill size (AOV) and revenue per customer, messaged vs
                  control, and the extra revenue the campaign can claim
  train_bill      learns log(first bill after a send) from the customer's features, whether they
                  were messaged and whether the message carried an offer
  referrers       loyal, recent, likely-to-stay customers: the people to ask to bring a friend
"""
from __future__ import annotations

import math

from .learn import Linear, r2
from .uplift import arm

MIN_BILLS, MIN_CONTROL_BILLS = 100, 10


def _side(people: dict, bills: dict) -> dict:
    n = len(people)
    came = [cid for cid in people if bills.get(cid)]
    revenue = sum(sum(bills[cid]) for cid in came)
    n_bills = sum(len(bills[cid]) for cid in came)
    return {"customers": n, "returned": len(came), "return_rate": round(len(came) / n, 4) if n else None,
            "aov": round(revenue / n_bills, 1) if n_bills else None,
            "revenue_per_customer": round(revenue / n, 1) if n else None, "revenue": round(revenue, 1)}


def campaign_value(groups: dict[str, dict], bills: dict[str, dict[str, list[float]]], names: dict[str, str] | None = None,
                   window_days: int = 14) -> dict:
    """`groups`: {campaign id: {"messaged": {customer id: sent at}, "control": {...}}};
    `bills`: {campaign id: {customer id: [bill amounts within the window]}}."""
    out, total = [], 0.0
    for camp, g in groups.items():
        b = bills.get(camp, {})
        m, c = _side(g["messaged"], b), _side(g["control"], b)
        extra = None
        if m["customers"] and c["customers"]:
            extra = round((m["revenue_per_customer"] - c["revenue_per_customer"]) * m["customers"], 0)
            total += extra
        out.append({"campaign_id": camp, "name": (names or {}).get(camp, ""), "messaged": m, "control": c,
                    "aov_change": (round(m["aov"] - c["aov"], 1) if m["aov"] is not None and c["aov"] is not None else None),
                    "extra_revenue": extra})
    out.sort(key=lambda r: -(r["extra_revenue"] or 0))
    return {"window_days": window_days, "campaigns": out, "extra_revenue_total": round(total, 0)}


def train_bill(ex: dict) -> dict:
    """Learn the bill of the first visit after a send from uplift.examples output.
    {"status": learning | trained, "metrics", "model" (when trained)}."""
    idx = [i for i, b in enumerate(ex["bill"]) if b]
    control = sum(1 for i in idx if not ex["messaged"][i])
    if len(idx) < MIN_BILLS or control < MIN_CONTROL_BILLS:
        return {"status": "learning", "bills": len(idx), "control_bills": control,
                "need": {"bills": MIN_BILLS, "control_bills": MIN_CONTROL_BILLS}}
    rows = [ex["rows"][i] for i in idx]
    y = [math.log1p(ex["bill"][i]) for i in idx]
    model = Linear().fit(rows, y, l2=1e-3, epochs=30)
    model.fit_offsets(rows, y, [ex["groups"][i] for i in idx])
    metrics = {"bills": len(idx), "control_bills": control, "r2": round(r2([model.predict(r) for r in rows], y) or 0, 4),
               "offer_effect_pct": _offer_effect(model, [ex["base"][i] for i in idx])}
    return {"status": "trained", "metrics": metrics, "model": model}


def _offer_effect(model: Linear, bases: list[dict]) -> float:
    """How much the model says an offer changes the bill, averaged over real customers (%): the
    offer's main term and its interactions together, as the model would use them."""
    diffs = [model.predict(arm(f, True, True)) - model.predict(arm(f, True, False)) for f in bases]
    return round(100 * (math.exp(sum(diffs) / len(diffs)) - 1), 1) if diffs else 0.0


def referrers(rows: list[dict], risk_days: int, limit: int = 20, min_visits: int = 3, min_stay: int = 60) -> dict:
    """The customers most worth asking to bring a friend: regulars, seen within `risk_days`, and
    likely to stay. `rows`: {id, name, frequency, recency_days, monetary, return_chance | churn_risk}."""
    picks = []
    for r in rows:
        recency = r.get("recency_days")
        if int(r.get("frequency") or 0) < min_visits or recency is None or int(recency) > risk_days:
            continue
        stay = r["return_chance"] if r.get("return_chance") is not None else 100 - int(r.get("churn_risk", 100))
        if stay < min_stay:
            continue
        picks.append({"id": r["id"], "name": (r.get("name") or "").split(" ")[0], "visits": r["frequency"],
                      "spend": r.get("monetary"), "return_chance": stay,
                      "why": f"{r['frequency']} visits, last one {r['recency_days']} days ago"})
    picks.sort(key=lambda p: (-p["return_chance"], -(p["spend"] or 0)))
    return {"count": len(picks), "customers": picks[:limit]}
