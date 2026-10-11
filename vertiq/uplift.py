"""Offer targeting: who a message actually brings back, who needs an offer to come, and who would
come anyway, learned from control groups kept out on purpose.

Every marketing send is an experiment: the customers messaged and the ones kept out on purpose
One shared model (all businesses, all sectors) learns the chance to
come back within the return window from the customer's features at send time, whether they were
messaged, and whether the message carried an offer, with those two interacting with the customer's
features. Uplift = chance if messaged − chance if not: the part the message can claim.

  persuadable     comes back because of the message: worth a send
  needs_offer     only an offer moves them: send it with the offer
  sure_thing      comes back anyway: don't spend a discount on them
  lost_cause      unlikely either way: don't spend a message on them
  do_not_disturb  less likely to come back when messaged: leave them alone

Until the control groups hold enough sends to learn from, the plan falls back to rules on the churn
model (or the formula) and says so (source "rules").
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .churn import History
from .features import Visit, cohort_cadence, reference_now, returned, snapshot
from .learn import Logistic

MIN_TREATED, MIN_CONTROL, MIN_RETURNS = 200, 40, 20
# what the interaction terms let the effect of a message (and of an offer) depend on
INTERACT = ("log_due", "log_freq", "one_visit", "trend", "log_avg_bill")
_OFFER = re.compile(r"(\d+\s?%|[₹$€£]\s?\d+|rs\.?\s?\d+|\bfree\b|\boff\b|\bdiscount\b|double points|\bcashback\b)", re.I)


@dataclass
class Send:
    """One marketing send: messaged, or held out on purpose (the control group)."""
    customer_id: str
    at: datetime
    messaged: bool
    offer: bool = False
    purpose: str = ""


@dataclass
class OfferSpec:
    """The business's offer: kind percent | flat | other, its value, and how it reads."""
    id: str
    kind: str
    value: float
    label: str = ""
    uses: int = 0


def mentions_offer(text: str) -> bool:
    return bool(_OFFER.search(text or ""))


def arm(f: dict, messaged: bool, offer: bool, purpose: str = "") -> dict:
    """A customer's features with the treatment terms for one arm (messaged? with an offer?)."""
    row = dict(f)
    t, o = (1.0 if messaged else 0.0), (1.0 if messaged and offer else 0.0)
    row["t"], row["t_offer"] = t, o
    for k in INTERACT:
        row[f"t*{k}"] = t * f.get(k, 0.0)
        row[f"o*{k}"] = o * f.get(k, 0.0)
    if purpose:
        row[f"purpose:{purpose}"] = 1.0
    return row


def examples(batches: list[tuple[History, int, list[Send]]]) -> dict:
    """Every finished send with features at send time and its outcome. `batches`: one per business:
    (its history, its return window in days, its sends)."""
    out = {"rows": [], "base": [], "messaged": [], "y": [], "bill": [], "groups": [], "treated": 0, "control": 0}
    for h, window, sends in batches:
        hist = h.visits
        cads: dict = {}
        for e in sends:
            vs = hist.get(e.customer_id)
            if not vs:
                continue
            key = e.at.date()
            if key not in cads:
                cads[key] = cohort_cadence(hist, e.at, 30.0)
            f = snapshot(vs, e.at, cads[key], h.sector or "", len(hist))
            if f is None:
                continue
            out["rows"].append(arm(f, e.messaged, e.offer, e.purpose or ""))
            out["base"].append(f)
            out["messaged"].append(e.messaged)
            out["y"].append(returned(vs, e.at, window))
            out["bill"].append(first_bill(vs, e.at, window))
            out["groups"].append(h.business_id)
            out["treated" if e.messaged else "control"] += 1
    return out


def first_bill(vs: list, start: datetime, days: int) -> float | None:
    """The bill of their first visit after a send (within the window), or None if they didn't come."""
    end = start + timedelta(days=days)
    for v in vs:
        if start <= v.at < end and v.amount > 0:
            return v.amount
    return None


def train(ex: dict) -> dict:
    """{"status": learning | trained, "metrics", "model" (when trained)}."""
    y = ex["y"]
    if ex["treated"] < MIN_TREATED or ex["control"] < MIN_CONTROL or sum(y) < MIN_RETURNS:
        return {"status": "learning", "treated": ex["treated"], "control": ex["control"], "returned": sum(y),
                "need": {"treated": MIN_TREATED, "control": MIN_CONTROL, "returned": MIN_RETURNS}}
    model = Logistic().fit(ex["rows"], y, l2=1e-2)
    model.fit_offsets(ex["rows"], y, ex["groups"], prior=400.0)
    metrics = {"treated": ex["treated"], "control": ex["control"], "rows": len(y),
               "return_rate": round(sum(y) / len(y), 4), **_observed_lift_by_half(model, ex)}
    return {"status": "trained", "metrics": metrics, "model": model}


def _observed_lift_by_half(model: Logistic, ex: dict) -> dict:
    """The honest check for an uplift model: rank sends by predicted uplift, then compare the
    real messaged-vs-control return gap in the top half with the bottom half."""
    scored = []
    for base, messaged, yv in zip(ex["base"], ex["messaged"], ex["y"]):
        u = model.predict(arm(base, True, False)) - model.predict(arm(base, False, False))
        scored.append((u, messaged, yv))
    scored.sort(key=lambda s: -s[0])
    half = len(scored) // 2

    def gap(part):
        t = [y for _u, m, y in part if m]
        c = [y for _u, m, y in part if not m]
        return round(sum(t) / len(t) - sum(c) / len(c), 4) if t and c else None
    return {"observed_lift_top_half": gap(scored[:half]), "observed_lift_bottom_half": gap(scored[half:])}


def offer_cost(o: OfferSpec | None, bill: float) -> float:
    if o is None:
        return 0.0
    v = float(o.value or 0)
    return bill * v / 100 if o.kind == "percent" else v if o.kind == "flat" else 0.0


def plan(visits: dict[str, list[Visit]], sector: str, business_id: str | None, *, goal: str = "winback",
         offer: OfferSpec | None = None, model: Logistic | None = None, churn_model: Logistic | None = None,
         bill_model=None, msg_cost: float = 0.0, limit: int = 200) -> dict:
    """Who to message for `goal`, with or without `offer`, ranked by the extra revenue the send adds.
    Without an uplift `model` it uses rules on the churn model (or on how overdue they are)."""
    if not visits:
        return {"status": "no_data", "customers": []}
    cut = reference_now(visits) + timedelta(seconds=1)
    cad = cohort_cadence(visits, cut, 30.0)
    hist = visits
    biz_sector = sector or ""
    people = []
    for cid, vs in hist.items():
        f = snapshot(vs, cut, cad, biz_sector, len(hist))
        if f is None:
            continue
        avg_bill = math.expm1(f["log_avg_bill"])
        if model is not None:
            p0 = model.predict(arm(f, False, False, goal), business_id)
            p_msg = model.predict(arm(f, True, False, goal), business_id)
            p_off = model.predict(arm(f, True, True, goal), business_id) if offer else p_msg
        else:
            p0, p_msg, p_off = _rules(f, churn_model, business_id)
        b0, b_msg, b_off = _bills(bill_model, f, goal, business_id, avg_bill, bool(offer))
        cost = offer_cost(offer, b_off)
        # the objective is extra revenue, not returns: a discount that brings someone back on a
        # smaller bill counts for less, and a message that grows the basket counts for more
        value_msg = p_msg * b_msg - p0 * b0 - msg_cost
        value_off = p_off * b_off - p0 * b0 - p_off * cost - msg_cost if offer else float("-inf")
        group = _group(p0, p_msg, p_off if offer else p_msg)
        send = group in ("persuadable", "needs_offer")
        use_offer = bool(offer) and send and value_off > value_msg
        people.append({"id": cid, "group": group, "send": send, "with_offer": use_offer,
                       "return_if_left": round(100 * p0), "return_if_messaged": round(100 * p_msg),
                       "return_with_offer": round(100 * p_off) if offer else None,
                       "added_return_pts": round(100 * ((p_off if use_offer else p_msg) - p0), 1),
                       "expected_bill": round(b_off if use_offer else b_msg),
                       "expected_value": round(max(value_msg, value_off) if send else 0.0, 1)})
    people.sort(key=lambda p: (-p["send"], -p["expected_value"], -p["added_return_pts"]))
    counts: dict[str, int] = {}
    for p in people:
        counts[p["group"]] = counts.get(p["group"], 0) + 1
    sending = [p for p in people if p["send"]]
    return {"status": "ok", "source": "model" if model is not None else "rules", "objective": "extra revenue",
            "bill_source": "model" if bill_model is not None else "average bill",
            "goal": goal,
            "offer": {"id": offer.id, "label": offer.label} if offer else None,
            "groups": counts, "send_count": len(sending),
            "with_offer_count": sum(1 for p in sending if p["with_offer"]),
            "discounts_saved": counts.get("sure_thing", 0),
            "expected_value": round(sum(p["expected_value"] for p in sending), 0),
            "customers": people[:limit]}

def _bills(bill_model, f: dict, goal: str, business_id: str, avg_bill: float, offer: bool):
    """Expected bill if left alone, if messaged, and if messaged with the offer."""
    if bill_model is None or avg_bill <= 0:
        return avg_bill, avg_bill, avg_bill
    est = lambda m, o: max(0.0, math.expm1(bill_model.predict(arm(f, m, o, goal), business_id)))  # noqa: E731
    return est(False, False), est(True, False), est(True, True) if offer else est(True, False)


def _group(p0: float, p_msg: float, p_off: float) -> str:
    best = max(p_msg, p_off)
    if p_msg - p0 <= -0.02 and p_off - p0 <= -0.02:
        return "do_not_disturb"
    if best < 0.08:
        return "lost_cause"
    if p0 >= 0.6 and best - p0 < 0.03:
        return "sure_thing"
    if p_msg - p0 >= 0.03:
        return "persuadable"
    if p_off - p0 >= 0.03:
        return "needs_offer"
    return "low_effect"


def _rules(f: dict, churn_model, business_id: str) -> tuple[float, float, float]:
    """Before there are control groups to learn from: the chance to return from the churn model
    (or from how overdue they are), and the industry rule of thumb that a message helps people who
    are due or a little overdue most, an offer helps the long-quiet, and the just-visited need nothing."""
    if churn_model is not None:
        p0 = churn_model.predict(f, business_id)
    else:
        p0 = max(0.02, min(0.95, 0.9 - 0.35 * f["log_due"]))
    due = math.expm1(f["log_due"])
    msg = 0.06 if 0.8 <= due <= 2.5 else 0.03 if due <= 4 else 0.01
    off = msg + (0.05 if due > 1.5 else 0.01)
    return p0, min(0.99, p0 + msg), min(0.99, p0 + off)
