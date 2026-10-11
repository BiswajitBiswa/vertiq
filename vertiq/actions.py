"""Smart actions: the few things worth doing now, each ready to become a campaign draft (who, why,
with which offer, and what it should bring in). A host turns an action into its own campaign.

  winback        persuadable and needs-offer customers from targeting (goal winback)
  second_visit   first-timers targeting says a message would bring back
  slow_mover     a slow-moving item, offered only to the people who buy it or its category
  new_arrival    something new, told to the people who buy that kind of thing
  cross_sell     people who buy A but never B, when A and B go together
  referral       loyal regulars, asked to bring a friend

Every action carries a stable key (so "create this" finds the same action again), the audience ids,
the goal, and either an expected extra revenue (when targeting has one) or the money at stake.
"""
from __future__ import annotations

from . import inventory

MIN_AUDIENCE = 5


def _targeted(plan: dict | None, key: str, title: str, goal: str, why: str, only=None) -> dict | None:
    if not plan or plan.get("status") != "ok":
        return None
    people = [p for p in plan["customers"] if p["send"] and (only is None or only(p))]
    if len(people) < MIN_AUDIENCE:
        return None
    offer = plan.get("offer")
    with_offer = [p["id"] for p in people if p["with_offer"]]
    return {"key": key, "title": title.format(n=len(people)), "goal": goal, "why": why,
            "audience_ids": [p["id"] for p in people], "audience_count": len(people),
            "offer": offer["label"] if offer and with_offer else "", "with_offer_count": len(with_offer),
            "expected_extra_revenue": round(sum(p["expected_value"] for p in people)), "source": plan.get("source")}


def _fans(choices: dict, inv_view: dict | None, item: str) -> list[str]:
    """Customers who bought `item`, or anything in its catalogue category."""
    k = inventory.key(item)
    cat = (inv_view or {}).get("category", {}).get(k, "")
    out = []
    for cid, bought in (choices.get("_bought") or {}).items():
        keys = {inventory.key(i) for i in bought}
        if k in keys or (cat and any((inv_view or {}).get("category", {}).get(x) == cat for x in keys)):
            out.append(cid)
    return out


def suggest(*, winback_plan: dict | None = None, second_visit_plan: dict | None = None, choices: dict | None = None,
            inv_report: dict | None = None, inv_view: dict | None = None, referrers: dict | None = None,
            one_visit_ids: set | None = None, currency: str = "₹", limit: int = 6) -> list[dict]:
    out = []
    a = _targeted(winback_plan, "winback", "Win back {n} customers who would come back if asked", "winback",
                  "Targeting learned from your control groups says a message brings these people back.")
    if a:
        out.append(a)
    a = _targeted(second_visit_plan, "second_visit", "Bring {n} first-timers back for a second visit", "secondVisit",
                  "A second visit is where most customers are won or lost.",
                  only=(lambda p: p["id"] in one_visit_ids) if one_visit_ids is not None else None)
    if a:
        out.append(a)
    ch = choices or {}
    rep = inv_report or {}
    for s in (rep.get("slow_movers") or [])[:2]:
        who = _fans(ch, inv_view, s["name"])
        if len(who) >= MIN_AUDIENCE:
            out.append({"key": f"slow_mover:{inventory.key(s['name'])}", "goal": "custom",
                        "title": f"Offer {s['name']} to {len(who)} people who buy it or its kind",
                        "why": f"{currency}{s.get('tied_up', 0):,} is tied up in it and it's selling slowly.",
                        "audience_ids": who, "audience_count": len(who), "offer": "", "item": s["name"],
                        "money_at_stake": s.get("tied_up", 0)})
    for n in (rep.get("new_arrivals") or [])[:1]:
        who = [c for c in _fans(ch, inv_view, n["name"])
               if inventory.key(n["name"]) not in {inventory.key(i) for i in ch.get("_bought", {}).get(c, {})}]
        if len(who) >= MIN_AUDIENCE:
            out.append({"key": f"new_arrival:{inventory.key(n['name'])}", "goal": "custom",
                        "title": f"Tell {len(who)} people about the new {n['name']}",
                        "why": "They buy this kind of thing and haven't tried it yet.",
                        "audience_ids": who, "audience_count": len(who), "offer": "", "item": n["name"]})
    for x in (ch.get("cross_sell") or [])[:1]:
        if inventory.blocked(inv_view, x["to"]):
            continue
        who = [c for c, bought in (ch.get("_bought") or {}).items() if x["from"] in bought and x["to"] not in bought]
        if len(who) >= MIN_AUDIENCE:
            out.append({"key": f"cross_sell:{inventory.key(x['from'])}>{inventory.key(x['to'])}", "goal": "custom",
                        "title": f"Suggest {x['to']} to {len(who)} people who buy {x['from']}",
                        "why": f"{x['also_buy_pct']:g}% of {x['from']} buyers also buy {x['to']}.",
                        "audience_ids": who, "audience_count": len(who), "offer": "", "item": x["to"]})
    r = referrers or {}
    if r.get("count", 0) >= MIN_AUDIENCE:
        ids = [c["id"] for c in r["customers"]]
        out.append({"key": "referral", "goal": "custom", "title": f"Ask {len(ids)} loyal regulars to bring a friend",
                    "why": "They visit often, recently, and are likely to stay.", "audience_ids": ids,
                    "audience_count": len(ids), "offer": ""})
    out.sort(key=_rank)
    return out[:limit]


def _rank(a: dict) -> tuple:
    """Actions with an expected extra revenue first (largest first), then stock actions by the money
    tied up, then the rest: the two numbers measure different things, so they are never compared."""
    if a.get("expected_extra_revenue") is not None:
        return (0, -a["expected_extra_revenue"])
    if a.get("money_at_stake"):
        return (1, -a["money_at_stake"])
    return (2, -a.get("audience_count", 0))
