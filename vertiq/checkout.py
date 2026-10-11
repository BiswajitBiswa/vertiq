"""At the counter: given who is buying and what is in the basket, what to suggest right now.

  add_ons     items that go with what's in the basket (bought together, or bought by the same
              people), in stock, not already in the basket
  spend_more  when the business's spend-more rule is on and the bill is close to this customer's
              threshold: "add X more to get Y off"
  refills     items this customer is due to buy again
  customer    a few safe facts (first name, visits, chance to return) for the cashier

Everything comes from analyses the host already holds (affinity, inventory view, offers rule), so
a call is a few dictionary lookups: fast enough for a till.
"""
from __future__ import annotations

from datetime import datetime

from . import affinity, inventory, offers

NUDGE_WITHIN = 0.35         # nudge when the bill is within 35% of the threshold


def suggest(basket: list[str], *, choices: dict | None = None, inv_view: dict | None = None,
            customer_id: str | None = None, avg_bill: float | None = None, total: float = 0.0,
            spend_more_rule: dict | None = None, customer: dict | None = None, today: datetime | None = None,
            currency: str = "₹", limit: int = 3) -> dict:
    a = choices or {}
    in_basket = {affinity.clean_item(i) for i in basket if affinity.clean_item(i)}
    keys = {inventory.key(i) for i in in_basket}
    scores: dict[str, tuple[float, str]] = {}
    pair_lift = {}
    for c in a.get("combos") or []:
        x, y = c["items"]
        pair_lift[(x, y)] = pair_lift[(y, x)] = c["lift"]
    for item in in_basket:
        for other, lift in (a.get("_lifts", {}).get(item) or {}).items():
            lift = max(lift, pair_lift.get((item, other), 0))
            if inventory.key(other) in keys or inventory.blocked(inv_view, other):
                continue
            if lift > scores.get(other, (0, ""))[0]:
                scores[other] = (lift, item)
    add_ons = [{"item": i, "because": b, "lift": round(l, 2)}
               for i, (l, b) in sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]]
    nudge = None
    sm = offers.spend_more(avg_bill or 0, spend_more_rule or {}, currency=currency) if avg_bill else None
    if sm and total < sm["threshold"] and sm["threshold"] - total <= NUDGE_WITHIN * sm["threshold"]:
        gap = sm["threshold"] - total
        nudge = {**sm, "add": round(gap), "say": f"Add {currency}{gap:,.0f} more to get {currency}{sm['reward']} off"}
    refills = []
    if customer_id and a.get("_bought"):
        refills = [r for r in affinity.due_refills(a, customer_id, today or datetime.utcnow())
                   if inventory.key(r["item"]) not in keys and not inventory.blocked(inv_view, r["item"])][:limit]
    return {"add_ons": add_ons, "spend_more": nudge, "refills": refills, "customer": customer or None}
