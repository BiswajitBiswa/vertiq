"""The context contract: one versioned shape every consumer reads (content generation, AI-drafted
messages, insights screens, campaign creation, a point of sale), so a better model improves all of
them at once and none of them needs to know which model said what.

  business_context  what sells, what goes together, what is new / slow / out of stock, what is
                    trending here, which tactics are worth testing, the offers in force, and how
                    the models are doing
  customer_context  one customer: a few safe facts, their usual and next items, refills due, their
                    chance to return and why, the group targeting puts them in, and the offer that
                    fits them; nothing out of stock

Plain dicts in, plain dicts out. `facts()` turns a business context into short sentences a language
model may use (and nothing else), so prompts never invent what isn't true.
"""
from __future__ import annotations

from . import inventory

SCHEMA_VERSION = 1


def _in_stock(names, inv_view) -> list[str]:
    return [n for n in names if not inventory.blocked(inv_view, n)]


def business_context(*, business: dict, choices: dict | None = None, inv_report: dict | None = None,
                     inv_view: dict | None = None, insights: list[dict] | None = None, ideas: list[dict] | None = None,
                     offers: list[dict] | None = None, spend_more_rule: dict | None = None, trending: dict | None = None,
                     revenue: dict | None = None, models: dict | None = None) -> dict:
    """`business`: {id, name, sector, currency, customer_noun}. Every other part is optional."""
    a = choices or {}
    rep = inv_report or {}
    top = _in_stock([t["item"] for t in a.get("top_items") or []], inv_view)
    combos = [c for c in a.get("combos") or [] if all(not inventory.blocked(inv_view, i) for i in c["items"])]
    return {
        "schema": SCHEMA_VERSION,
        "business": business,
        "sells": {"top_items": top[:8], "combos": combos[:5],
                  "cross_sell": [x for x in a.get("cross_sell") or [] if not inventory.blocked(inv_view, x["to"])][:5],
                  "refill_cycles": (a.get("refills") or [])[:5]},
        "stock": {"new_arrivals": [x["name"] for x in rep.get("new_arrivals") or []][:8],
                  "slow_movers": [x["name"] for x in rep.get("slow_movers") or []][:8],
                  "out_of_stock": [x["name"] for x in rep.get("out_of_stock") or []][:20],
                  "reorder": [{"name": x["name"], "days_left": x["days_left"], "order_qty": x["order_qty"]}
                              for x in rep.get("reorder") or []][:10],
                  "tied_up": rep.get("tied_up", 0)},
        "trending": [{"item": i, **t} for i, t in (trending or {}).items() if not inventory.blocked(inv_view, i)][:5],
        "insights": [{k: x[k] for k in ("key", "kind", "title", "action") if k in x} for x in insights or []][:6],
        "ideas": [{k: x[k] for k in ("tactic", "mechanism", "source", "status") if k in x} for x in ideas or []][:5],
        "offers": offers or [],
        "spend_more_rule": spend_more_rule or {},
        "revenue": {"extra_revenue_total": (revenue or {}).get("extra_revenue_total"),
                    "campaigns_measured": len((revenue or {}).get("campaigns") or [])},
        "models": models or {},
    }


def facts(ctx: dict, limit: int = 8) -> list[str]:
    """Short, true sentences from a business context that a writer may use."""
    s, st = ctx.get("sells", {}), ctx.get("stock", {})
    out = []
    if s.get("top_items"):
        out.append("Best sellers: " + ", ".join(s["top_items"][:4]))
    for c in s.get("combos", [])[:2]:
        out.append(f"{c['items'][0]} and {c['items'][1]} are often bought together")
    if st.get("new_arrivals"):
        out.append("New: " + ", ".join(st["new_arrivals"][:3]))
    for t in ctx.get("trending", [])[:2]:
        out.append(f"{t['item']} is trending")
    if st.get("slow_movers"):
        out.append("Worth featuring (plenty in stock): " + ", ".join(st["slow_movers"][:3]))
    return out[:limit]


def customer_context(*, customer: dict, choices: dict | None = None, outlook: dict | None = None,
                     targeting: dict | None = None, spend_more: dict | None = None, inv_view: dict | None = None,
                     business: dict | None = None) -> dict:
    """`customer`: {id, first_name, age_range?, visits, avg_bill}. `choices`: affinity.customer view
    ({items_bought, next_items, due_refills}); `outlook`: churn.explain; `targeting`: one row of
    uplift.plan's customers."""
    c = choices or {}
    o = outlook or {}
    t = targeting or {}
    return {
        "schema": SCHEMA_VERSION,
        "business": business or {},
        "customer": {k: customer.get(k) for k in ("id", "first_name", "age_range", "visits", "avg_bill") if k in customer},
        "usual": _in_stock(c.get("items_bought") or [], inv_view)[:5],
        "next_items": [x for x in c.get("next_items") or [] if not inventory.blocked(inv_view, x["item"])],
        "refills_due": [x for x in c.get("due_refills") or [] if not inventory.blocked(inv_view, x["item"])],
        "outlook": {k: o.get(k) for k in ("source", "return_chance", "churn_risk", "window_days", "risks", "strengths")
                    if k in o},
        "targeting": {k: t.get(k) for k in ("group", "send", "with_offer", "added_return_pts", "expected_bill",
                                           "expected_value") if k in t},
        "spend_more": spend_more,
    }
