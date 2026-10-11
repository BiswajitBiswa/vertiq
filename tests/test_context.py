"""The context contract, smart actions and the counter, on plain data."""
from datetime import datetime, timedelta

from vertiq import actions, affinity, checkout, context, inventory

NOW = datetime(2026, 9, 30)


def _shop():
    lines = [(f"c{i}", f"t{i}", NOW, it) for i in range(40) for it in ("Burger", "Fries")]
    lines += [(f"d{i}", f"u{i}", NOW, "Tea") for i in range(30)]
    lines += [(f"e{i}", f"v{i}", NOW, "Burger") for i in range(12)]
    a = affinity.analyse(lines)
    items = [inventory.Item("Burger", "Food", 200, 80, 50, added_at=NOW - timedelta(days=90)),
             inventory.Item("Fries", "Food", 100, 30, 40, added_at=NOW - timedelta(days=90)),
             inventory.Item("Tea", "Drinks", 60, 15, 0, added_at=NOW - timedelta(days=90)),
             inventory.Item("Brownie", "Food", 150, 50, 90, added_at=NOW - timedelta(days=90))]
    rep = inventory.report(items, {"burger": 30, "fries": 25}, NOW)
    return a, items, rep, inventory.view(items, rep)


def test_business_context_and_facts_never_promote_out_of_stock():
    a, _items, rep, view = _shop()
    ctx = context.business_context(business={"id": "b", "name": "Diner", "sector": "restaurant"}, choices=a,
                                   inv_report=rep, inv_view=view)
    assert ctx["schema"] == context.SCHEMA_VERSION and "Tea" not in ctx["sells"]["top_items"]
    assert ctx["stock"]["out_of_stock"] == ["Tea"] and "Brownie" in ctx["stock"]["slow_movers"]
    f = context.facts(ctx)
    assert any("Burger and Fries" in x or "Fries and Burger" in x for x in f) and not any("Tea" in x for x in f)


def test_actions_from_targeting_stock_and_regulars():
    a, _items, rep, view = _shop()
    plan = {"status": "ok", "source": "model", "offer": {"id": "o", "label": "10% off"},
            "customers": [{"id": f"c{i}", "send": True, "with_offer": i % 2 == 0, "expected_value": 40.0} for i in range(8)]}
    out = actions.suggest(winback_plan=plan, choices=a, inv_report=rep, inv_view=view,
                          referrers={"count": 6, "customers": [{"id": f"c{i}"} for i in range(6)]})
    keys = [x["key"] for x in out]
    assert keys[0] == "winback" and out[0]["expected_extra_revenue"] == 320 and out[0]["offer"] == "10% off"
    assert "cross_sell:burger>fries" in keys and "referral" in keys


def test_checkout_suggests_add_ons_and_a_spend_more_nudge():
    a, _items, _rep, view = _shop()
    out = checkout.suggest(["Burger"], choices=a, inv_view=view, avg_bill=500, total=520,
                           spend_more_rule={"on": True, "reward_pct": 10, "stretch_pct": 20})
    assert out["add_ons"][0]["item"] == "Fries" and out["add_ons"][0]["because"] == "Burger"
    assert out["spend_more"]["say"] == "Add ₹80 more to get ₹60 off"
    assert checkout.suggest(["Burger", "Fries"], choices=a, inv_view=view)["add_ons"] == []
