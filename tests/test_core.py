"""vertiq on its own: plain data in, plain data out (no database, no host)."""
import random
from datetime import datetime, timedelta

from vertiq import affinity, churn, inventory, offers, trends, verticals
from vertiq.features import Visit
from vertiq.learn import Logistic, auc

NOW = datetime(2026, 9, 30)


def test_churn_learns_from_plain_histories():
    rnd = random.Random(3)
    hists = []
    for b in range(2):
        visits = {}
        for i in range(250):
            gap = rnd.choice([7, 10, 14])
            stop = NOW + timedelta(days=999) if rnd.random() < 0.4 else NOW - timedelta(days=rnd.randint(20, 400))
            t, vs = NOW - timedelta(days=600 - rnd.randint(0, 100)), []
            while t < NOW and t < stop:
                vs.append(Visit(t, rnd.uniform(200, 900)))
                t += timedelta(days=gap * (2.5 if (stop - t).days < 70 else 1))
            if vs:
                visits[f"c{i}"] = vs
        hists.append(churn.History(f"b{b}", "cafe", 30, 90, visits))
    out = churn.train(churn.examples(hists))
    assert out["status"] == "trained" and out["metrics"]["auc"] > out["metrics"]["formula_auc"]
    p = churn.predict(out["model"], hists[0].visits, NOW, 30, "cafe", "b0")
    assert p and all(0 <= v <= 1 for v in p.values())
    assert isinstance(Logistic.from_dict(out["model"].to_dict()), Logistic)


def test_choices_inventory_offers_trends_and_verticals():
    lines = [(f"c{i}", f"t{i}", NOW, it) for i in range(40) for it in ("Burger", "Fries")]
    lines += [(f"d{i}", f"u{i}", NOW, "Tea") for i in range(40)]
    lines += [(f"e{i}", f"v{i}", NOW, "Burger") for i in range(10)]
    a = affinity.analyse(lines)
    assert affinity.next_items(a, "e0")[0]["item"] == "Fries"
    fields, rows = inventory.parse_table("Product,Stock,Price,Cost\nFries,0,100,30\nTea,50,60,55\n")
    assert fields == ["cost", "name", "price", "stock"] and len(rows) == 2
    items = [inventory.Item(name=r["name"], stock=inventory.num(r["stock"]), price=inventory.num(r["price"]),
                            cost=inventory.num(r["cost"]), added_at=NOW - timedelta(days=60)) for r in rows]
    view = inventory.view(items, inventory.report(items, {}, NOW))
    assert inventory.blocked(view, "fries") == "out of stock" and inventory.blocked(view, "Tea", "10% off")
    assert offers.spend_more(500, {"on": True, "reward_pct": 10, "stretch_pct": 20}, currency="$")["words"] == \
        "$60 off when you spend $600"
    assert trends.trending_for(a, [("owner", ["fries"])]) == {"Fries": {"term": "fries", "source": "owner"}}
    assert verticals.get("b2b").customer_noun == "accounts" and verticals.get("unknown").name == "generic"
    assert auc([0.1, 0.9], [0, 1]) == 1.0


def test_check_ins_beside_a_bill_are_not_extra_visits():
    from datetime import datetime
    from types import SimpleNamespace as E
    from vertiq.features import visits_from
    d1, d2, d3 = datetime(2026, 3, 1, 10), datetime(2026, 3, 1, 18), datetime(2026, 3, 4, 9)
    events = [E(occurred_at=d1, amount=420, channel="pos"), E(occurred_at=d2, amount=0, channel="counter"),
              E(occurred_at=d3, amount=0, channel="qr"), E(occurred_at=d3, amount=0, channel="app", is_checkin=True)]
    vs = visits_from(events, checkin_channel=("qr", "counter"))
    assert [(v.at, v.amount) for v in vs] == [(d1, 420.0), (d3, 0.0)]     # punch on bill day dropped; one check-in a day
    # with QR alone the Counter punch is taken for a sale: a second visit on bill day (what this fixes)
    assert len(visits_from(events, checkin_channel="qr")) == 3
