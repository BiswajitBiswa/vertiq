import random
from datetime import datetime, timedelta

from vertiq import patterns
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)
MENU = {"Margherita": "Pizza", "Farmhouse": "Pizza", "Cola": "Drinks", "Lemonade": "Drinks", "Fries": "Sides", "Salad": "Sides"}


def _orders(n=400, seed=5, planted=True):
    """Orders with a channel, a payment method, a time and line items, each drawn independently. Planted (only
    when `planted`): online orders rarely include a drink (10% against 60%), and orders with pizza are worth
    more. Everything else is unrelated."""
    rnd = random.Random(seed)
    events = []
    for i in range(n):
        online = rnd.random() < 0.4
        items = []
        pizza = rnd.random() < 0.5
        if pizza:
            items.append(rnd.choice(["Margherita", "Farmhouse"]))
        if rnd.random() < ((0.1 if online else 0.6) if planted else 0.4):
            items.append(rnd.choice(["Cola", "Lemonade"]))
        if rnd.random() < 0.5:
            items.append(rnd.choice(["Fries", "Salad"]))
        value = (300 if (pizza and planted) else 200) + rnd.gauss(0, 40)
        at = (NOW - timedelta(days=rnd.randint(0, 60))).replace(hour=rnd.randint(9, 21), minute=rnd.randint(0, 59))
        events.append(Event(f"C{rnd.randint(1, 300)}", at, "order", value,
                            {"channel": "online" if online else "store", "payment": rnd.choice(["card", "cash", "wallet"])},
                            tuple((x, 1) for x in items)))
    return Dataset([Entity(f"C{i}") for i in range(1, 301)], events)


def test_planted_patterns_are_found_and_described():
    r = patterns.find(_orders(), item_groups=MENU, noun="orders", currency="₹")
    texts = " | ".join(f["text"] for f in r["found"])
    assert any(f["kind"] == "rate" and {f["feature"], f["outcome"]} == {"channel = online", "includes Drinks"} for f in r["found"])
    assert "of orders where channel is online include Drinks" in texts or "of orders that include Drinks have channel online" in texts
    assert any(f["kind"] == "value" and f["feature"] == "includes Pizza" and f["with"] > f["without"] for f in r["found"])
    assert "payment" not in texts                                   # noise stays out
    assert r["tested"] > r["features"]


def test_nothing_is_found_in_random_data():
    false = sum(len(patterns.find(_orders(seed=s, planted=False), item_groups=MENU)["found"]) for s in range(10))
    assert false <= 2


def test_small_data_waits():
    assert patterns.find(_orders(n=12))["status"] == "learning"
