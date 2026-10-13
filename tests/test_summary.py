import random
from datetime import datetime, timedelta

from vertiq import narrate, summary
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)


def test_summary_reports_changes_and_the_group_that_moved():
    rnd = random.Random(3)
    ents, evs = [], []
    for i in range(600):
        g = ["north", "south", "east"][i % 3]
        ents.append(Entity(f"C{i}", group=g))
        for d in range(60):
            day = NOW - timedelta(days=60 - d)
            rate = 0.06 if (g == "north" and d >= 30) else 0.03          # north doubles in the last 30 days
            if rnd.random() < rate:
                evs.append(Event(f"C{i}", day, "order", 100.0))
    ds = Dataset(ents, evs)
    s = summary.summarise(ds, days=30, now=NOW)
    order = s["types"][0]
    assert order["type"] == "order" and order["after"] > order["before"] and order["clear"]
    assert s["groups"][0]["group"] == "north" and s["groups"][0]["clear"]
    assert not any(g["clear"] for g in s["groups"][1:])
    text = narrate.summary(s, entities="customers")
    assert text.startswith("In the last 30 days:") and "group north" in text
    assert narrate.check(text, narrate.facts(s)) == []                # every number comes from the result
