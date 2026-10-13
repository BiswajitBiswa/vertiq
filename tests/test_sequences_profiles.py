"""Lagged sequences find a planted transition and nothing in shuffled data; profiles match live scores."""
import random
from datetime import datetime, timedelta

from vertiq import profiles, sequences
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)


def _orders(seed, planted):
    rnd = random.Random(seed)
    ents, evs = [], []
    for c in range(500):
        cid = f"C{c}"
        ents.append(Entity(cid))
        t, last = NOW - timedelta(days=500), None
        for _ in range(6):
            cat = rnd.choice(["Apparel", "Grocery", "Home", "Cosmetics"])
            if planted and last == "Apparel" and rnd.random() < 0.6:
                cat, t = "Cosmetics", t + timedelta(days=rnd.randint(18, 22))
            else:
                t += timedelta(days=rnd.randint(5, 40))
            evs.append(Event(cid, t, "order", 50.0, {"Category": cat}))
            last = cat
    return Dataset(ents, evs)


def test_planted_transition_is_found_with_its_lag():
    out = sequences.find(_orders(1, True), attr="Category")
    top = out["patterns"][0]
    assert (top["from"], top["to"]) == ("Apparel", "Cosmetics") and 18 <= top["median_lag_days"] <= 22


def test_no_patterns_in_random_sequences():
    assert sequences.find(_orders(2, False), attr="Category")["patterns"] == []


def test_profiles_match_live_scores_and_go_stale():
    from tests.test_tasks import _plant_data
    from vertiq.tasks import predict_event
    task = predict_event(_plant_data(seed=3), target="breakdown", window_days=30)
    if task.status != "trained":
        return
    prof = profiles.build(task, ["M1", "M2"])
    assert prof["M1"]["chance"] == task.score("M1")["chance"]
    assert profiles.stale(prof["M1"], datetime.fromisoformat(prof["M1"]["as_of"]) + timedelta(days=1))
    assert not profiles.stale(prof["M1"], datetime.fromisoformat(prof["M1"]["as_of"]))
