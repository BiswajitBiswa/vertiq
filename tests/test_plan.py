import random
from datetime import datetime, timedelta

from vertiq import plan, plugins
from vertiq.contract import Action, Dataset, Entity, Event
from vertiq.tasks import predict_uplift

NOW = datetime(2026, 9, 30)


def test_allocate_respects_budget_and_one_action_per_entity():
    cands = [{"entity": "A", "action": "call", "effect": -0.20, "cost": 10, "value": -500},     # benefit 90
             {"entity": "A", "action": "visit", "effect": -0.30, "cost": 40, "value": -500},    # benefit 110
             {"entity": "B", "action": "call", "effect": -0.05, "cost": 10, "value": -500},     # benefit 15
             {"entity": "C", "action": "call", "effect": 0.02, "cost": 10, "value": -500}]      # harmful: never chosen
    r = plan.allocate(cands, budget=20)
    assert [(a["entity"], a["action"]) for a in r["assignments"]] == [("A", "call"), ("B", "call")]
    assert r["spent"] == 20 and r["expected_benefit"] == 105
    r = plan.allocate(cands, budget=50)
    assert ("A", "visit") in [(a["entity"], a["action"]) for a in r["assignments"]] and r["spent"] <= 50
    assert all(a["entity"] != "C" for a in r["assignments"])


def test_from_uplift_targets_the_entities_the_action_helps():
    rnd = random.Random(2)
    ents, evs, acts = [], [], []
    for i in range(1500):
        mid, old = f"M{i}", rnd.random() < 0.5
        ents.append(Entity(mid, attrs={"old": old}))
        evs.append(Event(mid, NOW - timedelta(days=800), "install"))
        when, treated = NOW - timedelta(days=45), rnd.random() >= 0.3
        acts.append(Action(mid, when, treated, "programme"))
        if rnd.random() < (0.25 if (treated and old) else (0.5 if old else 0.2)):
            evs.append(Event(mid, when + timedelta(days=rnd.randint(1, 29)), "failure"))
    plugins.clear()

    @plugins.feature("fleet")
    def age(entity, before, cut):
        return {"old": 1.0 if entity.attrs.get("old") else 0.0}
    t = predict_uplift(Dataset(ents, evs, acts), target="failure", window_days=30, pack="fleet")
    ids = [e.id for e in ents[:200]]
    r = plan.from_uplift({"programme": t}, ids, costs={"programme": 50}, value=-1000, budget=50 * 60)
    plugins.clear()
    old = {e.id for e in ents if e.attrs["old"]}
    picked = [a["entity"] for a in r["assignments"]]
    assert r["sources"]["programme"]["source"] in ("individual", "average")
    assert len(picked) <= 60 and sum(p in old for p in picked) >= 0.8 * len(picked)
