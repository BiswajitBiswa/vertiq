import random
from datetime import datetime, timedelta

from vertiq import guard, plugins
from vertiq.contract import Action, Dataset, Entity, Event
from vertiq.tasks import predict_uplift

NOW = datetime(2026, 9, 30)
TRUTH = {("call", True): -0.05, ("call", False): -0.10, ("overtime", True): 0.12, ("overtime", False): 0.12}


def _tasks():
    rnd = random.Random(8)
    ents, evs, acts = [], [], []
    for i in range(2000):
        mid, old = f"M{i}", rnd.random() < 0.5
        ents.append(Entity(mid, attrs={"old": old}))
        evs.append(Event(mid, NOW - timedelta(days=800), "install"))
        kind, treated, when = rnd.choice(["call", "overtime"]), rnd.random() >= 0.3, NOW - timedelta(days=45)
        acts.append(Action(mid, when, treated, kind))
        if rnd.random() < (0.5 if old else 0.2) + (TRUTH[(kind, old)] if treated else 0.0):
            evs.append(Event(mid, when + timedelta(days=rnd.randint(1, 29)), "failure"))
    ds = Dataset(ents, evs, acts)
    plugins.clear()

    @plugins.feature("fleet")
    def age(entity, before, cut):
        return {"old": 1.0 if entity.attrs.get("old") else 0.0}
    tasks = {k: predict_uplift(ds, target="failure", window_days=30, pack="fleet", kind=k) for k in ("call", "overtime")}
    plugins.clear()
    return tasks


def test_guard_blocks_harm_holds_the_unknown_and_respects_rules_and_limits():
    tasks = _tasks()
    harmful = guard.check({"entity": "M1", "action": "overtime"}, tasks)
    assert harmful["decision"] == "block" and harmful["evidence"]["interval"][0] > 0
    assert guard.check({"entity": "M1", "action": "replace_part"}, tasks)["decision"] == "hold"
    helpful = guard.check({"entity": "M2", "action": "call"}, tasks)
    assert helpful["decision"] in ("allow", "hold") and helpful["evidence"]["interval"][1] < 0
    flagged = guard.check({"entity": "M2", "action": "call"}, tasks, flags=[{"entity": "M2", "rule": "repeat"}])
    assert flagged["decision"] == "block"
    assert guard.check({"entity": "M2", "action": "call", "amount": 900}, tasks, limits={"call": 500})["decision"] == "hold"
    assert guard.check({"entity": "M2", "action": "call"}, tasks, goal="increase")["decision"] == "block"
