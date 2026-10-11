"""The general engine on non-retail data: machines across plants (event prediction, a plugin feature
that carries domain knowledge, declarative rules) and patients (value prediction). Each dataset has
a planted truth; each test checks Vertiq finds it, and that the evaluation gate guards the result."""
import random
from datetime import datetime, timedelta

import pytest

from vertiq import plugins, rules
from vertiq.contract import Dataset, Entity, Event
from vertiq.tasks import predict_event, predict_value

NOW = datetime(2026, 9, 30)


def _plant_data(seed=3, machines=240):
    """Machines get preventive tasks every ~30 days; some are skipped. Each skip in the last 60 days
    triples the chance of a breakdown. Breakdowns are otherwise random."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(machines):
        mid = f"M{i}"
        ents.append(Entity(mid, group=f"plant-{i % 3}", attrs={"model": rnd.choice(["pump", "compressor"])}))
        skipper = rnd.random() < 0.35
        t = NOW - timedelta(days=720)
        skips = []
        while t < NOW:
            done = not (skipper and rnd.random() < 0.6)
            evs.append(Event(mid, t, "pm", 0.0, {"done": done}))
            if not done:
                skips.append(t)
            t += timedelta(days=30)
        d = NOW - timedelta(days=720)
        while d < NOW:
            recent = sum(1 for s in skips if timedelta(0) <= d - s <= timedelta(days=60))
            if rnd.random() < 0.02 * (3 ** min(recent, 2)):
                evs.append(Event(mid, d, "breakdown", rnd.uniform(500, 3000), {"code": rnd.choice(["SEAL", "BRG"]),
                                                                                "note": rnd.choice(["fixed", "", "replaced seal, shaft misaligned 0.3mm"])},
                                 items=(("seal kit", rnd.choice([1, 1, 1, 2])),)))
            d += timedelta(days=7)
    return Dataset(ents, evs)


@pytest.fixture(autouse=True)
def _clean():
    plugins.clear()
    yield
    plugins.clear()


def test_a_plugin_teaches_the_engine_what_the_domain_knows():
    ds = _plant_data()
    plain = predict_event(ds, target="breakdown", window_days=30)
    assert plain.status in ("trained", "kept")

    @plugins.feature("plants")
    def skipped_pm_60(entity, before, cut):          # domain knowledge: skipped maintenance matters
        return {"skipped_pm_60": sum(1 for e in before if e.type == "pm" and not e.attrs.get("done")
                                     and e.at >= cut - timedelta(days=60))}
    smart = predict_event(ds, target="breakdown", window_days=30, pack="plants")
    assert smart.status == "trained", smart.report()
    m = smart.report()["metrics"]
    assert m["auc"] > m["baseline_auc"] and m["entities_held_out"] > 20
    assert m["auc"] > (plain.report()["metrics"]["auc"] or 0) + 0.03          # the plugin carried the signal
    s = smart.score("M1")
    assert 0 <= s["chance"] <= 1 and (s["raises"] or s["lowers"])


def test_the_gate_keeps_the_baseline_when_there_is_nothing_to_learn():
    rnd = random.Random(9)
    ents = [Entity(f"E{i}") for i in range(200)]
    evs = [Event(f"E{i}", NOW - timedelta(days=rnd.randint(0, 700)), "ping") for i in range(200) for _ in range(6)]
    t = predict_event(Dataset(ents, evs), target="ping", window_days=60)
    assert t.status in ("kept", "trained")
    if t.status == "trained":                                                # if it ships, it really beat the rule
        assert t.report()["metrics"]["auc"] >= t.report()["metrics"]["baseline_auc"]


def test_too_little_data_says_so():
    ds = Dataset([Entity("A")], [Event("A", NOW - timedelta(days=10), "x")])
    assert predict_event(ds, target="x", window_days=30).status == "learning"


def test_value_prediction_for_a_different_domain():
    """Patients: length of stay grows with how many admissions they had in the last 90 days."""
    rnd = random.Random(5)
    ents, evs = [], []
    for i in range(300):
        pid = f"P{i}"
        ents.append(Entity(pid, group=f"hospital-{i % 2}"))
        t = NOW - timedelta(days=700 - rnd.randint(0, 60))
        while t < NOW:
            recent = sum(1 for e in evs if e.entity == pid and e.at >= t - timedelta(days=90))
            evs.append(Event(pid, t, "admission", max(1.0, rnd.gauss(2 + 2.5 * recent, 0.8))))
            t += timedelta(days=rnd.choice([20, 35, 60, 120]))
    task = predict_value(Dataset(ents, evs), target="admission", window_days=120)
    assert task.status == "trained", task.report()
    m = task.report()["metrics"]
    assert m["mae"] < m["baseline_mae"]
    assert task.score("P3")["value"] > 0


def test_rules_flag_repeats_overdue_overuse_and_empty_notes():
    ds = _plant_data(seed=4, machines=60)
    for e in list(ds.events):                                              # give pm events a due/done date
        if e.type == "pm":
            ds.events.remove(e)
            ds.events.append(Event(e.entity, e.at, "pm", 0.0, {"due": e.at, "done": e.at + timedelta(days=2) if e.attrs["done"] else None}))
    ds._by = None
    out = rules.run(ds, [rules.repeat(30, key="code", type="breakdown"),
                         rules.overdue("due", "done", tolerance_days=7, type="pm"),
                         rules.exceeds({"SEAL": {"seal kit": 1}, "BRG": {"seal kit": 1}}, key="code", type="breakdown"),
                         rules.too_short("note", 2, type="breakdown")])
    s = out["summary"]
    assert s["repeat"] > 0 and s["overdue"] > 0 and s["exceeds"] > 0 and s["too_short"] > 0
    f = next(f for f in out["flags"] if f["rule"] == "exceeds")
    assert f["evidence"] == {"item": "seal kit", "used": 2, "expected": 1}
    assert all(f["evidence"]["days_late"] > 7 for f in out["flags"] if f["rule"] == "overdue")


def test_learners_are_pluggable_and_checked():
    class NotALearner:
        pass
    with pytest.raises(TypeError):
        plugins.register_learner("bad", NotALearner)
    with pytest.raises(KeyError):
        plugins.learner("nope")


def test_a_minimal_custom_learner_runs_through_the_gate():
    """Only fit, predict, to_dict and from_dict are required; offsets and reasons are optional."""
    class MeanRate:
        def fit(self, rows, y, **kw):
            self.p = sum(y) / len(y) if y else 0.0
            self.w = {"since_target": -0.2}
            return self

        def predict(self, row, group=None):
            z = self.w["since_target"] * row.get("since_target", 0.0)
            return max(0.0, min(1.0, self.p + z / 10))

        def to_dict(self):
            return {"p": self.p}

        @classmethod
        def from_dict(cls, d):
            m = cls()
            m.p = d["p"]
            return m

    plugins.register_learner("mean_rate", MeanRate)
    task = predict_event(_plant_data(seed=6), target="breakdown", window_days=30, learner="mean_rate")
    assert task.status in ("trained", "kept")
    if task.status == "trained":
        s = task.score("M1")
        assert 0 <= s["chance"] <= 1 and s["raises"] == [] and s["lowers"] == []
