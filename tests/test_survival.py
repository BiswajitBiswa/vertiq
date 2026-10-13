"""Time-to-next-event estimation recovers known probabilities under independent censoring, and runs as a
task feature that the gate evaluates."""
import math
import random
from datetime import datetime, timedelta

from vertiq import survival
from vertiq.contract import Dataset, Entity, Event


def test_kaplan_meier_recovers_a_known_curve_with_censoring():
    rnd = random.Random(3)
    d, o = [], []
    for _ in range(4000):
        x, c = rnd.expovariate(1 / 12), rnd.uniform(0, 60)       # censoring independent of the gap
        d.append(round(min(x, c)))
        o.append(x <= c)
    curve = survival.kaplan_meier(d, o)
    theory = 1 - math.exp(-7 / 12)
    assert abs(survival.within(curve, 0, 7) - theory) < 0.04
    assert abs(survival.within(curve, 12, 7) - theory) < 0.04          # memoryless
    assert abs(survival.within(curve, 0, 14, scale=2) - theory) < 0.04  # a twice-as-slow entity


def test_open_gaps_are_censored_at_the_cutoff():
    t0 = datetime(2026, 1, 1)
    by = {"A": [Event("A", t0), Event("A", t0 + timedelta(days=10))]}
    d, o = survival.gaps(by, t0 + timedelta(days=25))
    assert sorted(zip(d, o)) == [(10.0, True), (15.0, False)]


def test_survival_feature_runs_in_a_task():
    from vertiq.tasks import predict_event
    rnd = random.Random(5)
    now = datetime(2026, 9, 30)
    ents, evs = [], []
    for i in range(300):
        gap = rnd.choice([7, 14, 28])
        ents.append(Entity(f"C{i}"))
        t = now - timedelta(days=500)
        while t < now:
            evs.append(Event(f"C{i}", t, "order", 100.0))
            t += timedelta(days=max(1, round(rnd.expovariate(1 / gap))))
    task = predict_event(Dataset(ents, evs), target="order", window_days=14, survival=True)
    assert task.status in ("trained", "kept")
    row = task.b.row("C1", now, task.b.cadence(now, 14))
    assert 0.0 <= row["km_within"] <= 1.0
