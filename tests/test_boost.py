"""Boosted trees: they learn an interaction a linear model cannot, explain each score, round-trip as a
plain dict, and run through the same evaluation gate as every other learner."""
import random

from vertiq.boost import BoostedClassifier, BoostedRegressor
from vertiq.learn import Logistic, auc


def _xor(n, seed):
    rnd = random.Random(seed)
    rows, y = [], []
    for _ in range(n):
        a, b, noise = rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        rows.append({"a": a, "b": b, "noise": noise})
        p = 0.85 if (a > 0) != (b > 0) else 0.15
        y.append(1 if rnd.random() < p else 0)
    return rows, y


def test_boosted_learns_an_interaction_a_linear_model_cannot():
    rows, y = _xor(1500, 1)
    test_rows, test_y = _xor(800, 2)
    lin = Logistic().fit(rows, y)
    bst = BoostedClassifier().fit(rows, y)
    lin_auc = auc([lin.predict(r) for r in test_rows], test_y)
    bst_auc = auc([bst.predict(r) for r in test_rows], test_y)
    assert lin_auc < 0.6 < 0.75 < bst_auc


def test_explanations_and_storage():
    rows, y = _xor(800, 3)
    m = BoostedClassifier(trees=30).fit(rows, y)
    c = dict(m.contributions({"a": 0.8, "b": -0.7, "noise": 0.0}))
    assert abs(c.get("a", 0)) + abs(c.get("b", 0)) > abs(c.get("noise", 0))
    again = BoostedClassifier.from_dict(m.to_dict())
    r = {"a": 0.3, "b": 0.6, "noise": -0.2}
    assert again.predict(r) == m.predict(r)


def test_regression_fits_a_step():
    rnd = random.Random(4)
    rows = [{"x": rnd.uniform(0, 10)} for _ in range(600)]
    y = [5.0 if r["x"] > 6 else 1.0 for r in rows]
    m = BoostedRegressor().fit(rows, y)
    assert abs(m.predict({"x": 8.0}) - 5.0) < 0.6 and abs(m.predict({"x": 2.0}) - 1.0) < 0.6


def test_boosted_runs_through_the_gate():
    from tests.test_tasks import _plant_data
    from vertiq.tasks import predict_event
    task = predict_event(_plant_data(seed=8), target="breakdown", window_days=30, learner="boosted")
    assert task.status in ("trained", "kept")
    if task.status == "trained":
        s = task.score("M1")
        assert 0 <= s["chance"] <= 1


def test_auto_selects_on_training_entities_and_reports_the_choice():
    from tests.test_tasks import _plant_data
    from vertiq.tasks import predict_event
    task = predict_event(_plant_data(seed=8), target="breakdown", window_days=30, learner="auto")
    m = task.report()["metrics"]
    assert set(m["selection"]) == {"Logistic", "BoostedClassifier"}
    assert m["learner"] in m["selection"]
