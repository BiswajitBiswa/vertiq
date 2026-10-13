"""Drift: stable data reads as stable, a shifted feature or degraded accuracy recommends retraining."""
import random

from vertiq import drift


def _rows(n, seed, shift=0.0):
    rnd = random.Random(seed)
    return [{"x": rnd.gauss(shift, 1), "y": rnd.gauss(0, 1)} for _ in range(n)]


def test_same_distribution_is_stable():
    assert drift.check(_rows(2000, 1), _rows(2000, 2))["verdict"] == "stable"


def test_a_shifted_feature_is_flagged_and_named():
    out = drift.check(_rows(2000, 1), _rows(2000, 2, shift=1.0))
    assert out["verdict"] == "retrain" and out["features"][0]["feature"] == "x" and out["features"][0]["level"] == "major"


def test_degraded_accuracy_recommends_retraining():
    rnd = random.Random(3)
    outcomes = [rnd.random() < 0.3 for _ in range(1000)]
    noisy = [rnd.random() for _ in outcomes]                       # scores no longer track outcomes
    perf = drift.performance(noisy, [int(o) for o in outcomes], reference_auc=0.78)
    assert perf["degraded"] and drift.check(_rows(500, 4), _rows(500, 5), perf=perf)["verdict"] == "retrain"


def test_task_rows_at_feed_drift_checks():
    from datetime import timedelta
    from tests.test_tasks import NOW, _plant_data
    from vertiq.tasks import predict_event
    task = predict_event(_plant_data(seed=3), target="breakdown", window_days=30)
    then, now = task.rows_at(NOW - timedelta(days=180)), task.rows_at(NOW)
    out = drift.check(list(then.values()), list(now.values()))
    assert out["verdict"] in ("stable", "watch", "retrain") and out["features"]
