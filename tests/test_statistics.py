"""Matched-cohort effects, sequential tests and dependence-robust FDR control."""
import random

from vertiq.causal import matched_att
from vertiq.learn import benjamini_hochberg, benjamini_yekutieli
from vertiq.sequential import SequentialTest


def test_matching_recovers_an_effect_under_observed_confounding():
    rnd = random.Random(1)
    recs = []
    for _ in range(800):
        x = rnd.gauss(0, 1)
        t = 1 if rnd.random() < 1 / (1 + 2.718 ** (-1.5 * x)) else 0          # treated more often when x is high
        recs.append({"t": t, "x": x, "y": 2.0 * x + 1.0 * t + rnd.gauss(0, 0.3)})
    r = matched_att(recs, treatment="t", outcome="y", features=["x"])
    naive = (sum(z["y"] for z in recs if z["t"]) / sum(z["t"] for z in recs)
             - sum(z["y"] for z in recs if not z["t"]) / sum(1 - z["t"] for z in recs))
    assert abs(r["att"] - 1.0) < 0.15 < abs(naive - 1.0)
    assert r["interval_90"][0] <= r["att"] <= r["interval_90"][1]
    assert abs(r["balance_after"]["x"]) < abs(r["balance_before"]["x"])


def test_sequential_test_stops_early_and_controls_false_alarms():
    rnd = random.Random(2)
    false_alarms, sizes = 0, []
    for _ in range(400):                                                      # true rate = theta0
        s = SequentialTest(0.1, 0.15, alpha=0.05, beta=0.2)
        d = s.run(1 if rnd.random() < 0.1 else 0 for _ in range(5000))
        false_alarms += d == "accept_h1"
        sizes.append(s.n)
    assert false_alarms / 400 < 0.08
    s = SequentialTest(0.1, 0.2)
    assert s.run(1 if rnd.random() < 0.3 else 0 for _ in range(5000)) == "accept_h1"
    lower = SequentialTest(0.2, 0.1)                                          # an improvement can be a lower rate
    assert lower.run(1 if rnd.random() < 0.05 else 0 for _ in range(5000)) == "accept_h1"


def test_sequential_evidence_can_block_but_not_force_promotion():
    from tests.test_tasks import _plant_data
    from vertiq.tasks import predict_event
    no = {"decision": "accept_h0", "n": 300}
    t = predict_event(_plant_data(seed=3), target="breakdown", window_days=30, evidence=no)
    assert t.status == "kept" and "sequential test found no improvement" in t.report()["reason"]


def test_by_is_more_conservative_than_bh():
    ps = [0.001, 0.008, 0.012, 0.02, 0.03, 0.2, 0.5, 0.8]
    assert sum(benjamini_yekutieli(ps)) <= sum(benjamini_hochberg(ps))
    assert benjamini_yekutieli([]) == []


def test_two_sample_sequential_keeps_false_alarms_and_detects():
    import random
    from vertiq.sequential import TwoSampleTest
    rnd = random.Random(5)

    def stream(pa, pb, n=6000):
        for _ in range(n):
            yield "a", 1 if rnd.random() < pa else 0
            yield "b", 1 if rnd.random() < pb else 0
    false = sum(TwoSampleTest(0.05).run(stream(0.1, 0.1)) == "accept_h1" for _ in range(200))
    found = sum(TwoSampleTest(0.05).run(stream(0.1, 0.15)) == "accept_h1" for _ in range(100))
    assert false <= 200 * 0.05 * 1.6           # checked after every pair, still about alpha
    assert found >= 80
    t = TwoSampleTest(0.05, max_n=500)
    assert t.run(stream(0.1, 0.1)) in ("accept_h0", "accept_h1") and t.state()["n_a"] <= 500


def test_doubly_robust_corrects_confounding_with_a_wrong_outcome_model():
    import math
    import random
    from vertiq.causal import doubly_robust
    rnd = random.Random(0)
    recs = []
    for _ in range(3000):
        x = rnd.gauss(0, 1)
        t = 1 if rnd.random() < 1 / (1 + math.exp(-1.2 * x)) else 0
        recs.append({"t": t, "x": x, "y": math.exp(x) + 1.0 * t + rnd.gauss(0, 0.5)})   # linear outcome models are wrong
    naive = (sum(r["y"] for r in recs if r["t"]) / sum(r["t"] for r in recs)
             - sum(r["y"] for r in recs if not r["t"]) / sum(1 - r["t"] for r in recs))
    r = doubly_robust(recs, treatment="t", outcome="y", features=["x"])
    assert naive > 2.0                                   # confounded
    assert r["status"] == "ok" and abs(r["ate"] - 1.0) < 0.2 and r["interval"][0] <= 1.0 <= r["interval"][1]
    assert doubly_robust(recs[:15], treatment="t", outcome="y", features=["x"])["status"] == "insufficient_data"
