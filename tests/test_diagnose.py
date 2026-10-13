import random

from vertiq import diagnose


def _records(seed=1):
    rnd, recs = random.Random(seed), []
    for per, shares in (("before", {"A": 0.5, "B": 0.3, "C": 0.2}), ("after", {"A": 0.3, "B": 0.3, "C": 0.4})):
        for g, s in shares.items():
            rate = {"A": 0.05, "B": 0.10 if per == "before" else 0.20, "C": 0.15}[g]
            recs += [{"group": g, "period": per, "outcome": 1 if rnd.random() < rate else 0} for _ in range(int(4000 * s))]
    return recs


def test_change_splits_rate_and_mix_exactly_and_names_the_driver():
    r = diagnose.change(_records())
    assert abs(r["rate_effects"] + r["mix_effects"] - r["change"]) < 1e-4
    assert r["drivers"] == ["B"]                                     # only plant B's own rate changed
    c = next(g for g in r["groups"] if g["group"] == "C")
    assert c["mix_effect"] > 0.02 and not c["significant"]           # C grew in share; its rate did not change
    assert diagnose.change([{"group": "A", "period": "before", "outcome": 1}])["status"] == "insufficient_data"
