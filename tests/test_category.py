import random
from datetime import datetime, timedelta

import pytest

from vertiq.category import predict_category
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)


def _fleet(n=600, seed=6):
    """Some machines need attention far more often than others; the more repairs in the last 90 days, the
    likelier the next outcome is a replacement rather than another repair."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(n):
        mid, d, recent = f"M{i}", NOW - timedelta(days=720), []
        rate = rnd.choice([0.02, 0.15])
        ents.append(Entity(mid))
        evs.append(Event(mid, d - timedelta(days=5), "install"))
        while d < NOW:
            recent = [r for r in recent if (d - r).days <= 90]
            if rnd.random() < rate:
                if rnd.random() < min(0.9, 0.05 + 0.3 * len(recent)):
                    evs.append(Event(mid, d, "replace"))
                    recent = []
                else:
                    evs.append(Event(mid, d, "repair"))
                    recent.append(d)
            d += timedelta(days=7)
    return Dataset(ents, evs)


def test_category_beats_the_past_mix_and_scores_sum_to_one():
    t = predict_category(_fleet(), types=["repair", "replace"], window_days=60)
    m = t.report()["metrics"]
    assert m["macro_auc"] > m["baseline_macro_auc"]
    assert t.status == "trained"
    s = t.score("M3")
    assert abs(sum(s["probabilities"].values()) - 1) < 1e-3 and s["most_likely"] in ("repair", "replace", "none")


def test_category_needs_two_types():
    with pytest.raises(ValueError):
        predict_category(_fleet(50), types=["repair"], window_days=60)
