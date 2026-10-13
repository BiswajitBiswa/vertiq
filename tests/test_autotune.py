import hashlib
import random
from datetime import datetime, timedelta

from vertiq import autotune
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)


def _world(n=400, seed=4):
    """Failure risk depends on an attribute that no built-in feature sees."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(n):
        hot = rnd.random() < 0.4
        ents.append(Entity(f"A{i}", attrs={"hot": hot}))
        d = NOW - timedelta(days=500)
        while d < NOW:
            if rnd.random() < (0.09 if hot else 0.02):
                evs.append(Event(f"A{i}", d, "failure"))
            d += timedelta(days=7)
        evs.append(Event(f"A{i}", NOW - timedelta(days=510), "install"))
    return Dataset(ents, evs)


def hot(entity, before, cut):
    return {"hot": 1.0 if entity.attrs["hot"] else 0.0}


def noise(entity, before, cut):
    return {"noise": int(hashlib.md5(f"{entity.id}{cut}".encode()).hexdigest(), 16) % 100 / 100}


def test_search_keeps_the_real_feature_and_rejects_noise():
    r = autotune.search(_world(), target="failure", window_days=30, features={"hot": hot, "noise": noise}, folds=3, rounds=1)
    assert r["features"] == ["hot"]
    kept = [t["change"] for t in r["trials"] if t["kept"]]
    assert kept == ["start", "+feature hot"]
    assert all("folds_improved" in t for t in r["trials"][1:])
    assert r["task"] in (r["tuned"], r["untuned"])
    if r["tuned_promoted"]:
        assert r["tuned"].status == "trained"
