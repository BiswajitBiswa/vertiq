"""The gate's segment check: a model that wins on average but is clearly worse for one segment is not promoted,
and checking many segments does not block a model that is better everywhere."""
import random

from vertiq import harness, narrate, plugins


def _data(harm: bool, seed: int = 3, n_entities: int = 1500, extra_dims: int = 0):
    """Entities in segment A (85%) or B. The outcome follows x in A; in B it follows x too, unless `harm`, when it
    follows -x. The baseline knows nothing in A and, under `harm`, is right in B. A model trained mostly on A
    learns x, so under `harm` it wins on average and loses in B."""
    rnd = random.Random(seed)
    rows, y, keys, base, segs = [], [], [], [], []
    for e in range(n_entities):
        seg = "B" if rnd.random() < 0.15 else "A"
        extra = {f"dim{d}": rnd.choice(["p", "q", "r"]) for d in range(extra_dims)}
        for _cut in range(2):
            x = rnd.gauss(0, 1)
            sign = -1 if (harm and seg == "B") else 1
            outcome = 1 if sign * x + rnd.gauss(0, 0.6) > 0 else 0
            rows.append({"x": x})
            y.append(outcome)
            keys.append(f"E{e}")
            base.append(-x if (harm and seg == "B") else rnd.random())
            segs.append({"segment": seg, **extra})
    return rows, y, keys, base, segs


def _gate(harm: bool, **kw):
    rows, y, keys, base, segs = _data(harm, **kw)
    return harness.evaluate(rows, y, keys, [""] * len(y), base, plugins.learner("linear", "classify"), kind="classify",
                            segments=segs)


def test_a_model_that_harms_one_segment_is_not_promoted():
    res = _gate(harm=True)
    b = next(s for s in res["metrics"]["segments"] if s["label"] == "B")
    assert res["metrics"]["auc"] > res["metrics"]["baseline_auc"]          # it wins on average ...
    assert b["worse"] and res["status"] == "kept"                         # ... and loses in B, so it is kept
    assert "segment B" in res["reason"]
    text = narrate.gate({"status": res["status"], "metrics": res["metrics"], "reason": res["reason"],
                         "task": "event", "target": "purchase", "window_days": 30})
    assert "clearly worse than the baseline rule for segment B" in text


def test_checking_many_segments_does_not_block_a_model_better_everywhere():
    res = _gate(harm=False, extra_dims=6)
    assert res["status"] == "trained"
    assert len(res["metrics"]["segments"]) >= 10 and not any(s["worse"] for s in res["metrics"]["segments"])
    text = narrate.gate({"status": "trained", "metrics": res["metrics"], "task": "event", "target": "purchase", "window_days": 30})
    assert "not clearly worse than the baseline rule in any of them" in text


def test_small_segments_are_not_judged():
    rows, y, keys, base, segs = _data(harm=False, n_entities=1500)
    segs = [dict(s, tiny="rare" if k in ("E1", "E2", "E3") else "") for s, k in zip(segs, keys)]
    res = harness.evaluate(rows, y, keys, [""] * len(y), base, plugins.learner("linear", "classify"), kind="classify", segments=segs)
    assert not any(s["dimension"] == "tiny" for s in res["metrics"]["segments"])
