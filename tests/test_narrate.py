import random
from datetime import datetime, timedelta

from vertiq import narrate
from vertiq.contract import Dataset, Entity, Event
from vertiq.tasks import predict_event

NOW = datetime(2026, 9, 30)


def _machines(n=600, seed=1):
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(n):
        risk = rnd.choice([0.02, 0.06, 0.12])
        ents.append(Entity(f"M{i}"))
        d = NOW - timedelta(days=700)
        while d < NOW:
            if rnd.random() < risk:
                evs.append(Event(f"M{i}", d, "breakdown"))
            d += timedelta(days=7)
    return Dataset(ents, evs)


def test_templates_state_the_result():
    t = predict_event(_machines(), target="breakdown", window_days=30)
    text = narrate.gate(t.report())
    assert "breakdown" in text and "30 days" in text and f"{t.report()['metrics']['auc']:.3f}" in text
    assert ("was promoted" in text) == (t.status == "trained")
    eid = next(e.id for e in t.b.ds.entities if t.score(e.id))
    line = narrate.prediction(t.score(eid), target="breakdown", window_days=30)
    assert line.startswith(f"{eid}: ") and "% chance of breakdown within 30 days" in line


def test_learning_and_rejection_are_stated():
    tiny = Dataset([Entity("A")], [Event("A", NOW, "x")])
    assert narrate.gate(predict_event(tiny, target="x", window_days=30).report()).startswith("Not enough data yet")


def test_check_accepts_rounded_facts_and_rejects_invented_numbers():
    facts = narrate.facts({"auc": 0.8117, "baseline_auc": 0.7615, "held_out": 1176, "rate": 0.2341})
    assert narrate.check("AUC 0.81 against 0.76 on 1,176 customers; 23% ordered again.", facts) == []
    assert narrate.check("AUC 0.81, about 5 points better.", facts) == ["the number 5 is not in the facts"]
    assert narrate.check("Machine C123 and model v2 scored 0.812.", facts) == []      # ids are not numbers


def test_describe_falls_back_to_the_template():
    report = {"status": "trained", "task": "event", "target": "order", "window_days": 60,
              "metrics": {"auc": 0.8117, "baseline_auc": 0.7615, "entities_held_out": 1176}}
    text = narrate.gate(report)
    assert narrate.describe(text) == {"text": text, "source": "template", "problems": []}
    good = narrate.describe(text, generate=lambda p: "The new model (AUC 0.812) beats the rule (0.761) on 1,176 entities.")
    assert good["source"] == "model"
    bad = narrate.describe(text, generate=lambda p: "Orders will rise 15% next quarter.")
    assert bad["source"] == "template" and bad["text"] == text and bad["problems"]
    long = narrate.describe(text, generate=lambda p: text + " " + text)
    assert long["source"] == "template" and "longer than the template" in long["problems"][-1]

    def broken(_p):
        raise RuntimeError("model not loaded")
    assert narrate.describe(text, generate=broken)["problems"] == ["model error: RuntimeError"]


def test_rejection_reasons_are_plain():
    report = {"status": "kept", "task": "event", "target": "order", "window_days": 60,
              "reason": "AUC 0.45 does not exceed the baseline's 0.55 by 0.0; the gain over the baseline is not clear at 90% confidence",
              "metrics": {"auc": 0.45, "baseline_auc": 0.55, "entities_held_out": 77, "min_gain": 0.0,
                          "intervals": {"level": 0.9, "gain_ci": [-0.26, 0.06]}}}
    text = narrate.gate(report)
    assert text.endswith("It does not beat the baseline rule and shows no clear gain, since that range includes zero, "
                         "so the current model or rule stays in use.")
    assert "0.0;" not in text and "Reason" not in text
