from datetime import datetime, timedelta

import pytest

from vertiq import menu
from vertiq.contract import Dataset, Entity, Event

NOW = datetime(2026, 9, 30)


def _ds():
    ents = [Entity("A", group="north", attrs={"age": 7}), Entity("B", group="north", attrs={"age": 2}), Entity("C", group="south", attrs={"age": 4})]
    evs = []
    for eid in "ABC":
        for k in range(12):
            evs.append(Event(eid, NOW - timedelta(days=30 * k + 3), "pm", 0.0, {"done": not (eid == "A" and k % 2)}))
    evs += [Event("A", NOW - timedelta(days=10), "breakdown", 900.0), Event("B", NOW - timedelta(days=40), "breakdown", 300.0)]
    return Dataset(ents, evs)


def test_propose_lists_rare_conditions_first():
    specs = menu.propose(_ds(), target="breakdown", window_days=30)
    names = [menu.name(s) for s in specs]
    assert names[0].startswith("count:pm[done=False]")       # the rarer value first
    assert any(n.startswith("count:pm[done=True]") for n in names)
    assert "attr:age" in names and "group_rate:breakdown@180d" in names


def test_features_use_only_the_past_and_compute_the_spec():
    ds = _ds()
    f = menu.features([{"op": "count", "type": "pm", "where": {"done": False}, "days": 60},
                       {"op": "since", "type": "breakdown"}, {"op": "attr", "name": "age"},
                       {"op": "group_rate", "type": "breakdown", "days": 90}], ds)
    a = ds.entities[0]
    cut = NOW - timedelta(days=20)
    before = [e for e in ds.events if e.entity == "A" and e.at < cut]
    vals = {k: list(fn(a, before, cut).values())[0] for k, fn in f.items()}
    assert vals["count:pm[done=False]@60d"] == 1.0          # one skipped visit in the 60 days before the cutoff
    assert vals["since:breakdown"] > 4                       # the breakdown 10 days ago is after the cutoff
    assert vals["attr:age"] == 7.0
    assert vals["group_rate:breakdown@90d"] == 1.0          # B, A's group peer, had one breakdown in the window


def test_parse_rejects_anything_outside_the_menu():
    for bad in ({"op": "exec", "code": "import os"}, {"op": "count", "type": "pm"}, {"op": "share", "type": "pm", "days": 30},
                {"op": "sum", "type": "pm", "where": {"done": True}, "days": 30}, {"op": "count", "type": "pm", "days": 99999}):
        with pytest.raises(ValueError):
            menu.parse([bad])
