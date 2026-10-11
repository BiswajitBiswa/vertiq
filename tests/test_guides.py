"""Every sector guide's code example runs: each is executed against generated rows in that guide's
own column names and date format, so the docs can't drift from the package."""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
from guides import GUIDES, snippet  # noqa: E402

from vertiq import plugins  # noqa: E402

NOW = datetime(2026, 9, 30)


def _types(g):
    found = {g["event"]["target"]} | ({g["value"]["target"]} if g["value"] else set())
    for r, _why in g["rules"]:
        if 'type="' in r:
            found.add(r.split('type="')[1].split('"')[0])
    body = g["plugin"]["body"]
    found |= {part.split('"')[0] for part in body.split('e.type == "')[1:]}
    return sorted(t for t in found if t)


def rows_for(g, entities=40, seed=1):
    rnd, m, out = random.Random(seed), g["mapping"], []
    types = _types(g) or [""]
    for i in range(entities):
        eid = g["event"]["id"] if i == 0 else f"E{i:04d}"
        t = NOW - timedelta(days=400 + rnd.randint(0, 40))
        while t < NOW:
            r = {m["entity"]: eid, m["time"]: t.strftime(m["date_format"])}
            if m["type"]:
                r[m["type"]] = rnd.choice(types)
            if m["value"]:
                r[m["value"]] = f"{rnd.uniform(1, 900):.2f}"
            if m["group"]:
                r[m["group"]] = f"G{i % 3}"
            for a in m["attrs"]:
                low = a.lower()
                if any(w in low for w in ("due", "done", "filed on", "paid", "submitted", "end")):
                    r[a] = (t + timedelta(days=rnd.randint(-3, 20))).date().isoformat()
                else:
                    r[a] = rnd.choice(["yes", "no", "late", "SEAL", "E11", "a"])
            out.append(r)
            t += timedelta(days=rnd.choice([5, 9, 14, 30]))
    return out


@pytest.mark.parametrize("g", GUIDES, ids=[g["slug"] for g in GUIDES])
def test_the_guide_example_runs(g):
    plugins.clear()
    env = {"rows": rows_for(g), "print": lambda *a, **k: None}
    exec(compile(snippet(g), f"<guide {g['slug']}>", "exec"), env)
    assert env["problems"] == [], env["problems"][:3]
    assert env["task"].status in ("learning", "kept", "trained")
    if g["rules"]:
        assert set(env["flags"]["summary"]) <= {r.split("(")[0] for r, _w in g["rules"]} | set(env["flags"]["summary"])
    plugins.clear()
