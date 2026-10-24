"""Candidate features from a fixed menu, for the gated search in vertiq.autotune.

A feature is a small declarative spec, not code, so every candidate can be read and reviewed, and every
one uses only events recorded before the cutoff:

  {"op": "count",    "type": T, "where": {attr: value} | None, "days": W}   events of type T in the last W days
  {"op": "since",    "type": T, "where": ...}                               days since the last such event
  {"op": "share",    "type": T, "where": {attr: value}, "days": W}          share of T events meeting the condition
  {"op": "sum",      "type": T, "days": W}                                  total value of T events in the window
  {"op": "group_rate", "type": T, "days": W}                                T events per entity in the same group,
                                                                            excluding the entity itself
  {"op": "attr",     "name": A}                                             an entity attribute (number or yes/no)

A condition can also name an item on the event's line items: {"@item": "Jeans"} holds when the event includes it.

`propose(ds, target=...)` lists candidates from the data: the most frequent event types, attribute values
that occur often enough to matter, and windows scaled to the prediction window. `parse(specs)` checks
specs from another source, such as a language model asked which menu items suit a sector, and rejects
anything outside the menu. `build(spec)` returns the plugin function autotune expects.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import timedelta

from .contract import Dataset

OPS = {"count", "since", "share", "sum", "group_rate", "attr"}
MAX_VALUES = 6          # an attribute with more distinct values is not used as a condition
MIN_SHARE = 0.05        # a condition must hold for at least 5% of an event type's events


def name(spec: dict) -> str:
    op = spec["op"]
    if op == "attr":
        return f"attr:{spec['name']}"
    where = "".join(f"[{k}={v}]" for k, v in sorted((spec.get("where") or {}).items()))
    days = f"@{spec['days']}d" if "days" in spec else ""
    return f"{op}:{spec['type']}{where}{days}"


ITEM_KEY = "@item"      # a condition on the event's line items, not on an attribute
ITEMS = 8               # at most this many of the most common items become conditions


def words(feature: str) -> str:
    """A menu feature's name in words: "share:bill[@item=Jeans]@180d" -> "share of bills including Jeans over
    180 days"; "since:bill[payment_mode=Cash]" -> "time since the last bill with payment mode Cash"."""
    import re
    from .patterns import _name
    m = re.match(r"^(\w+):([^\[@]*)((?:\[[^\]]*\])*)(?:@(\d+)d)?$", feature)
    if not m:
        return feature.replace("_", " ")
    op, t, conds, days = m.group(1), m.group(2) or "event", m.group(3), m.group(4)
    if op == "attr":
        return _name(feature.split(":", 1)[1])
    parts = []
    for k, v in re.findall(r"\[([^=\]]+)=([^\]]*)\]", conds):
        parts.append(f"including {v}" if k == ITEM_KEY else f"with {_name(k)} {v}")
    cond = (" " + " and ".join(parts)) if parts else ""
    if op == "count":
        return f"{t}s{cond} in the last {days} days"
    if op == "share":
        return f"share of {t}s{cond} over {days} days"
    if op == "since":
        return f"time since the last {t}{cond}"
    if op == "sum":
        return f"value of {t}s in the last {days} days"
    if op == "group_rate":
        return f"{t}s per entity in the same group over {days} days"
    return feature


def _matches(e, where: dict | None) -> bool:
    if not where:
        return True
    for k, v in where.items():
        if k == ITEM_KEY:
            if not any(n == v for n, _q in (e.items or ())):
                return False
        elif str((e.attrs or {}).get(k)) != str(v):
            return False
    return True


def build(spec: dict, ds: Dataset | None = None):
    """The plugin function for a spec: fn(entity, before, cut) -> {name: number}."""
    spec = parse([spec])[0]
    key, op = name(spec), spec["op"]
    if op == "attr":
        def fn(entity, before, cut):
            v = entity.attrs.get(spec["name"], 0)
            return {key: float(v) if isinstance(v, (int, float)) else (1.0 if str(v).lower() in ("1", "true", "yes") else 0.0)}
        return fn
    t, where, days = spec["type"], spec.get("where"), spec.get("days")
    if op == "group_rate":
        if ds is None:
            raise ValueError("group_rate needs the dataset")
        groups = {e.id: e.group for e in ds.entities}
        members: dict[str, list[str]] = defaultdict(list)
        for e in ds.entities:
            members[e.group].append(e.id)
        by_entity: dict[str, list] = defaultdict(list)
        for ev in ds.events:
            if ev.type == t:
                by_entity[ev.entity].append(ev.at)

        def fn(entity, before, cut):
            peers = [m for m in members.get(groups.get(entity.id, ""), []) if m != entity.id]
            if not peers:
                return {key: 0.0}
            lo = cut - timedelta(days=days)
            n = sum(1 for m in peers for at in by_entity.get(m, ()) if lo <= at < cut)
            return {key: n / len(peers)}
        return fn

    def fn(entity, before, cut):
        evs = [e for e in before if e.type == t]
        if op == "since":
            hits = [e.at for e in evs if _matches(e, where)]
            return {key: math.log1p((cut - max(hits)).days) if hits else math.log1p(9999)}
        lo = cut - timedelta(days=days)
        recent = [e for e in evs if e.at >= lo]
        if op == "count":
            return {key: float(sum(1 for e in recent if _matches(e, where)))}
        if op == "share":
            return {key: sum(1 for e in recent if _matches(e, where)) / len(recent) if recent else 0.0}
        return {key: math.log1p(max(0.0, sum(float(e.value or 0) for e in recent)))}          # sum
    return fn


def parse(specs: list[dict]) -> list[dict]:
    """Specs checked against the menu; raises ValueError for anything outside it."""
    out = []
    for s in specs:
        if not isinstance(s, dict) or s.get("op") not in OPS:
            raise ValueError(f"not a menu item: {s!r}")
        op = s["op"]
        if op == "attr":
            if not isinstance(s.get("name"), str):
                raise ValueError(f"attr needs a name: {s!r}")
            out.append({"op": "attr", "name": s["name"]})
            continue
        if not isinstance(s.get("type"), str):
            raise ValueError(f"{op} needs an event type: {s!r}")
        clean = {"op": op, "type": s["type"]}
        where = s.get("where")
        if where is not None:
            if op in ("sum", "group_rate") or not isinstance(where, dict) or not where:
                raise ValueError(f"invalid condition for {op}: {s!r}")
            clean["where"] = {str(k): v for k, v in where.items()}
        if op == "share" and "where" not in clean:
            raise ValueError(f"share needs a condition: {s!r}")
        if op != "since":
            d = s.get("days")
            if not isinstance(d, int) or not 1 <= d <= 3650:
                raise ValueError(f"{op} needs days between 1 and 3650: {s!r}")
            clean["days"] = d
        out.append(clean)
    return out


def propose(ds: Dataset, *, target: str, window_days: int, types: int = 4, limit: int = 24) -> list[dict]:
    """Menu items suggested by the data, most promising kinds first, at most `limit`. Features the
    tasks already compute (counts of each type over 90 days, time since the target) are not repeated."""
    counts = Counter(e.type for e in ds.events)
    top = [t for t, _n in counts.most_common(types)]
    windows = sorted({max(7, window_days), max(14, 2 * window_days), max(30, 6 * window_days)})
    specs: list[dict] = []
    rarity: dict[str, float] = {}             # share of the type's events meeting each condition
    for t in top:
        evs = [e for e in ds.events if e.type == t]
        values: dict[str, Counter] = defaultdict(Counter)
        for e in evs:
            for k, v in (e.attrs or {}).items():
                if isinstance(v, (str, bool, int)) and not isinstance(v, float):
                    values[k][str(v)] += 1
        items = Counter(n for e in evs for n in {n for n, _q in (e.items or ())})
        if items:
            values[ITEM_KEY] = Counter(dict(items.most_common(ITEMS)))
        for k, c in values.items():
            if len(c) > MAX_VALUES and k != ITEM_KEY:
                continue
            for v, n in c.items():
                if n < MIN_SHARE * len(evs) or n == len(evs):
                    continue
                where = {k: v}
                rarity[name({"op": "count", "type": t, "where": where, "days": 0})[:-3]] = n / len(evs)
                for d in windows:
                    specs.append({"op": "count", "type": t, "where": where, "days": d})
                specs.append({"op": "share", "type": t, "where": where, "days": windows[-1]})
                specs.append({"op": "since", "type": t, "where": where})
        if t != target:
            specs.append({"op": "since", "type": t})
        specs.append({"op": "count", "type": t, "days": windows[0]})
        if any(e.value for e in evs):
            specs.append({"op": "sum", "type": t, "days": windows[-1]})
    if len({e.group for e in ds.entities}) > 1:
        specs.append({"op": "group_rate", "type": target, "days": windows[-1]})
    attrs = Counter(k for e in ds.entities for k, v in (e.attrs or {}).items() if isinstance(v, (int, float, bool)))
    specs += [{"op": "attr", "name": k} for k, n in attrs.items() if n >= 0.5 * len(ds.entities)]
    # conditional items carry the information the built-in features lack, so they go first, rarer
    # conditions (a skipped visit, a failed check) before common ones
    def order(sp):
        if sp.get("where"):
            cond = "".join(f"[{k}={v}]" for k, v in sorted(sp["where"].items()))
            return (0, rarity.get(f"count:{sp['type']}{cond}", 1.0))
        return (1, 0.0) if sp["op"] in ("attr", "group_rate") else (2, 0.0)
    specs.sort(key=order)
    seen, out = set(), []
    for s in specs:
        if name(s) not in seen:
            seen.add(name(s))
            out.append(s)
    return out[:limit]


def features(specs: list[dict], ds: Dataset | None = None) -> dict:
    """{name: plugin function} for autotune.search(features=...)."""
    return {name(s): build(s, ds) for s in parse(specs)}
