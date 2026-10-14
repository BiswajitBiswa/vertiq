"""Patterns in any dataset, found without being told where to look.

Every record (event) is described by features taken from the data itself:

  attributes   each value of each event attribute with 2 to 12 distinct values ("Type = TakeAway"), and
               the event type when there are 2 to 12
  group        each entity group, when there are 2 to 12
  weekday      the day of the week, when records fall on more than one
  part of day  before noon, afternoon (12:00 to 17:00) or evening (17:00 on), when records carry a time
  item groups  "includes Pizza" for each item group (or item, when there are few) on the record's line items

and every comparison is tested:

  value  the record's value with a feature against without it (a normal approximation to Welch's test)
  rate   how often one feature occurs with another feature against without it (two proportions), for every
         pair of features from different sources

All p-values form one family and are filtered with Benjamini-Hochberg at `q` (10% by default), so the more
comparisons are made, the stronger a pattern must be to be reported. A pattern must also be large enough to
matter: a value gap of at least 10% of the overall average, or a rate gap of at least 10 percentage points.
Each side of a comparison needs at least `min_n` records. The search decides what to compare; nothing is
chosen by hand, and the sentences are written from the numbers.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from .contract import Dataset
from .learn import benjamini_hochberg, two_proportion_p

MAX_LEVELS = 12
ITEM_LIMIT = 40          # without item groups, items themselves are features only when there are this few


def _mean_p(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    if se == 0:
        return 1.0 if statistics.mean(a) == statistics.mean(b) else 0.0
    return math.erfc(abs(statistics.mean(a) - statistics.mean(b)) / se / math.sqrt(2))


def _part_of_day(h: int) -> str:
    return "before noon" if h < 12 else "afternoon" if h < 17 else "evening"


def features(ds: Dataset, item_groups: dict[str, str] | None = None, min_n: int = 10) -> tuple[list[dict], list[set]]:
    """The features found in the data ({"source", "label"}) and, for each event, the set of features it has.
    Entity attributes with 2 to 12 labels (from an entity table) count too, as "{name} = {label}"."""
    events = ds.events
    ent_attrs = {e.id: (e.attrs or {}) for e in ds.entities}
    ent_values: dict[str, Counter] = defaultdict(Counter)
    for e in ds.entities:
        for k, v in (e.attrs or {}).items():
            if isinstance(v, str) and v.strip() not in ("", "-"):
                ent_values[k][v.strip()] += 1
    n = len(events)
    groups = ds.groups()
    raw: list[set] = [set() for _ in events]
    attr_values: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        for k, v in (e.attrs or {}).items():
            if v not in (None, "", "-"):
                attr_values[k][str(v).strip()] += 1
    timed = any(e.at.hour or e.at.minute for e in events)
    weekdays = {e.at.strftime("%A") for e in events}
    item_names = {name for e in events for name, _q in (e.items or ())}
    use_items = bool(item_groups) or len(item_names) <= ITEM_LIMIT
    grp_levels = {groups.get(e.entity, "") for e in events} - {""}
    types = {e.type for e in events} - {""}
    for i, e in enumerate(events):
        if 2 <= len(types) <= MAX_LEVELS and e.type:
            raw[i].add(("type", f"type = {e.type}"))
        for k, c in attr_values.items():
            if 2 <= len(c) <= MAX_LEVELS:
                v = str((e.attrs or {}).get(k) or "").strip()
                if v in c:
                    raw[i].add((f"attr:{k}", f"{k} = {v}"))
        if 2 <= len(grp_levels) <= MAX_LEVELS and groups.get(e.entity):
            raw[i].add(("group", f"group {groups[e.entity]}"))
        for k, c in ent_values.items():
            if 2 <= len(c) <= MAX_LEVELS and k not in attr_values:
                v = str(ent_attrs.get(e.entity, {}).get(k) or "").strip()
                if v in c:
                    raw[i].add((f"entity:{k}", f"{k} = {v}"))
        if len(weekdays) > 1:
            raw[i].add(("weekday", e.at.strftime("%A")))
        if timed:
            raw[i].add(("part_of_day", _part_of_day(e.at.hour)))
        if use_items:
            for name, _q in (e.items or ()):
                g = (item_groups or {}).get(name, name)
                raw[i].add((f"item:{g}", f"includes {g}"))
    counts = Counter(f for s in raw for f in s)
    keep = {f for f, c in counts.items() if min_n <= c <= n - min_n}
    # a two-level attribute gives two complementary features; one of them carries all the information
    by_source: dict[str, list] = defaultdict(list)
    for f in keep:
        by_source[f[0]].append(f)
    for src, fs in by_source.items():
        levels = {f for s in raw for f in s if f[0] == src}
        if len(levels) == 2 and len(fs) == 2:
            keep.discard(sorted(fs, key=lambda f: f[1])[1])
    feats = sorted(keep)
    return [{"source": s, "label": lab} for s, lab in feats], [s & keep for s in raw]


def find(ds: Dataset, *, item_groups: dict[str, str] | None = None, noun: str = "records", currency: str = "",
         q: float = 0.1, min_n: int = 10, min_value_gap: float = 0.1, min_rate_gap: float = 0.1, limit: int = 20) -> dict:
    """{"status", "records", "features", "tested", "found": [{"kind", "text", "p_value", ...}], "not_clear"}."""
    events = ds.events
    n = len(events)
    if n < 2 * min_n:
        return {"status": "learning", "records": n, "need": 2 * min_n, "found": []}
    feats, has = features(ds, item_groups, min_n)
    keys = [(f["source"], f["label"]) for f in feats]
    values = [float(e.value or 0) for e in events]
    use_value = any(values)
    overall = statistics.mean(values) if use_value else 0.0
    tests = []
    if use_value:
        for k in keys:
            a = [v for v, s in zip(values, has) if k in s]
            b = [v for v, s in zip(values, has) if k not in s]
            tests.append({"kind": "value", "feature": k[1], "n_with": len(a), "n_without": len(b),
                          "with": statistics.mean(a), "without": statistics.mean(b), "p": _mean_p(a, b),
                          "big": abs(statistics.mean(a) - statistics.mean(b)) >= min_value_gap * abs(overall)})
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[0] == kb[0]:
                continue                      # levels of one attribute exclude each other by construction
            # context (weekday, part of day, group) is the condition; otherwise the first feature is
            cond, out = (kb, ka) if kb[0] in ("weekday", "part_of_day", "group") and ka[0] not in ("weekday", "part_of_day", "group") else (ka, kb)
            w = [out in s for s in has if cond in s]
            wo = [out in s for s in has if cond not in s]
            if len(w) < min_n or len(wo) < min_n or not (any(w) or any(wo)):
                continue
            r1, r0 = sum(w) / len(w), sum(wo) / len(wo)
            tests.append({"kind": "rate", "feature": cond[1], "outcome": out[1], "n_with": len(w), "n_without": len(wo),
                          "with": r1, "without": r0, "p": two_proportion_p(sum(w), len(w), sum(wo), len(wo)),
                          "big": abs(r1 - r0) >= min_rate_gap})
    if not tests:
        return {"status": "ok", "records": n, "features": len(keys), "tested": 0, "found": [], "not_clear": 0}
    flags = benjamini_hochberg([t["p"] for t in tests], q)
    found = [t for t, ok in zip(tests, flags) if ok and t["big"]]
    found.sort(key=lambda t: t["p"])
    for t in found:
        t["text"] = _sentence(t, noun, currency)
        t["p_value"] = round(t.pop("p"), 6)
        for k in ("with", "without"):
            t[k] = round(t[k], 4)
        t.pop("big", None)
    return {"status": "ok", "records": n, "features": len(keys), "tested": len(tests), "q": q,
            "found": found[:limit], "not_clear": len(tests) - len(found)}


def _where(label: str) -> str:
    """The feature as a condition on records: "that include Pizza", "where Type is TakeAway", "on Saturday"."""
    if label.startswith("includes "):
        return f"that include {label[9:]}"
    if " = " in label:
        k, v = label.split(" = ", 1)
        return f"where {k} is {v}"
    if label.startswith("group "):
        return f"in {label}"
    if label in ("before noon",):
        return label
    if label in ("afternoon", "evening"):
        return f"in the {label}"
    return f"on {label}"


def _does(label: str) -> str:
    """The feature as something a record does: "include Pizza", "have Type TakeAway", "fall on Saturday"."""
    if label.startswith("includes "):
        return f"include {label[9:]}"
    if " = " in label:
        k, v = label.split(" = ", 1)
        return f"have {k} {v}"
    if label.startswith("group "):
        return f"are in {label}"
    if label == "before noon":
        return "come before noon"
    if label in ("afternoon", "evening"):
        return f"come in the {label}"
    return f"fall on {label}"


def _sentence(t: dict, noun: str, cur: str) -> str:
    p = "p < 0.001" if t["p"] < 0.001 else f"p = {t['p']:.3f}"
    if t["kind"] == "value":
        return (f"{noun.capitalize()} {_where(t['feature'])} average {cur}{t['with']:,.0f}, against {cur}{t['without']:,.0f} "
                f"for the rest ({t['n_with']} vs {t['n_without']} {noun}; {p}).")
    return (f"{t['with']:.0%} of {noun} {_where(t['feature'])} {_does(t['outcome'])}, against {t['without']:.0%} of the rest "
            f"({t['n_with']} vs {t['n_without']} {noun}; {p}).")
