"""Patterns in any dataset, found without being told where to look.

Every record (event) is described by features taken from the data itself:

  attributes   each value of each event attribute with 2 to 12 distinct values ("Type = TakeAway"), and
               the event type when there are 2 to 12
  group        each entity group, when there are 2 to 12
  weekday      the day of the week, when records fall on more than one
  part of day  before noon, afternoon (12:00 to 17:00) or evening (17:00 on), when records carry a time
  item groups  "includes Pizza" for each item group (or item, without groups) on the record's line items;
               at most the 40 most common, so a catalogue of thousands of products stays one family of tests

and every comparison is tested:

  value  the record's value with a feature against without it (a normal approximation to Welch's test)
  rate   how often one feature occurs with another feature against without it (two proportions), for every
         pair of features from different sources, except two item groups (vertiq.affinity covers what is
         bought together)

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
ITEM_LIMIT = 40          # at most this many item groups (or items) are features: the most common ones


def _mean_p_sums(a: tuple, b: tuple) -> float:
    """Welch's test by a normal approximation, from (count, sum, sum of squares) of each side, values centred on the
    overall mean."""
    (na, sa, qa), (nb, sb, qb) = a, b
    if na < 2 or nb < 2:
        return 1.0
    va, vb = max(0.0, (qa - sa * sa / na) / (na - 1)), max(0.0, (qb - sb * sb / nb) / (nb - 1))
    se = math.sqrt(va / na + vb / nb)
    gap = abs(sa / na - sb / nb)
    if se == 0:
        return 1.0 if gap < 1e-12 else 0.0
    return math.erfc(gap / se / math.sqrt(2))


def _bitset(ix: list[int], n: int) -> int:
    buf = bytearray((n + 7) // 8)
    for i in ix:
        buf[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(buf, "little")


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
    use_items = bool(item_names)
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
    # chosen by how often they occur, never by outcome, so the false-discovery control still holds
    items = sorted((f for f in keep if f[0].startswith("item:")), key=lambda f: (-counts[f], f[1]))
    keep -= set(items[ITEM_LIMIT:])
    # a two-level attribute gives two complementary features; one of them carries all the information
    by_source: dict[str, list] = defaultdict(list)
    levels_of: dict[str, int] = Counter(f[0] for f in counts)
    for f in keep:
        by_source[f[0]].append(f)
    for src, fs in by_source.items():
        if levels_of[src] == 2 and len(fs) == 2:
            # keep the "yes" side of a yes/no column ("is_return = True"), else the first label
            keep.discard(sorted(fs, key=lambda f: (not _truthy(f[1].split(" = ", 1)[-1]), f[1]))[1])
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
    # each feature as the indices of the records that have it, and as a bitset, so every comparison is a count
    rows: dict[tuple, list[int]] = {k: [] for k in keys}
    for i, s in enumerate(has):
        for k in s:
            rows[k].append(i)
    bits = {k: _bitset(ix, n) for k, ix in rows.items()}
    tests = []
    if use_value:
        shifted = [v - overall for v in values]       # sums of squares around the mean keep the variance exact enough
        tot, tot2 = sum(shifted), sum(v * v for v in shifted)
        for k in keys:
            ix = rows[k]
            s1 = sum(shifted[i] for i in ix)
            s2 = sum(shifted[i] * shifted[i] for i in ix)
            a, b = (len(ix), s1, s2), (n - len(ix), tot - s1, tot2 - s2)
            ma, mb = s1 / a[0] + overall, (tot - s1) / b[0] + overall
            tests.append({"kind": "value", "feature": k[1], "n_with": a[0], "n_without": b[0],
                          "with": ma, "without": mb, "p": _mean_p_sums(a, b),
                          "big": abs(ma - mb) >= min_value_gap * abs(overall)})
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[0] == kb[0]:
                continue                      # levels of one attribute exclude each other by construction
            if ka[0].startswith("item:") and kb[0].startswith("item:"):
                continue                      # items bought together: vertiq.affinity, with its own thresholds
            # context (weekday, part of day, group) is the condition; otherwise the first feature is
            cond, out = (kb, ka) if kb[0] in ("weekday", "part_of_day", "group") and ka[0] not in ("weekday", "part_of_day", "group") else (ka, kb)
            nw, no = len(rows[cond]), n - len(rows[cond])
            hw = (bits[cond] & bits[out]).bit_count()
            ho = len(rows[out]) - hw
            if nw < min_n or no < min_n or not (hw or ho):
                continue
            r1, r0 = hw / nw, ho / no
            tests.append({"kind": "rate", "feature": cond[1], "outcome": out[1], "n_with": nw, "n_without": no,
                          "with": r1, "without": r0, "p": two_proportion_p(hw, nw, ho, no),
                          "big": abs(r1 - r0) >= min_rate_gap})
    if not tests:
        return {"status": "ok", "records": n, "features": len(keys), "tested": 0, "found": [], "not_clear": 0}
    flags = benjamini_hochberg([t["p"] for t in tests], q)
    found = [t for t, ok in zip(tests, flags) if ok and t["big"]]
    found.sort(key=lambda t: t["p"])
    # with several items per record, records that include any common item are bigger records; say so
    many_items = sum(len(e.items or ()) for e in events) / n > 2
    for t in found:
        t["text"] = _sentence(t, noun, currency, many_items)
        t["p_value"] = round(t.pop("p"), 6)
        for k in ("with", "without"):
            t[k] = round(t[k], 4)
        t.pop("big", None)
    return {"status": "ok", "records": n, "features": len(keys), "tested": len(tests), "q": q,
            "found": found[:limit], "not_clear": len(tests) - len(found)}


_TRUE, _FALSE = {"true", "yes", "y", "1"}, {"false", "no", "n", "0"}


def _truthy(v: str) -> bool:
    return v.strip().lower() in _TRUE


def _flag(v: str) -> bool | None:
    """True or False for a yes/no value, None for any other label."""
    v = v.strip().lower()
    return True if v in _TRUE else False if v in _FALSE else None


def _name(column: str) -> str:
    """A column name as words: "payment_mode" -> "payment mode", "is_return" -> "return", "NoGuest" -> "no guest"."""
    import re
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", column).replace("_", " ").replace("-", " ").split()
    if len(words) > 1 and words[0].lower() in ("is", "has", "was"):
        words = words[1:]
    return " ".join(w.lower() if w[:1].isupper() and w[1:].islower() else w for w in words)


def _where(label: str) -> str:
    """The feature as a condition on records: "that include Pizza", "where payment mode is Cash", "marked return",
    "on Saturday"."""
    if label.startswith("includes "):
        return f"that include {label[9:]}"
    if " = " in label:
        k, v = label.split(" = ", 1)
        flag = _flag(v)
        if flag is not None:
            return f"{'marked' if flag else 'not marked'} {_name(k)}"
        return f"where {_name(k)} is {v}"
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
        flag = _flag(v)
        if flag is not None:
            return f"are {'marked' if flag else 'not marked'} {_name(k)}"
        return f"have {_name(k)} {v}"
    if label.startswith("group "):
        return f"are in {label}"
    if label == "before noon":
        return "come before noon"
    if label in ("afternoon", "evening"):
        return f"come in the {label}"
    return f"fall on {label}"


def _sentence(t: dict, noun: str, cur: str, many_items: bool = False) -> str:
    p = "p < 0.001" if t["p"] < 0.001 else f"p = {t['p']:.3f}"
    if t["kind"] == "value":
        size = (f" Part of the gap is size: {noun} with more items are more likely to include any one item."
                if many_items and t["feature"].startswith("includes ") else "")
        return (f"{noun.capitalize()} {_where(t['feature'])} average {cur}{t['with']:,.0f}, against {cur}{t['without']:,.0f} "
                f"for the rest ({t['n_with']} vs {t['n_without']} {noun}; {p}).{size}")
    return (f"{t['with']:.0%} of {noun} {_where(t['feature'])} {_does(t['outcome'])}, against {t['without']:.0%} of the rest "
            f"({t['n_with']} vs {t['n_without']} {noun}; {p}).")
