"""Proposals from the values alone, for when no mapping is given: which column plays each role in a table,
and how a table of line items joins a table of records. Every proposal goes through mapping.repair and
apply afterwards like any other mapping, and every choice is reported.

  propose(rows)                 a Mapping chosen by the fitness scores in vertiq.mapping
  find_join(records, lines)     the shared key, item, item group and quantity columns, from value overlap
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .mapping import (_TIME_OF_DAY, MIN_EVIDENCE, Mapping, _complete, _quantities, date_formats, fitness, placeholder_ids,
                      profile, structural_rows)


def _vals(rows, c):
    return [str(r.get(c) if r.get(c) is not None else "").strip() for r in rows]


def propose(rows: list[dict]) -> tuple[Mapping | None, list[str]]:
    """A mapping chosen from the values: the time is the date column that is filled best, with a separate
    time-of-day column if there is one; the value is the most complete amount column; the entity is the best
    identifier that repeats (a person or asset), else the best identifier; attributes are the remaining
    columns with 2 to 12 repeated labels, and columns that identify each record (kept for joins)."""
    skip = structural_rows(rows)
    data = [r for i, r in enumerate(rows, start=1) if i not in skip]
    if not data:
        return None, ["no records"]
    kind = profile(data)
    notes = []
    dates = sorted(((fitness(data, c, "time", kind), c) for c in kind if kind[c] == "date"), reverse=True)
    if not dates or dates[0][0] < 0.5:
        return None, ["no column holds dates"]
    time = dates[0][1]
    fmts = date_formats(_vals(data, time))
    clock = next((c for c in kind if c != time and sum(1 for v in _vals(data, c) if _TIME_OF_DAY.match(v)) >= 0.8 * len(data)), None)
    amounts = [c for c in kind if kind[c] == "number" and fitness(data, c, "value", kind) >= 0.5 and c not in _quantities(data, kind)]
    value = None
    if amounts:
        # prefer a column no other amount column is more complete than, then the one with the fewest zeros
        complete = [c for c in amounts if not any(_complete(data, c, o, _quantities(data, kind)) for o in amounts if o != c)]
        value = max(complete or amounts, key=lambda c: sum(1 for v in _vals(data, c) if v not in ("", "0", "0.0")))
    n = len(data)
    ids = []
    for c in kind:
        if c in (time, clock, value):
            continue
        f = fitness(data, c, "entity", kind)
        if f < 0.5:
            continue
        real = [v for v in _vals(data, c) if v not in placeholder_ids(data, c)]
        distinct = len(set(real)) / max(1, len(real))
        # an identifier is a code with letters and digits, or a long number (an account or phone number);
        # short plain numbers are amounts or counts
        coded = sum(1 for v in real if any(ch.isdigit() for ch in v)
                    and (any(ch.isalpha() for ch in v) or len(v.replace(" ", "")) >= 6)) >= 0.8 * len(real)
        if coded and distinct >= 0.3:
            ids.append((distinct < 0.98, f, c))       # repeating identifiers first: they name who the record is about
    if not ids:
        return None, notes + ["no column identifies whom the records are about"]
    ids.sort(reverse=True)
    entity = ids[0][2]
    record_keys = [c for rep, _f, c in ids if not rep and c != entity]
    labels = [c for c in kind if c not in (time, clock, value, entity) and kind[c] == "text"
              and 2 <= len(set(v for v in _vals(data, c) if v)) <= 12 and n >= 2]
    m = Mapping(entity=entity, time=time, value=value, attrs=labels + record_keys, date_format=fmts[0] if fmts else "%Y-%m-%d",
                clock=clock)
    notes.append(f"proposed from the values: entity {entity!r}, time {time!r}" + (f" with time of day {clock!r}" if clock else "")
                 + (f", value {value!r}" if value else "") + (f", attributes {labels + record_keys}" if labels or record_keys else ""))
    if len(fmts) > 1:
        notes.append("ambiguous: " + " or ".join(fmts) + " all fit; confirm")
    return m, notes


def find_join(records: list[dict], lines: list[dict]) -> tuple[dict | None, list[str]]:
    """How line items join records: {"event_key", "key", "item", "qty", "item_group"}, from the values. The key
    pair is the one whose record values most often appear in the line table (at least 80% of records); the
    item is the line column with the most distinct repeated labels; the item group is a column with fewer labels
    that each item belongs to consistently; the quantity is a small whole-number column."""
    rs = [r for i, r in enumerate(records, start=1) if i not in structural_rows(records)]
    ls = [r for i, r in enumerate(lines, start=1) if i not in structural_rows(lines)]
    if not rs or not ls:
        return None, ["empty table"]
    best = (0.0, None, None)
    line_sets = {c: set(_vals(ls, c)) - {""} for c in ls[0]}
    for rc in rs[0]:
        rv = set(_vals(rs, rc)) - {""}
        if len(rv) < 0.8 * len(rs):
            continue                               # a key differs on (almost) every record
        for lc, lv in line_sets.items():
            share = len(rv & lv) / len(rv)
            if share > best[0]:
                best = (share, rc, lc)
    if best[0] < 0.8:
        return None, [f"no shared key: the best pair matches {best[0]:.0%} of records"]
    _s, event_key, key = best
    kind = profile(ls)
    texts = [c for c in kind if kind[c] == "text" and c != key]
    distinct = {c: len(set(_vals(ls, c)) - {""}) for c in texts}
    items = [c for c in texts if 2 < distinct[c] < len(ls)]
    if not items:
        return None, ["no item column"]
    item = max(items, key=lambda c: distinct[c])
    group = None
    # the most informative grouping that each item belongs to consistently: at least 3 labels, filled on most lines
    for c in sorted(texts, key=lambda c: -distinct[c]):
        filled = sum(1 for v in _vals(ls, c) if v) / len(ls)
        if c == item or not 3 <= distinct[c] < distinct[item] or filled < 0.8:
            continue
        by: dict[str, Counter] = defaultdict(Counter)
        for r in ls:
            by[str(r.get(item) or "").strip()][str(r.get(c) or "").strip()] += 1
        consistent = sum(cnt.most_common(1)[0][1] for cnt in by.values()) / max(1, sum(sum(cnt.values()) for cnt in by.values()))
        if consistent >= 0.95:
            group = c
            break
    qtys = _quantities(ls, kind)
    qty = next((c for c in qtys if c != key), None) if len(ls) >= MIN_EVIDENCE else None
    out = {"event_key": event_key, "key": key, "item": item, "qty": qty, "item_group": group}
    return out, [f"joined on {event_key!r} = {key!r} ({best[0]:.0%} of records); item {item!r}"
                 + (f", item group {group!r}" if group else "") + (f", quantity {qty!r}" if qty else "")]
