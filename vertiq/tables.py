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


def _id_numbers(vals: list[str]) -> bool:
    """Whole numbers of 4 or more digits that nearly all have the same length (customer 13085, phone 9075255510):
    identifiers. Amounts vary in length (999, 1499, 12000) or carry decimals."""
    vs = [v for v in vals if v]
    if not vs or not all(v.isdigit() and len(v) >= 4 for v in vs):
        return False
    return Counter(len(v) for v in vs).most_common(1)[0][1] >= 0.9 * len(vs)


def propose(rows: list[dict], record_key: str | None = None, latest: bool = False) -> tuple[Mapping | None, list[str]]:
    """A mapping chosen from the values: the time is the date column that is filled best, with a separate
    time-of-day column if there is one; the value is the most complete amount column; the entity is the best
    identifier that repeats (a person or asset), else the best identifier; attributes are the remaining
    columns with 2 to 12 repeated labels, and columns that identify each record (kept for joins).
    `record_key`: a column known to identify the record itself (the key line items join on); it is kept as
    an attribute and never proposed as the entity, since it names the order, not whom it is about."""
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
    if latest:                                        # one row per entity: the latest date column is the last activity
        good = [c for f, c in dates if f >= 0.5]

        def last(c):
            fs = date_formats(_vals(data, c))
            from datetime import datetime as _dt
            got = []
            for v in _vals(data, c):
                try:
                    got.append(_dt.strptime(v, fs[0]))
                except (ValueError, IndexError):
                    pass
            return sorted(got)[len(got) // 2] if got else _dt.min
        time = max(good, key=last) if good else time
    fmts = date_formats(_vals(data, time))
    clock = next((c for c in kind if c != time and sum(1 for v in _vals(data, c) if _TIME_OF_DAY.match(v)) >= 0.8 * len(data)), None)
    amounts = [c for c in kind if kind[c] == "number" and fitness(data, c, "value", kind) >= 0.5 and c not in _quantities(data, kind)
               and not _id_numbers(_vals(data, c))]
    value = None
    if amounts:
        # prefer a column no other amount column is more complete than, then the one with the fewest zeros
        complete = [c for c in amounts if not any(_complete(data, c, o, _quantities(data, kind)) for o in amounts if o != c)]
        value = max(complete or amounts, key=lambda c: sum(1 for v in _vals(data, c) if v not in ("", "0", "0.0")))
        rivals = [c for c in (complete or amounts) if c != value
                  and sum(1 for v in _vals(data, c) if v not in ("", "0", "0.0")) >= 0.9 * len(data)]
        if rivals:
            notes.append(f"confirm: value {value!r}; {', '.join(repr(c) for c in rivals)} also complete")
    n = len(data)
    ids = []
    for c in kind:
        if c in (time, clock, value, record_key):
            continue
        f = fitness(data, c, "entity", kind)
        if f < 0.2:
            continue                                  # records without an id become one-off entities (reported)
        real = [v for v in _vals(data, c) if v not in placeholder_ids(data, c)]
        distinct = len(set(real)) / max(1, len(real))
        # an identifier is a code with letters and digits, a fixed-length whole number (an account, customer or phone
        # number), or a long number; short numbers and numbers with decimals are amounts or counts
        coded = _id_numbers(real) or sum(1 for v in real if any(ch.isdigit() for ch in v)
                                         and (any(ch.isalpha() for ch in v) or len(v.replace(" ", "")) >= 6)) >= 0.8 * len(real)
        many = len(set(real)) >= max(20, 0.05 * len(real))     # an entity has many values; a store or branch code has few
        named = kind[c] == "text" and not coded and many and distinct < 0.9    # names that repeat: customers, suppliers
        if (coded and many) or named:
            ids.append((distinct < 0.98, f, c))       # repeating identifiers first: they name who the record is about
    if not ids:
        return None, notes + ["no column identifies whom the records are about"]
    ids.sort(reverse=True)
    entity = ids[0][2]
    record_keys = [c for rep, _f, c in ids if not rep and c != entity]
    if record_key and record_key in kind and record_key not in record_keys:
        record_keys.append(record_key)
    labels = [c for c in kind if c not in (time, clock, value, entity) and kind[c] == "text"
              and 2 <= len(set(v for v in _vals(data, c) if v)) <= 12 and n >= 2]
    m = Mapping(entity=entity, time=time, value=value, attrs=labels + record_keys, date_format=fmts[0] if fmts else "%Y-%m-%d",
                clock=clock)
    fill = ids[0][1]
    if fill < 0.8:
        notes.append(f"{entity!r} is filled on {fill:.0%} of records; the rest have no id and count as one-off entities")
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


def find_lines(rows: list[dict]) -> tuple[dict | None, list[str]]:
    """Whether a single table holds line items (one row per item) rather than records, and its columns:
    {"key", "item", "qty", "price", "amount", "item_group"}. The key is a coded column that repeats (2% to 90%
    distinct) and whose rows share one timestamp in at least 95% of keys; the item is the text column with the
    most distinct values; quantity and price are found from their values. None when the table is not lines."""
    data = [r for i, r in enumerate(rows, start=1) if i not in structural_rows(rows)]
    n = len(data)
    if n < MIN_EVIDENCE:
        return None, []
    kind = profile(data)
    dates = [c for c in kind if kind[c] == "date"]
    if not dates:
        return None, []
    time = max(dates, key=lambda c: sum(1 for v in _vals(data, c) if v))
    best = None
    for c in kind:
        if c in dates:
            continue
        vals = _vals(data, c)
        distinct = len(set(v for v in vals if v))
        if not 0.02 * n <= distinct <= 0.9 * n or sum(1 for v in vals if any(ch.isdigit() for ch in v)) < 0.8 * n:
            continue
        times: dict[str, set] = defaultdict(set)
        for r, v in zip(data, vals):
            times[v].add(str(r.get(time) or ""))
        same = sum(1 for t in times.values() if len(t) == 1) / max(1, len(times))
        if same >= 0.95 and (best is None or distinct > best[1]):
            best = (c, distinct)
    if not best:
        return None, []
    key = best[0]
    texts = [c for c in kind if kind[c] == "text" and c != key]
    distinct = {c: len(set(_vals(data, c)) - {""}) for c in texts}
    items = [c for c in texts if distinct[c] >= 10]
    if not items:
        return None, []
    item = max(items, key=lambda c: distinct[c])
    nums = [c for c in kind if kind[c] == "number" and c != key]
    whole = [c for c in nums if all(float(v).is_integer() for v in _vals(data, c) if _isnum(v))
             and sum(1 for v in _vals(data, c) if _isnum(v) and 1 <= abs(float(v)) <= 100) >= 0.7 * n]
    qty = whole[0] if whole else None
    prices = [c for c in nums if c != qty and any("." in v for v in _vals(data, c)[:200])]
    price = prices[0] if prices else None
    amount = None
    if qty and price:
        for c in nums:
            if c in (qty, price):
                continue
            ok = sum(1 for r in data if _isnum(r.get(c)) and _isnum(r.get(qty)) and _isnum(r.get(price))
                     and abs(float(r[c]) - float(r[qty]) * float(r[price])) <= 0.01 * max(1.0, abs(float(r[c]))))
            if ok >= 0.9 * n:
                amount = c
                break
    group = None
    for c in sorted(texts, key=lambda c: -distinct[c]):
        filled = sum(1 for v in _vals(data, c) if v) / n
        if c in (item,) or not 3 <= distinct[c] < distinct[item] or filled < 0.8:
            continue
        by: dict[str, Counter] = defaultdict(Counter)
        for r in data:
            by[str(r.get(item) or "").strip()][str(r.get(c) or "").strip()] += 1
        if sum(cnt.most_common(1)[0][1] for cnt in by.values()) / n >= 0.95:
            group = c
            break
    spec = {"key": key, "item": item, "qty": qty, "price": price, "amount": amount, "item_group": group, "time": time}
    return spec, [f"one row per item: records keyed by {key!r} ({best[1]} records from {n} lines), item {item!r}"
                  + (f", quantity {qty!r}" if qty else "") + (f", price {price!r}" if price else "")
                  + (f", line amount {amount!r}" if amount else "") + (f", item group {group!r}" if group else "")]


def _isnum(v) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def records_from_lines(rows: list[dict], spec: dict) -> list[dict]:
    """One row per record from a line table: the record's columns taken from its first line (those that are the
    same on all its lines), and "Line value" = the sum of the line amounts, or of quantity x price."""
    skip = structural_rows(rows)
    by: dict[str, list] = defaultdict(list)
    for i, r in enumerate(rows, start=1):
        k = str(r.get(spec["key"]) or "").strip()
        if i not in skip and k:
            by[k].append(r)
    line_cols = {spec["item"], spec.get("qty"), spec.get("price"), spec.get("amount"), spec.get("item_group")} - {None}
    out = []
    for k, ls in by.items():
        first = ls[0]
        rec = {c: v for c, v in first.items() if c not in line_cols and all(str(x.get(c)) == str(v) for x in ls)}
        total = 0.0
        for x in ls:
            try:
                if spec.get("amount"):
                    total += float(str(x[spec["amount"]]).replace(",", ""))
                elif spec.get("qty") and spec.get("price"):
                    total += float(str(x[spec["qty"]]).replace(",", "")) * float(str(x[spec["price"]]).replace(",", ""))
            except (TypeError, ValueError):
                continue
        rec["Line value"] = f"{total:.2f}"
        out.append(rec)
    return out


def entity_table(rows: list[dict]) -> dict | None:
    """Whether a table has one row per entity (a customer list, an asset register, a policy master): an
    identifier column that differs on (nearly) every row. Returns {"key", "dates", "counts"}: the date columns
    (first and last activity, renewal) and whole-number columns, which in such tables are usually totals
    computed when the file was exported. None when no column identifies each row."""
    data = [r for i, r in enumerate(rows, start=1) if i not in structural_rows(rows)]
    if len(data) < MIN_EVIDENCE:
        return None
    kind = profile(data)
    keys = []
    for c in kind:
        real = [v for v in _vals(data, c) if v not in placeholder_ids(data, c)]
        if len(real) >= 0.8 * len(data) and len(set(real)) >= 0.98 * len(real) and kind[c] != "date" and \
                (_id_numbers(real) or sum(1 for v in real if any(ch.isdigit() for ch in v)) >= 0.8 * len(real)):
            keys.append((len(real), c))
    if not keys:
        return None
    key = max(keys)[1]
    dates = [c for c in kind if kind[c] == "date"]
    counts = [c for c in _quantities(data, kind) if c != key]
    return {"key": key, "dates": dates, "counts": counts}


def attach_entities(ds, rows: list[dict], key: str | None = None) -> tuple[object, list[str], set]:
    """Entity attributes from a table with one row per entity, joined to the dataset's entities by the column
    whose values match most entity ids (at least 30% of entities). Returns the dataset, notes, and the names of
    attributes that are fixed labels (safe as prediction features); dates and numbers in such tables are often
    totals computed at export time and would leak the future into a prediction, so they are kept for
    description only."""
    from .contract import Dataset, Entity
    data = [r for i, r in enumerate(rows, start=1) if i not in structural_rows(rows)]
    ids = {e.id for e in ds.entities}
    if key is None:
        best = (0.0, None)
        for c in (data[0] if data else {}):
            share = len(set(_vals(data, c)) & ids) / max(1, len(ids))
            if share > best[0]:
                best = (share, c)
        if best[0] < 0.3:
            return ds, [f"entity table: no column matches the entity ids (best {best[0]:.0%})"], set()
        key = best[1]
    kind = profile(data)
    labels = {c for c in kind if c != key and kind[c] == "text" and 2 <= len(set(_vals(data, c)) - {"", "-"}) <= 12}
    by = {str(r.get(key) or "").strip(): r for r in data}
    ents, matched = [], 0
    for e in ds.entities:
        r = by.get(e.id)
        if r:
            matched += 1
            extra = {c: r[c] for c in r if c != key and str(r[c]).strip() not in ("", "-")}
            ents.append(Entity(e.id, e.group, {**(e.attrs or {}), **extra}))
        else:
            ents.append(e)
    note = (f"entity table joined on {key!r}: {matched} of {len(ds.entities)} entities found; fixed labels "
            f"{sorted(labels) or 'none'} can describe and predict; dates and totals describe only")
    return Dataset(ents, ds.events, ds.actions), [note], labels
