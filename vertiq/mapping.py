"""How a table in any format becomes Vertiq's contract: a mapping names which column is the entity,
the time, the event type, the value, the group, and which columns to keep as attributes. A person
or a model (Vertiq LM) writes the mapping; `apply` checks it against the real rows, so a wrong
mapping is caught by the data, not trusted.

  {"entity": "patient_no", "time": "admit_dt", "type": "ward", "value": "los_days",
   "group": "hospital", "attrs": ["diagnosis"], "date_format": "%d/%m/%Y"}
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import datetime

from .contract import Dataset, Entity, Event

ROLES = ("entity", "time", "type", "value", "group")
# values that stand for "no value" in exported tables; an entity column filled with them has no identity
PLACEHOLDERS = {"", "-", "--", "---", "—", "–", ".", "x", "na", "n/a", "n.a.", "none", "null", "nil", "nan", "#n/a", "unknown",
                "not available", "not applicable", "0", "00", "000"}
_TIME_OF_DAY = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*([AaPp]\.?[Mm]\.?)?$")
# fewer rows than this settle only what is certain (empty, placeholder, wrong kind); judgements that
# need volume (how many distinct labels a group has) wait for more rows
MIN_EVIDENCE = 30
# labels that mark a summary row rather than a record (several languages)
TOTAL_LABELS = re.compile(r"^\s*(grand\s+total|sub\s*-?\s*total|totals?|sum|summe|gesamt|total\s+general|totale|somme|"
                          r"subtotal|kul|कुल|合计|総計|итого)\b", re.I)
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y%m%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M",
                "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S",
                "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p", "%d-%m-%Y %I:%M %p")


@dataclass
class Mapping:
    entity: str
    time: str
    type: str | None = None
    value: str | None = None
    group: str | None = None
    attrs: list[str] = field(default_factory=list)
    date_format: str = "%Y-%m-%d"
    clock: str | None = None          # a separate time-of-day column ("09:35 PM", "21:35"), when the date has none

    @classmethod
    def parse(cls, text_or_dict) -> "Mapping":
        """From a dict or from model output (the first JSON object in the text). ValueError if unusable."""
        d = text_or_dict
        if isinstance(d, str):
            m = re.search(r"\{.*\}", d, re.S)
            if not m:
                raise ValueError("no JSON object found")
            d = json.loads(m.group(0))
        if not isinstance(d, dict) or not d.get("entity") or not d.get("time"):
            raise ValueError("a mapping needs at least entity and time")
        clean = lambda v: (str(v) if v not in (None, "", "null", "none") else None)      # noqa: E731
        return cls(entity=str(d["entity"]), time=str(d["time"]), type=clean(d.get("type")), value=clean(d.get("value")),
                   group=clean(d.get("group")), attrs=[str(a) for a in (d.get("attrs") or []) if a],
                   date_format=str(d.get("date_format") or "%Y-%m-%d"), clock=clean(d.get("clock")))

    def as_dict(self) -> dict:
        d = {"entity": self.entity, "time": self.time, "type": self.type, "value": self.value, "group": self.group,
             "attrs": list(self.attrs), "date_format": self.date_format}
        if self.clock:
            d["clock"] = self.clock
        return d


def snap(m: Mapping, columns: list[str], min_ratio: float = 0.75) -> tuple[Mapping, list[str]]:
    """Repair names that aren't real columns by snapping each to the most similar real column (when
    similar enough), and drop a role that repeats the entity or time column. Returns the repaired
    mapping and what was changed. Deterministic: it never invents a column, only picks a real one."""
    cols, changes = list(columns), []

    def fix(name, role):
        if name is None or name in cols:
            return name
        best = max(cols, key=lambda c: SequenceMatcher(None, name.lower(), c.lower()).ratio(), default=None)
        ratio = SequenceMatcher(None, name.lower(), best.lower()).ratio() if best else 0
        if best and ratio >= min_ratio:
            changes.append(f"{role}: {name!r} -> {best!r}")
            return best
        changes.append(f"{role}: {name!r} isn't a column, dropped")
        return None
    out = Mapping(entity=fix(m.entity, "entity") or m.entity, time=fix(m.time, "time") or m.time,
                  type=fix(m.type, "type"), value=fix(m.value, "value"), group=fix(m.group, "group"),
                  attrs=[a for a in (fix(a, "attr") for a in m.attrs) if a], date_format=m.date_format,
                  clock=fix(m.clock, "clock"))
    used = {out.entity, out.time}
    for role in ("type", "value", "group"):
        if getattr(out, role) in used:
            changes.append(f"{role}: repeats another role, dropped")
            setattr(out, role, None)
    out.attrs = [a for a in dict.fromkeys(out.attrs) if a not in {out.entity, out.time, out.type, out.value, out.group, out.clock}]
    return out, changes


def date_formats(values: list[str]) -> list[str]:
    """Every known format that parses all the values. One answer means the data decides; two or more
    (e.g. 03/04/2026 fits day-first and month-first) means it is genuinely ambiguous."""
    vals = [str(v).strip() for v in values if str(v or "").strip()]
    fits = []
    for f in DATE_FORMATS:
        try:
            for v in vals:
                datetime.strptime(v, f)
        except ValueError:
            continue
        fits.append(f)
    return fits if vals else []


def fix_dates(m: Mapping, rows: list[dict]) -> tuple[Mapping, str | None]:
    """Let the rows, not the model, decide the date format. Returns the mapping and a note:
    None when the model's format is the only one that fits, a change note when the data proves a
    different one, or "ambiguous: ..." when several fit (the model's guess is kept but flagged so a
    person confirms it; nothing is silently guessed)."""
    if not rows or m.time not in rows[0]:
        return m, None
    fits = date_formats([r[m.time] for r in rows])
    if fits == [m.date_format]:
        return m, None
    if len(fits) == 1:
        return Mapping(**{**m.as_dict(), "date_format": fits[0]}), f"date_format: {m.date_format!r} -> {fits[0]!r} (only one that fits)"
    if len(fits) > 1:
        keep = m.date_format if m.date_format in fits else fits[0]
        return Mapping(**{**m.as_dict(), "date_format": keep}), "ambiguous: " + " or ".join(fits) + " all fit; confirm"
    return m, None                      # no known format fits: apply() will name the bad rows


def _blank(v) -> bool:
    return str(v if v is not None else "").strip().lower() in PLACEHOLDERS - {"0", "00", "000"}


def structural_rows(rows: list[dict]) -> dict[int, str]:
    """Rows that are not records: {row number (1-based): why}. A row is structural when it is labelled as a
    total or subtotal in its first filled cell, or when at least 70% of its cells are empty. These are
    reported and skipped by apply(), never loaded as data."""
    out = {}
    cols = list(rows[0]) if rows else []
    for i, r in enumerate(rows, start=1):
        cells = [r.get(c) for c in cols]
        filled = [str(v).strip() for v in cells if not _blank(v)]
        first = next((v for v in filled if not _AMOUNT.match(v)), "")
        if first and TOTAL_LABELS.match(first) and len(filled) <= max(2, 0.6 * len(cols)):
            out[i] = f"a summary row ({first!r})"
        elif cols and len(filled) < 0.3 * len(cols):
            out[i] = "mostly empty"
    return out


def placeholder_ids(rows: list[dict], col: str) -> set[str]:
    """Values in an entity column that stand for "no identity": known fillers ("-", "NA", "unknown", "0"),
    and any single value covering at least 20% of rows when the other values are nearly all distinct
    (a default number typed for walk-ins)."""
    vals = [str(r.get(col, "") if r.get(col) is not None else "").strip() for r in rows]
    out = {v for v in set(vals) if v.lower() in PLACEHOLDERS}
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    rest = [v for v in vals if v not in out]
    if rest:
        top, n = max(((v, c) for v, c in counts.items() if v not in out), key=lambda vc: vc[1])
        others = [v for v in rest if v != top]
        if n >= 0.2 * len(vals) and n >= 5 and others and len(set(others)) >= 0.9 * len(others):
            out.add(top)
    return out


def fitness(rows: list[dict], col: str, role: str, kind: dict | None = None) -> float:
    """How well a column can play a role, from 0 (cannot) to 1, judged only from its values.
      entity  filled with non-placeholder values; not dates, times of day, amounts or counts; names and short
              plain numbers rank below codes and long numbers (ids, phone numbers)
      time    dates, filled
      value   numbers that are not identifiers or row counters, filled
      group   repeated labels (2 to 50 distinct, or up to 5% of rows), filled; judged from MIN_EVIDENCE rows
      type    repeated labels (as for group, from 1), filled; judged from MIN_EVIDENCE rows
    With fewer rows only what is certain counts: emptiness, placeholders and the kind of value."""
    kind = kind or profile(rows)
    vals = [str(r.get(col, "") if r.get(col) is not None else "").strip() for r in rows]
    if not vals:
        return 0.0
    k = kind.get(col)
    if role == "entity":
        ph = placeholder_ids(rows, col)
        real = [v for v in vals if v not in ph]
        fill = len(real) / len(vals)
        if k == "date" or (k == "number" and any("." in v or "," in v for v in real[:50])):
            return 0.0                       # dates and amounts are not identities
        if real and all(_TIME_OF_DAY.match(v) for v in real):
            return 0.0
        if len(vals) >= MIN_EVIDENCE and col in _quantities(rows, kind):
            return 0.0                       # small whole numbers that repeat are counts, not identities
        if real and sum(1 for v in real if re.search(r"\d", v)) < 0.8 * len(real):
            fill *= 0.85                     # names identify less reliably than codes and numbers
        elif len(vals) >= MIN_EVIDENCE and k == "number" and sum(1 for v in real if len(v) >= 6) < 0.8 * len(real):
            fill *= 0.6                      # short plain numbers are as often prices or counts as identifiers
        if len(vals) >= MIN_EVIDENCE and len(set(real)) < 2:
            return 0.0                       # across many rows, one value everywhere carries no identity
        return fill                          # in a few rows, one entity throughout is fine (one asset's history)
    filled = [v for v in vals if not _blank(v)]
    fill = len(filled) / len(vals)
    if role == "time":
        return fill if k == "date" else 0.0
    if role == "value":
        # zeros are legitimate amounts; a value column is only replaced by a more complete one (_complete)
        looks_like_ids = len(vals) >= MIN_EVIDENCE and _int_ids(rows, col)      # in a few rows, quantities differ too
        return fill if k == "number" and not looks_like_ids else 0.0
    if len(vals) < MIN_EVIDENCE:
        return fill
    distinct = len(set(filled))
    lo = 2 if role == "group" else 1
    return fill if lo <= distinct <= max(50, len(vals) // 20) and k in ("text", "number") else 0.0


def _quantities(rows: list[dict], kind: dict) -> list[str]:
    """Columns that look like quantities: whole numbers, mostly 1 to 100, few distinct values (at most 25),
    not all the same. Prices and totals in whole currency units vary far more."""
    out = []
    for c, k in kind.items():
        if k != "number":
            continue
        vals = [_num(r.get(c)) for r in rows if str(r.get(c) or "").strip()]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3 and all(v == int(v) for v in vals) and sum(1 for v in vals if 1 <= v <= 100) >= 0.9 * len(vals) \
                and 1 < len(set(vals)) <= 25:
            out.append(c)
    return out


def _complete(rows: list[dict], a: str, b: str, qty: list[str] | None = None) -> bool:
    """Column b records the same amounts as column a where a is filled (or a times a quantity column, as a
    line total does for a unit price), and holds amounts where a is 0: b is the more complete record."""
    da, db = decimal_mark([r.get(a) for r in rows]), decimal_mark([r.get(b) for r in rows])
    close = lambda x, y: abs(x - y) <= 0.01 * max(abs(x), abs(y), 1)          # noqa: E731
    both = gap = same = n = 0
    for r in rows:
        x, y = _num(r.get(a), da), _num(r.get(b), db)
        if x is None or y is None:
            continue
        n += 1
        if x == 0:
            gap += y != 0
            continue
        both += 1
        qs = [_num(r.get(q)) for q in (qty or [])]
        same += close(x, y) or any(q and close(x * q, y) for q in qs)
    return both >= 3 and same >= 0.8 * both and gap >= 0.1 * n


def profile(rows: list[dict]) -> dict[str, str]:
    """What each column holds, judged only from its values: "date", "number", "rownum" (1, 2, 3...),
    or "text". Empty cells are ignored."""
    out = {}
    for c in (rows[0] if rows else {}):
        vals = [str(r.get(c, "")).strip() for r in rows if str(r.get(c, "")).strip()]
        dec = decimal_mark(vals)
        nums = [_num(v, dec) for v in vals]
        if vals and date_formats(vals):
            out[c] = "date"
        elif vals and all(n is not None for n in nums):
            ints = [int(n) for n in nums if n == int(n)]
            # a row counter counts from 0 or 1 (15, 16, 17 in three rows can just as well be amounts)
            counter = len(vals) > 1 and len(ints) == len(vals) and ints[0] in (0, 1) and ints == list(range(ints[0], ints[0] + len(vals)))
            out[c] = "rownum" if counter else "number"
        else:
            out[c] = "text"
    return out


def _id_like(rows: list[dict], c: str) -> bool:
    """An identifier: a code with letters and digits (C39793, EQ-602) that differs on every row and is
    not a date or a plain number."""
    vals = [str(r.get(c, "")).strip() for r in rows]
    return (len(set(vals)) == len(vals) and not date_formats(vals)
            and all(re.search(r"\d", v) and re.search(r"[A-Za-z]", v) for v in vals))


def _int_ids(rows: list[dict], c: str) -> bool:
    """Whole numbers that differ on every row (459201, 882103): plausibly an identifier."""
    vals = [str(r.get(c, "")).strip() for r in rows]
    return len(set(vals)) == len(vals) and all(v.isdigit() for v in vals)


def check(m: Mapping, rows: list[dict], exclude: set | tuple = ()) -> tuple[Mapping, list[str]]:
    """Hold each role to what its values can be: time holds dates, the entity is not a date, the value
    is a number, and a row counter carries no information. A role is only moved when the data leaves
    exactly one candidate; otherwise it is left alone. Returns the mapping and what was changed.
    `exclude`: columns never offered as a replacement (for example a key that identifies each record)."""
    kind, changes = profile(rows), []
    if not kind:
        return m, changes
    d = m.as_dict()
    taken = lambda: {d[r] for r in ROLES if d[r]}                    # noqa: E731
    free = [c for c in kind if c not in taken() and _id_like(rows, c)]
    if kind.get(d["entity"]) == "date" and len(free) == 1:            # first, so a swapped entity frees the date column
        changes.append(f"entity: {d['entity']!r} holds dates -> {free[0]!r}")
        d["entity"] = free[0]
    elif not _id_like(rows, d["entity"]) and not _int_ids(rows, d["entity"]) and len(free) == 1:
        # weaker evidence (a customer name is a fine entity), so flag it for a person, don't move it
        changes.append(f"confirm: entity {d['entity']!r} or {free[0]!r}, which looks like an identifier")
    if kind.get(d["time"]) != "date":
        free = [c for c in kind if kind[c] == "date" and c not in taken()]
        if len(free) == 1:
            changes.append(f"time: {d['time']!r} holds no dates -> {free[0]!r}")
            d["time"] = free[0]
    if d["value"] and kind.get(d["value"]) != "number":
        why = "is a row counter" if kind.get(d["value"]) == "rownum" else "isn't numeric"
        changes.append(f"value: {d['value']!r} {why}, dropped")
        d["value"] = None
    dropped = [a for a in d["attrs"] if kind.get(a) == "rownum"]
    if dropped:
        changes.append(f"attrs: row counters dropped {dropped}")
        d["attrs"] = [a for a in d["attrs"] if a not in dropped]
    # fitness: a role whose column cannot play it is moved to the clearly best free column, or flagged
    for role in ("entity", "time", "value", "group"):
        cur = d[role]
        if cur is None or cur not in kind:
            continue
        f = fitness(rows, cur, role, kind)
        if f >= 0.5:
            continue
        # a replacement group or type must be text labels; a numeric code is kept when proposed, never chosen
        cands = sorted(((fitness(rows, c, role, kind), c) for c in kind if c not in taken() and c not in exclude
                        and (role not in ("group", "type") or kind[c] == "text")), reverse=True)
        best = cands[0] if cands else (0.0, None)
        second = cands[1][0] if len(cands) > 1 else 0.0
        if best[1] and best[0] >= 0.8 and best[0] - second >= 0.15:
            changes.append(f"{role}: {cur!r} is {_why(rows, cur, role, kind)} -> {best[1]!r}")
            d[role] = best[1]
        elif role in ("entity", "time"):
            changes.append(f"confirm: {role} {cur!r} is {_why(rows, cur, role, kind)}; no clearly better column")
        else:
            changes.append(f"{role}: {cur!r} is {_why(rows, cur, role, kind)}, dropped")
            d[role] = None
    if d["value"]:
        for c in kind:
            if c not in taken() and kind[c] == "number" and _complete(rows, d["value"], c, _quantities(rows, kind)):
                changes.append(f"value: {d['value']!r} is 0 on rows where {c!r} records the same amount -> {c!r}")
                d["value"] = c
                break
    d["attrs"] = [a for a in d["attrs"] if a not in taken()]
    return Mapping(**d), changes


def _why(rows, col, role, kind) -> str:
    vals = [str(r.get(col, "") if r.get(col) is not None else "").strip() for r in rows]
    if role == "entity":
        ph = placeholder_ids(rows, col)
        share = sum(1 for v in vals if v in ph) / max(1, len(vals))
        return f"a placeholder in {share:.0%} of rows" if share >= 0.5 else "not an identifier"
    empty = sum(1 for v in vals if _blank(v)) / max(1, len(vals))
    if empty >= 0.5:
        return f"empty in {empty:.0%} of rows"
    return {"time": "not dates", "value": "not an amount", "group": "not a set of repeated labels"}.get(role, "unfit")


def repair(m: Mapping, rows: list[dict], exclude: set | tuple = ()) -> tuple[Mapping, list[str]]:
    """Everything the data can settle, in order: names snapped to real columns, roles held to their
    value types, then the date format decided by the rows. The model proposes; the data disposes.
    `exclude`: columns never offered as a replacement role (see check)."""
    m, notes = snap(m, list(rows[0]) if rows else [])
    m, more = check(m, rows, exclude)
    m, note = fix_dates(m, rows)
    return m, notes + more + ([note] if note else [])


# an amount: optional sign, an optional currency (a symbol, R$/Rp/Rs, or a 3-letter code followed by a
# space, so "ACC4399" or "SKU-1678" stay identifiers), digits with separators, an optional trailing code
_AMOUNT = re.compile(r"^(-)?\s*(?:(?:R\$|Rp\.?|Rs\.?|[$€£¥₹₩₽₺]|[A-Z]{3}\s)\s*)?(-)?(\d[\d.,\s']*)\s*(?:[A-Z]{3}|[$€£¥₹]|%)?$")


def _mark(n: str) -> str | None:
    """The decimal mark one number proves, if any: "," for 1.250.000 / 4.350,50 / 12,5; "." for
    1,250,000 / 4,350.50 / 12.5; None when it can't tell (1,200 or 4.350)."""
    if n.count(".") > 1 or ("." in n and "," in n and n.rindex(",") > n.rindex(".")) or \
            ("." not in n and re.search(r",\d{1,2}$", n)):
        return ","
    if n.count(",") > 1 or ("." in n and "," in n) or ("," not in n and re.search(r"\.\d{1,2}$", n)):
        return "."
    return None


def decimal_mark(values) -> str:
    """How a column writes decimals: "," or "." when its values agree (one unambiguous value settles
    the rest), "mixed" when they disagree (each value is then read on its own). R$ and Rp default to
    ","; everything else to "."."""
    seen = set()
    for v in values:
        m = _AMOUNT.match(str(v or "").strip())
        if m and (k := _mark(m.group(3).replace(" ", "").replace("'", ""))):
            seen.add(k)
    if len(seen) == 2:
        return "mixed"
    if seen:
        return seen.pop()
    if any(re.match(r"\s*-?\s*(R\$|Rp)", str(v or "")) for v in values):
        return ","
    return "."


def _num(v, decimal: str = ".") -> float | None:
    s = str(v or "").strip()
    if not s:
        return 0.0
    m = _AMOUNT.match(s)
    if not m:
        return None
    n = m.group(3).replace(" ", "").replace("'", "")
    if decimal == "mixed":
        decimal = _mark(n) or "."
    n = n.replace(".", "").replace(",", ".") if decimal == "," else n.replace(",", "")
    try:
        x = float(n)
    except ValueError:
        return None
    return -x if (m.group(1) or m.group(2)) else x


def apply(m: Mapping, rows: list[dict]) -> tuple[Dataset, list[str]]:
    """Rows -> Dataset, plus problems (missing columns, unparseable dates or values). Rows that
    fail are skipped and named; nothing is guessed."""
    problems = []
    cols = set(rows[0]) if rows else set()
    for role in ROLES + ("clock",):
        c = getattr(m, role)
        if c is not None and c not in cols:
            problems.append(f"column {c!r} (for {role}) isn't in the table")
    problems += [f"attribute column {a!r} isn't in the table" for a in m.attrs if a not in cols]
    if problems:
        return Dataset(), problems
    events, groups = [], {}
    skip = structural_rows(rows)
    ph = placeholder_ids(rows, m.entity)
    anonymous = 0
    dec = decimal_mark([r[m.value] for i, r in enumerate(rows, start=1) if i not in skip]) if m.value else "."
    for i, r in enumerate(rows, start=1):
        if i in skip:
            problems.append(f"row {i}: {skip[i]}, skipped")
            continue
        try:
            at = datetime.strptime(str(r[m.time]).strip(), m.date_format)
        except ValueError:
            problems.append(f"row {i}: {r[m.time]!r} doesn't match {m.date_format}")
            continue
        if m.clock:
            t = _clock(r.get(m.clock))
            if t is None:
                problems.append(f"row {i}: time of day {r.get(m.clock)!r} not understood; date kept")
            else:
                at = at.replace(hour=t[0], minute=t[1])
        val = _num(r[m.value], dec) if m.value else 0.0
        if val is None:
            problems.append(f"row {i}: value {r[m.value]!r} isn't a number")
            continue
        eid = str(r[m.entity] if r[m.entity] is not None else "").strip()
        if eid in ph:                        # no identity: its own one-off entity, never merged with others
            eid = f"anonymous:{i}"
            anonymous += 1
        if m.group:
            groups[eid] = str(r[m.group])
        events.append(Event(eid, at, str(r[m.type]) if m.type else "", val, {a: r[a] for a in m.attrs}))
    if anonymous:
        problems.append(f"{anonymous} rows have a placeholder entity ({', '.join(sorted(repr(v) for v in ph))}); "
                        "each is kept as its own one-off entity")
    ents = [Entity(e, groups.get(e, "")) for e in sorted({ev.entity for ev in events})]
    return Dataset(ents, events), problems


def _clock(v) -> tuple[int, int] | None:
    s = str(v or "").strip()
    for f in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S", "%I:%M:%S %p"):
        try:
            t = datetime.strptime(s.upper(), f)
            return t.hour, t.minute
        except ValueError:
            continue
    return None


def attach_items(ds: Dataset, rows: list[dict], *, key: str, event_key: str, item: str, qty: str | None = None,
                 item_group: str | None = None) -> tuple[Dataset, dict[str, str], list[str]]:
    """Attach line items to the records they belong to through a shared key (order lines to orders, claim
    lines to claims, part lines to work orders). `event_key` is the attribute of each event holding the key
    (keep that column in the mapping's attrs); `key`, `item`, `qty` and `item_group` are columns of the line
    table. Returns the dataset with items on each event, {item: item group} and notes. Summary and empty
    line rows are skipped; lines whose key matches no record are counted, not guessed."""
    skip = structural_rows(rows)
    lines: dict[str, list] = {}
    groups: dict[str, str] = {}
    for i, r in enumerate(rows, start=1):
        k = str(r.get(key) or "").strip()
        name = str(r.get(item) or "").strip()
        if i in skip or not k or not name:
            continue
        n = _num(r.get(qty)) if qty else 1.0
        lines.setdefault(k, []).append((name, n if n else 1.0))
        if item_group and str(r.get(item_group) or "").strip():
            groups[name] = str(r[item_group]).strip()
    events, matched = [], set()
    for e in ds.events:
        # the key is normally kept as an attribute; when the mapping made it the entity, the entity holds it
        k = str((e.attrs or {}).get(event_key) or "").strip() or (e.entity if event_key not in (e.attrs or {}) else "")
        if k in lines:
            matched.add(k)
            events.append(Event(e.entity, e.at, e.type, e.value, e.attrs, tuple(lines[k])))
        else:
            events.append(e)
    notes = [f"{len(matched)} of {len(ds.events)} records got line items"]
    if len(lines) - len(matched):
        notes.append(f"{len(lines) - len(matched)} line keys match no record")
    return Dataset(ds.entities, events, ds.actions), groups, notes


def score(pred: Mapping, truth: Mapping) -> dict:
    """Field-by-field agreement with a known-correct mapping (for evaluation)."""
    roles = {r: getattr(pred, r) == getattr(truth, r) for r in ROLES}
    pa, ta = set(pred.attrs), set(truth.attrs)
    attrs = len(pa & ta) / len(pa | ta) if (pa | ta) else 1.0
    return {"roles": roles, "roles_correct": sum(roles.values()) / len(ROLES), "attrs_overlap": round(attrs, 3),
            "date_format": pred.date_format == truth.date_format,
            "exact": all(roles.values()) and pa == ta and pred.date_format == truth.date_format}
