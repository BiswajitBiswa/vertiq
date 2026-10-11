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
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y%m%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M")


@dataclass
class Mapping:
    entity: str
    time: str
    type: str | None = None
    value: str | None = None
    group: str | None = None
    attrs: list[str] = field(default_factory=list)
    date_format: str = "%Y-%m-%d"

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
                   date_format=str(d.get("date_format") or "%Y-%m-%d"))

    def as_dict(self) -> dict:
        return {"entity": self.entity, "time": self.time, "type": self.type, "value": self.value, "group": self.group,
                "attrs": list(self.attrs), "date_format": self.date_format}


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
                  attrs=[a for a in (fix(a, "attr") for a in m.attrs) if a], date_format=m.date_format)
    used = {out.entity, out.time}
    for role in ("type", "value", "group"):
        if getattr(out, role) in used:
            changes.append(f"{role}: repeats another role, dropped")
            setattr(out, role, None)
    out.attrs = [a for a in dict.fromkeys(out.attrs) if a not in {out.entity, out.time, out.type, out.value, out.group}]
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


def check(m: Mapping, rows: list[dict]) -> tuple[Mapping, list[str]]:
    """Hold each role to what its values can be: time holds dates, the entity is not a date, the value
    is a number, and a row counter carries no information. A role is only moved when the data leaves
    exactly one candidate; otherwise it is left alone. Returns the mapping and what was changed."""
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
    d["attrs"] = [a for a in d["attrs"] if a not in taken()]
    return Mapping(**d), changes


def repair(m: Mapping, rows: list[dict]) -> tuple[Mapping, list[str]]:
    """Everything the data can settle, in order: names snapped to real columns, roles held to their
    value types, then the date format decided by the rows. The model proposes; the data disposes."""
    m, notes = snap(m, list(rows[0]) if rows else [])
    m, more = check(m, rows)
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
    for role in ROLES:
        c = getattr(m, role)
        if c is not None and c not in cols:
            problems.append(f"column {c!r} (for {role}) isn't in the table")
    problems += [f"attribute column {a!r} isn't in the table" for a in m.attrs if a not in cols]
    if problems:
        return Dataset(), problems
    events, groups = [], {}
    dec = decimal_mark([r[m.value] for r in rows]) if m.value else "."
    for i, r in enumerate(rows, start=1):
        try:
            at = datetime.strptime(str(r[m.time]).strip(), m.date_format)
        except ValueError:
            problems.append(f"row {i}: {r[m.time]!r} doesn't match {m.date_format}")
            continue
        val = _num(r[m.value], dec) if m.value else 0.0
        if val is None:
            problems.append(f"row {i}: value {r[m.value]!r} isn't a number")
            continue
        eid = str(r[m.entity]).strip()
        if not eid:
            problems.append(f"row {i}: empty entity")
            continue
        if m.group:
            groups[eid] = str(r[m.group])
        events.append(Event(eid, at, str(r[m.type]) if m.type else "", val, {a: r[a] for a in m.attrs}))
    ents = [Entity(e, groups.get(e, "")) for e in sorted({ev.entity for ev in events})]
    return Dataset(ents, events), problems


def score(pred: Mapping, truth: Mapping) -> dict:
    """Field-by-field agreement with a known-correct mapping (for evaluation)."""
    roles = {r: getattr(pred, r) == getattr(truth, r) for r in ROLES}
    pa, ta = set(pred.attrs), set(truth.attrs)
    attrs = len(pa & ta) / len(pa | ta) if (pa | ta) else 1.0
    return {"roles": roles, "roles_correct": sum(roles.values()) / len(ROLES), "attrs_overlap": round(attrs, 3),
            "date_format": pred.date_format == truth.date_format,
            "exact": all(roles.values()) and pa == ta and pred.date_format == truth.date_format}
