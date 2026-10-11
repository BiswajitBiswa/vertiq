"""Declarative rules: checks you configure with data instead of code, for any domain. Each flag names
the rule, the entity, the event and the evidence. The rule types are standard audit ideas:

  repeat(within_days, key)          the same kind of event on the same entity again within N days
                                    (a repeat failure, a readmission, a duplicate claim)
  overdue(due, done, tolerance)     something due that wasn't done, or was done late
                                    (a missed preventive task, a late follow-up, a late filing)
  exceeds(expected, key, qty)       more of an item than expected for that kind of event
                                    (parts beyond a bill of materials, units beyond a protocol)
  too_short(attr, max_words)        a required note left blank or too short to learn from

Rules only flag; people decide. Thresholds are plain numbers a pack (or a user) sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .contract import Dataset, Event


@dataclass
class Rule:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    type: str | None = None          # only events of this type (None = all)

    def applies(self, e: Event) -> bool:
        return self.type is None or e.type == self.type


def repeat(within_days: int, key: str | None = None, type: str | None = None, name: str = "repeat") -> Rule:
    return Rule(name, "repeat", {"within_days": within_days, "key": key}, type)


def overdue(due: str = "due", done: str = "done", tolerance_days: int = 0, type: str | None = None,
            name: str = "overdue") -> Rule:
    return Rule(name, "overdue", {"due": due, "done": done, "tolerance_days": tolerance_days}, type)


def exceeds(expected: dict, key: str, tolerance: float = 0, type: str | None = None, name: str = "exceeds") -> Rule:
    """`expected`: {key value: {item: quantity}}; event.items: [(item, quantity), ...]."""
    return Rule(name, "exceeds", {"expected": expected, "key": key, "tolerance": tolerance}, type)


def too_short(attr: str, max_words: int = 2, type: str | None = None, name: str = "too_short") -> Rule:
    return Rule(name, "too_short", {"attr": attr, "max_words": max_words}, type)


def _flag(rule: Rule, e: Event, detail: str, **evidence) -> dict:
    return {"rule": rule.name, "kind": rule.kind, "entity": e.entity, "at": e.at.isoformat(), "type": e.type,
            "detail": detail, "evidence": evidence}


def _date(v):
    if v is None or v == "":
        return None
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def run(ds: Dataset, rules: list[Rule], today: datetime | None = None) -> dict:
    """{"flags": [...], "summary": {rule name: count}} for every rule over the dataset."""
    today = today or ds.reference_now()
    flags = []
    for eid, evs in ds.by_entity().items():
        for r in rules:
            mine = [e for e in evs if r.applies(e)]
            p = r.params
            if r.kind == "repeat":
                last: dict = {}
                for e in mine:
                    k = e.attrs.get(p["key"]) if p["key"] else e.type
                    prev = last.get(k)
                    if prev is not None and (e.at - prev.at).days <= p["within_days"]:
                        flags.append(_flag(r, e, f"Same {k} again {(e.at - prev.at).days} days after the last one.",
                                           previous=prev.at.isoformat(), gap_days=(e.at - prev.at).days, key=k))
                    last[k] = e
            elif r.kind == "overdue":
                for e in mine:
                    due, done = _date(e.attrs.get(p["due"])), _date(e.attrs.get(p["done"]))
                    if due is None:
                        continue
                    late = ((done or today) - due).days
                    if late > p["tolerance_days"]:
                        flags.append(_flag(r, e, "Not done." if done is None and due < today else f"Done {late} days late.",
                                           due=due.isoformat(), done=done.isoformat() if done else None, days_late=late))
            elif r.kind == "exceeds":
                for e in mine:
                    exp = p["expected"].get(e.attrs.get(p["key"]))
                    if not exp:
                        continue
                    for item, qty in e.items:
                        allowed = exp.get(item, 0) + p["tolerance"]
                        if qty > allowed:
                            flags.append(_flag(r, e, f"{qty:g} × {item}; expected {exp.get(item, 0):g}.",
                                               item=item, used=qty, expected=exp.get(item, 0)))
            elif r.kind == "too_short":
                for e in mine:
                    text = str(e.attrs.get(p["attr"]) or "").strip()
                    if len(text.split()) <= p["max_words"]:
                        flags.append(_flag(r, e, "Left blank." if not text else f'Too short: "{text}".', text=text))
    summary: dict[str, int] = {}
    for f in flags:
        summary[f["rule"]] = summary.get(f["rule"], 0) + 1
    return {"flags": flags, "summary": summary}
