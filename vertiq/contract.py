"""The universal data contract: any domain's records in one shape, so every task works on all of them.

  Entity  who or what a prediction is about (a customer, machine, patient, account, holding, filing),
          with a group (store, plant, hospital, portfolio) that learning can pool across
  Event   something that happened to an entity: when, what type, an optional value, attributes
          (fault code, ward, channel...), and the items involved
  Action  something done to an entity on purpose, or held back from it (the control group), so
          effects can be learned as cause, not correlation

A host maps its own tables to these once; Vertiq never reads a database itself.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Entity:
    id: str
    group: str = ""
    attrs: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class Event:
    entity: str
    at: datetime
    type: str = ""
    value: float = 0.0
    attrs: dict = field(default_factory=dict, hash=False, compare=False)
    items: tuple = ()


@dataclass(frozen=True)
class Action:
    entity: str
    at: datetime
    treated: bool
    kind: str = ""
    attrs: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Dataset:
    """Entities, their events and the actions taken. Lookups are built once and cached."""
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    _by: dict | None = field(default=None, repr=False)

    def by_entity(self) -> dict[str, list[Event]]:
        """{entity id: events, oldest first}."""
        if self._by is None:
            by: dict[str, list[Event]] = defaultdict(list)
            for e in self.events:
                by[e.entity].append(e)
            for v in by.values():
                v.sort(key=lambda e: e.at)
            self._by = dict(by)
        return self._by

    def groups(self) -> dict[str, str]:
        return {e.id: e.group for e in self.entities}

    def reference_now(self) -> datetime:
        """The latest event: the dataset's "today"."""
        return max(e.at for e in self.events) if self.events else datetime.utcnow()

    def types(self) -> set[str]:
        return {e.type for e in self.events}
