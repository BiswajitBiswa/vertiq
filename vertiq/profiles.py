"""Precomputed profiles for serving: each entity's current score, its contributing factors and the
features behind them, stamped with the time of the latest event included. A scheduled job builds them;
a live request (a checkout, a dispatch screen, a campaign send) reads one record instead of recomputing
history. Where profiles are stored is the host's choice; they are plain dictionaries.
"""
from __future__ import annotations

from datetime import datetime


def build(task, entity_ids: list[str] | None = None, at: datetime | None = None, top: int = 3) -> dict[str, dict]:
    """{entity id: profile} for a trained tasks.Task (or the given entities)."""
    if task.status != "trained":
        return {}
    ids = entity_ids if entity_ids is not None else list(task.b.by)
    out = {}
    for eid in ids:
        s = task.score(eid, at, top=top)
        if s is None:
            continue
        evs = task.b.by.get(eid, [])
        last = max((e.at for e in evs if at is None or e.at < at), default=None)
        out[eid] = {**s, "as_of": last.isoformat() if last else None, "task": task.kind, "target": task.b.target,
                    "window_days": task.window, "learner": task.report().get("metrics", {}).get("learner")}
    return out


def stale(profile: dict, latest_event_at: datetime) -> bool:
    """True when the entity has an event newer than the profile, so the profile should be rebuilt."""
    as_of = profile.get("as_of")
    return as_of is None or latest_event_at > datetime.fromisoformat(as_of)
