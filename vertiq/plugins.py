"""Extension points: add domain knowledge without forking Vertiq.

  @feature(pack)   fn(entity, events_before, cutoff) -> {name: number}; events_before holds only what
                   was known before the cutoff, so a plugin can't leak the future
  @rule(pack)      fn(entity, events) -> [flag dicts]; see vertiq.rules for ready-made rule types
  register_learner(name, cls)   any class with fit(rows, y) / predict(row, group) / to_dict / from_dict

Plugins are registered under a pack name ("manufacturing", "healthcare", or your own) and a task picks
up the pack's plugins when you pass pack=... .
"""
from __future__ import annotations

from collections import defaultdict

from .learn import Linear, Logistic

_FEATURES: dict[str, list] = defaultdict(list)
_RULES: dict[str, list] = defaultdict(list)
_LEARNERS: dict[str, type] = {"logistic": Logistic, "linear": Linear}


def feature(pack: str):
    def wrap(fn):
        _FEATURES[pack].append(fn)
        return fn
    return wrap


def rule(pack: str):
    def wrap(fn):
        _RULES[pack].append(fn)
        return fn
    return wrap


def register_learner(name: str, cls: type) -> None:
    for need in ("fit", "predict", "to_dict", "from_dict"):
        if not hasattr(cls, need):
            raise TypeError(f"a learner needs {need}()")
    _LEARNERS[name] = cls


def features_for(pack: str | None) -> list:
    return list(_FEATURES.get(pack or "", []))


def rules_for(pack: str | None) -> list:
    return list(_RULES.get(pack or "", []))


def learner(name: str) -> type:
    if name not in _LEARNERS:
        raise KeyError(f"no learner named {name!r}; registered: {', '.join(sorted(_LEARNERS))}")
    return _LEARNERS[name]


def clear(pack: str | None = None) -> None:
    """Forget a pack's plugins (or all), e.g. between tests."""
    for reg in (_FEATURES, _RULES):
        if pack is None:
            reg.clear()
        else:
            reg.pop(pack, None)
