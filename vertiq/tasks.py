"""Generic tasks: the questions any domain can ask of its own history.

  predict_event   will an event of a given type happen to an entity within N days?
                  (a customer returns, a machine breaks down, a patient is readmitted, an invoice is paid late)
  predict_value   how much will the next event of a given type be worth?
  predict_uplift  how much does an action (from Dataset.actions, with a held-out control group) change
                  the chance of the event, and for which entities?
                  (next bill, repair cost, length of stay, order value)

Both build examples at several past cutoffs from only what was known at each one, add the pack's
plugin features, run the evaluation gate against the obvious baseline, and return a Task you can
score entities with, each score explained in words.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

from . import harness, plugins, survival
from .contract import Dataset, Event
from .features import Visit, cohort_cadence, snapshot

TOP_TYPES = 8


def _visits(events: list[Event]) -> list[Visit]:
    return [Visit(e.at, float(e.value or 0)) for e in events]


class _Builder:
    """Turns a dataset into feature rows at a cutoff: generic features + per-type counts + plugins."""

    def __init__(self, ds: Dataset, target: str, history: set | None, pack: str | None, use_survival: bool = False):
        self.ds, self.target, self.history, self.pack = ds, target, history, pack
        self.use_survival, self.window, self._km = use_survival, 30, {}
        self.by = ds.by_entity()
        self.groups = ds.groups()
        self.ents = {e.id: e for e in ds.entities}
        self.types = [t for t, _n in Counter(e.type for e in ds.events).most_common(TOP_TYPES)]
        self.feature_fns = plugins.features_for(pack)

    def hist(self, eid: str, cut: datetime) -> list[Event]:
        return [e for e in self.by.get(eid, []) if e.at < cut and (self.history is None or e.type in self.history)]

    def row(self, eid: str, cut: datetime, cadence: float) -> dict | None:
        before = [e for e in self.by.get(eid, []) if e.at < cut]
        h = [e for e in before if self.history is None or e.type in self.history]
        f = snapshot(_visits(h), cut, cadence, self.groups.get(eid, ""), len(self.by)) if h else None
        if f is None:
            return None
        tgt = [e for e in before if e.type == self.target]
        f["n_target"] = math.log1p(len(tgt))
        f["since_target"] = math.log1p((cut - tgt[-1].at).days) if tgt else math.log1p(3650)
        for t in self.types:
            f[f"n90:{t}"] = sum(1 for e in before if e.type == t and e.at >= cut - timedelta(days=90))
        if self.use_survival:
            f["km_within"] = self.km_within(eid, before, cut)
        ent = self.ents.get(eid)
        for fn in self.feature_fns:
            f.update({str(k): float(v) for k, v in (fn(ent, before, cut) or {}).items()})
        return f

    def km_within(self, eid: str, before: list[Event], cut: datetime) -> float:
        """P(target within the window | time since the last target event), from the cohort's survival
        curve at this cutoff, rescaled by the entity's own rhythm (vertiq.survival)."""
        typ = self.target if self.target else None
        if cut not in self._km:
            d, o = survival.gaps(self.by, cut, typ)
            obs = sorted(x for x, ok in zip(d, o) if ok and x > 0)
            self._km[cut] = (survival.kaplan_meier(d, o), obs[len(obs) // 2] if obs else float(self.window))
        curve, cohort = self._km[cut]
        ts = [e.at for e in before if typ is None or e.type == typ]
        if not ts:
            return 0.0
        scale = survival.rhythm(before, cut, typ, cohort) / max(cohort, 1.0)
        return round(survival.within(curve, (cut - max(ts)).days, self.window, scale), 6)

    def cadence(self, cut: datetime, window: int) -> float:
        hist = {eid: _visits(self.hist(eid, cut)) for eid in self.by}
        return cohort_cadence(hist, cut, float(window))

    def cutoffs(self, window: int, n: int) -> list[datetime]:
        ref = self.ds.reference_now()
        first = min(e.at for e in self.ds.events)
        out = []
        for k in range(1, n + 1):
            cut = ref - timedelta(days=window * k)
            if cut <= first + timedelta(days=window):
                break
            out.append(cut)
        return out


def _baseline_event(row: dict, window: int) -> float:
    """The obvious rule: the more recently this event last happened, the likelier it happens again."""
    days = math.expm1(row["since_target"])
    return 1.0 / (1.0 + days / max(1, window))


_WORDS = {"log_recency": "time since the last event", "log_due": "overdue against its own rhythm",
          "trend": "events this quarter vs last", "log_freq": "number of past events", "one_visit": "only one event so far",
          "log_avg_bill": "average value", "spend_trend": "value trend", "gap_cv": "irregular timing",
          "weekend_share": "share on weekends", "log_tenure": "time since the first event", "rel_recency": "quiet vs the group"}


def _label(name: str, target: str) -> str:
    if name == "since_target":
        return f"time since the last {target}"
    if name == "n_target":
        return f"number of past {target} events"
    if name.startswith("n90:"):
        return f"{name[4:] or 'untyped'} events in the last 90 days"
    return _WORDS.get(name, name.replace("_", " "))


class Task:
    def __init__(self, kind: str, builder: _Builder, window: int, learner: str, result: dict):
        self.kind, self.b, self.window, self.learner_name, self.result = kind, builder, window, learner, result
        self.model = result.get("model")

    @property
    def status(self) -> str:
        return self.result["status"]

    def report(self) -> dict:
        return {k: v for k, v in self.result.items() if k != "model"} | {"task": self.kind, "target": self.b.target,
                                                                         "window_days": self.window}

    def score(self, entity_id: str, at: datetime | None = None, top: int = 3) -> dict | None:
        """The prediction for one entity now (or at `at`), with the strongest reasons either way."""
        if self.model is None:
            return None
        cut = at or (self.b.ds.reference_now() + timedelta(seconds=1))
        f = self.b.row(entity_id, cut, self.b.cadence(cut, self.window))
        if f is None:
            return None
        raw = self.model.predict(f, self.b.groups.get(entity_id))
        out = {"entity": entity_id}
        if self.kind == "event":
            out["chance"] = round(raw, 4)
        else:
            out["value"] = round(math.expm1(raw), 2)
        contrib = self.model.contributions(f) if hasattr(self.model, "contributions") else []   # optional for custom learners
        pushes = [(k, c) for k, c in contrib if abs(c) >= 0.05][: top * 2]
        out["raises"] = [_label(k, self.b.target) for k, c in pushes if c > 0][:top]
        out["lowers"] = [_label(k, self.b.target) for k, c in pushes if c < 0][:top]
        return out


def predict_event(ds: Dataset, *, target: str, window_days: int, history: set | None = None, pack: str | None = None,
                  learner: str = "logistic", cutoffs: int = 4, previous=None, min_gain: float = 0.0,
                  confidence: float | None = None, survival: bool = False) -> Task:
    """Will an event of type `target` happen to each entity within `window_days`? `min_gain`: the AUC margin a
    candidate must win by over the baseline and `previous` (see harness.evaluate). `survival` adds the
    time-to-event feature "km_within" (vertiq.survival); the gate decides whether it helps."""
    b = _Builder(ds, target, history, pack, use_survival=survival)
    b.window = window_days
    rows, y, keys, groups, base = [], [], [], [], []
    for cut in b.cutoffs(window_days, cutoffs):
        cad = b.cadence(cut, window_days)
        for eid, evs in b.by.items():
            f = b.row(eid, cut, cad)
            if f is None:
                continue
            rows.append(f)
            y.append(1 if any(e.type == target and cut <= e.at < cut + timedelta(days=window_days) for e in evs) else 0)
            keys.append(eid)
            groups.append(b.groups.get(eid, ""))
            base.append(_baseline_event(f, window_days))
    res = harness.evaluate(rows, y, keys, groups, base, plugins.learner(learner, "classify"), kind="classify", previous=previous,
                           min_gain=min_gain, confidence=confidence)
    return Task("event", b, window_days, learner, res)


def predict_value(ds: Dataset, *, target: str, window_days: int, history: set | None = None, pack: str | None = None,
                  learner: str = "linear", cutoffs: int = 4, previous=None, min_gain: float = 0.0,
                  confidence: float | None = None) -> Task:
    """How much will the next `target` event (within `window_days`) be worth? Entities with no such
    event in the window teach nothing here (predict_event answers whether it happens)."""
    b = _Builder(ds, target, history, pack)
    rows, y, keys, groups, base = [], [], [], [], []
    overall = [e.value for e in ds.events if e.type == target and e.value > 0]
    mean_all = math.log1p(sum(overall) / len(overall)) if overall else 0.0
    for cut in b.cutoffs(window_days, cutoffs):
        cad = b.cadence(cut, window_days)
        for eid, evs in b.by.items():
            nxt = next((e for e in evs if e.type == target and cut <= e.at < cut + timedelta(days=window_days) and e.value > 0), None)
            if nxt is None:
                continue
            f = b.row(eid, cut, cad)
            if f is None:
                continue
            past = [e.value for e in evs if e.type == target and e.at < cut and e.value > 0]
            rows.append(f)
            y.append(math.log1p(nxt.value))
            keys.append(eid)
            groups.append(b.groups.get(eid, ""))
            base.append(math.log1p(sum(past) / len(past)) if past else mean_all)   # the obvious rule: same as before
    res = harness.evaluate(rows, y, keys, groups, base, plugins.learner(learner, "regress"), kind="regress", previous=previous, min_gain=min_gain, confidence=confidence,
                           fit_kw={"epochs": 30})
    return Task("value", b, window_days, learner, res)


# ---- interventions -------------------------------------------------------------------------------
MIN_TREATED, MIN_CONTROL, MIN_EVENTS = 100, 40, 20
FOLDS = 5


class _UpliftModel:
    """Estimated effect of an action on one entity's features.
    "two_model" (default): one model on treated actions and one on the control group (a T-learner);
    the effect is the difference of their predictions. Robust with a few hundred actions.
    "boosted": one boosted-tree model with the treatment as a feature (an S-learner); it captures
    effects that depend on combinations of features and needs more data."""

    def __init__(self, method: str):
        self.method, self.treated, self.control, self.single = method, None, None, None

    def fit(self, feats: list[dict], y: list[int], treated: list[bool]) -> "_UpliftModel":
        from .boost import BoostedClassifier
        from .learn import Logistic
        if self.method == "boosted":
            self.single = BoostedClassifier(min_leaf=15).fit([{**f, "treated": float(t)} for f, t in zip(feats, treated)], y)
        else:
            self.treated = Logistic().fit([f for f, t in zip(feats, treated) if t], [v for v, t in zip(y, treated) if t], l2=1e-2)
            self.control = Logistic().fit([f for f, t in zip(feats, treated) if not t], [v for v, t in zip(y, treated) if not t], l2=1e-2)
        return self

    def both(self, f: dict) -> tuple[float, float]:
        if self.single is not None:
            return self.single.predict({**f, "treated": 1.0}), self.single.predict({**f, "treated": 0.0})
        return self.treated.predict(f), self.control.predict(f)


class UpliftTask:
    """The estimated effect of an action on each entity: how much more (or less) likely the target event
    becomes within the window if the action is taken, compared with not taking it."""

    def __init__(self, builder: _Builder, window: int, result: dict):
        self.b, self.window, self.result = builder, window, result
        self.model = result.get("model")

    @property
    def status(self) -> str:
        return self.result["status"]

    def report(self) -> dict:
        return {k: v for k, v in self.result.items() if k != "model"} | {"task": "uplift", "target": self.b.target,
                                                                         "window_days": self.window}

    def score(self, entity_id: str, at: datetime | None = None) -> dict | None:
        """{"uplift", "with_action", "without_action"} for one entity now (or at `at`)."""
        if self.model is None:
            return None
        cut = at or (self.b.ds.reference_now() + timedelta(seconds=1))
        f = self.b.row(entity_id, cut, self.b.cadence(cut, self.window))
        if f is None:
            return None
        p1, p0 = self.model.both(f)
        return {"entity": entity_id, "uplift": round(p1 - p0, 4), "with_action": round(p1, 4), "without_action": round(p0, 4)}

    def rank(self, entity_ids: list[str], at: datetime | None = None, *, reduce: bool = False) -> list[dict]:
        """Entities ordered by estimated effect: largest increase first, or with `reduce=True` largest
        decrease first (for actions meant to prevent the event, such as maintenance or outreach)."""
        scored = [s for s in (self.score(e, at) for e in entity_ids) if s]
        return sorted(scored, key=lambda s: s["uplift"] if reduce else -s["uplift"])


def predict_uplift(ds: Dataset, *, target: str, window_days: int, history: set | None = None, pack: str | None = None,
                   kind: str | None = None, method: str = "two_model") -> UpliftTask:
    """Learn the effect of an action from `ds.actions`, which records each time the action was taken
    (treated=True) or deliberately withheld (treated=False, the control group). Outcome: whether a
    `target` event followed within `window_days`. `kind` limits to one kind of action; `method` is
    "two_model" or "boosted" (see _UpliftModel).

    The gate for an uplift model differs from a prediction model. Effects are estimated by cross-fitting
    (five folds by entity, each scored by a model that never saw it); actions are ranked by estimated
    effect, and the observed treated-minus-control gap must be larger in the top half than in the bottom
    half. Otherwise the report gives the average effect only."""
    from .learn import two_proportion_p
    b = _Builder(ds, target, history, pack)
    b.window = window_days
    acts = [a for a in ds.actions if kind is None or a.kind == kind]
    feats, y, treated, keys = [], [], [], []
    for a in acts:
        f = b.row(a.entity, a.at, b.cadence(a.at, window_days))
        if f is None:
            continue
        evs = b.by.get(a.entity, [])
        y.append(1 if any(e.type == target and a.at <= e.at < a.at + timedelta(days=window_days) for e in evs) else 0)
        feats.append(f)
        treated.append(a.treated)
        keys.append(a.entity)
    nt, nc = sum(treated), len(treated) - sum(treated)
    xt = sum(v for v, t in zip(y, treated) if t)
    xc = sum(v for v, t in zip(y, treated) if not t)
    counts = {"treated": nt, "control": nc, "events": sum(y), "method": method}
    if nt < MIN_TREATED or nc < MIN_CONTROL or sum(y) < MIN_EVENTS:
        return UpliftTask(b, window_days, {"status": "learning", **counts,
                                           "need": {"treated": MIN_TREATED, "control": MIN_CONTROL, "events": MIN_EVENTS}})
    ate = {"effect": round(xt / nt - xc / nc, 4), "p_value": round(two_proportion_p(xt, nt, xc, nc), 5),
           "treated_rate": round(xt / nt, 4), "control_rate": round(xc / nc, 4)}
    # cross-fitting: every action is scored by a model trained without its entity, so the check uses all
    # actions (and all of the control group) instead of a small held-out slice
    fold = [int(harness.hashlib.md5(k.encode(), usedforsecurity=False).hexdigest(), 16) % FOLDS for k in keys]
    est = {}
    for f in range(FOLDS):
        tr = [i for i in range(len(y)) if fold[i] != f]
        te = [i for i in range(len(y)) if fold[i] == f]
        if not te or sum(treated[i] for i in tr) < 2 or sum(not treated[i] for i in tr) < 2:
            continue
        model = _UpliftModel(method).fit([feats[i] for i in tr], [y[i] for i in tr], [treated[i] for i in tr])
        for i in te:
            p1, p0 = model.both(feats[i])
            est[i] = p1 - p0
    scored = sorted(est, key=lambda i: -est[i])
    half = len(scored) // 2

    def gap(part):
        t = [y[i] for i in part if treated[i]]
        c = [y[i] for i in part if not treated[i]]
        return round(sum(t) / len(t) - sum(c) / len(c), 4) if t and c else None
    top, bottom = gap(scored[:half]), gap(scored[half:])
    metrics = {**counts, "average_effect": ate, "out_of_fold": len(scored), "folds": FOLDS,
               "lift_top_half": top, "lift_bottom_half": bottom}
    if top is None or bottom is None or top <= bottom:
        return UpliftTask(b, window_days, {"status": "kept", "metrics": metrics,
                                           "reason": "ranking by estimated effect does not separate the observed effect "
                                                     "out of fold; use the average effect"})
    final = _UpliftModel(method).fit(feats, y, treated)
    return UpliftTask(b, window_days, {"status": "trained", "metrics": metrics, "model": final})
