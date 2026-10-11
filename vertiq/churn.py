"""Learned churn: the chance a customer (or account) comes back within a window, learned from many
businesses and sectors at once.

How it learns: for each business, step back in time (cutoffs one window apart), take what was known
about each customer at that cutoff (features.snapshot), and record whether they came back in the next
window. One shared logistic model learns from every sector (the sector is a feature), and each
business gets its own learned correction (Logistic.fit_offsets), so a new business starts from what
was learned everywhere and moves toward its own pattern as its customers come and go.

A host should use it only when it beats a simple recency formula on customers it never saw
(train() reports both), and replace a model only with one that is no worse.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import timedelta

from .features import Visit, cohort_cadence, facts, reference_now, returned, snapshot
from .learn import Logistic, auc, brier, calibration

MAX_CUTOFFS = 4
MIN_ROWS = 300
MIN_CLASS = 30
MIN_CUSTOMERS = 20
TEST_PCT = 20


@dataclass
class History:
    """One business's customers and their visits, with its windows (in days)."""
    business_id: str
    sector: str
    window_days: int            # "came back" means a visit within this many days
    churn_days: int             # the formula baseline's "gone" point
    visits: dict[str, list[Visit]] = field(default_factory=dict)


def held_out(key: str) -> bool:
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % 100 < TEST_PCT


def formula_p_return(recency: int, freq: int, churn_days: int, freq_hi: int) -> float:
    """The formula's view as a chance to return (recency against the churn window, damped for regulars): the baseline every learned model must beat."""
    base = min(100.0, recency / max(1, churn_days) * 100)
    if freq >= freq_hi:
        base *= 0.82
    return 1 - base / 100


def examples(histories: list[History]) -> dict:
    """Every (customer, cutoff) example with its label, business, the formula's guess and a key."""
    out = {"rows": [], "y": [], "groups": [], "base": [], "keys": [], "businesses": 0}
    for h in histories:
        hist = h.visits
        if len(hist) < MIN_CUSTOMERS:
            continue
        window = int(h.window_days)
        ref = reference_now(hist)
        first = min(vs[0].at for vs in hist.values() if vs)
        used = False
        for k in range(1, MAX_CUTOFFS + 1):
            cut = ref - timedelta(days=window * k)
            if cut <= first + timedelta(days=window):
                break
            cad = cohort_cadence(hist, cut, window)
            freqs = sorted(sum(1 for v in vs if v.at < cut) for vs in hist.values())
            freqs = [f for f in freqs if f]
            freq_hi = max(2, freqs[int(len(freqs) * 0.6)]) if freqs else 2
            for cid, vs in hist.items():
                f = snapshot(vs, cut, cad, h.sector or "", len(hist))
                if f is None:
                    continue
                n_before = sum(1 for v in vs if v.at < cut)
                recency = (cut - max(v.at for v in vs if v.at < cut)).days
                out["rows"].append(f)
                out["y"].append(returned(vs, cut, window))
                out["groups"].append(h.business_id)
                out["base"].append(formula_p_return(recency, n_before, int(h.churn_days), freq_hi))
                out["keys"].append(f"{h.business_id}:{cid}")
                used = True
        out["businesses"] += int(used)
    return out


def _score(model: Logistic, rows, y, groups=None) -> dict:
    p = [model.predict(r, g) for r, g in zip(rows, groups or [None] * len(rows))]
    return {"auc": _r(auc(p, y)), "brier": _r(brier(p, y)), "calibration": calibration(p, y)}


def _r(x):
    return round(x, 4) if x is not None else None


def train(ex: dict, previous: Logistic | None = None) -> dict:
    """Learn from examples, test on held-out customers, and say whether the model earned its place
    against the formula and against `previous`. {"status": learning | kept | trained, "metrics",
    "model" (when trained), "reason" (when kept)}."""
    rows, y, n = ex["rows"], ex["y"], len(ex["y"])
    pos = sum(y)
    if n < MIN_ROWS or pos < MIN_CLASS or n - pos < MIN_CLASS:
        return {"status": "learning", "rows": n, "need_rows": MIN_ROWS, "returned": pos,
                "businesses": ex["businesses"]}
    test = [held_out(k) for k in ex["keys"]]
    tr = [i for i in range(n) if not test[i]]
    te = [i for i in range(n) if test[i]]
    pick = lambda xs, idx: [xs[i] for i in idx]      # noqa: E731
    model = Logistic().fit(pick(rows, tr), pick(y, tr))
    held_rows, held_y = pick(rows, te), pick(y, te)
    metrics = {"rows": n, "businesses": ex["businesses"], "return_rate": round(pos / n, 4),
               "held_out": len(te), **_score(model, held_rows, held_y),
               "formula_auc": _r(auc(pick(ex["base"], te), held_y)),
               "formula_brier": _r(brier(pick(ex["base"], te), held_y))}
    if previous is not None:
        try:
            metrics["previous_auc"] = _score(previous, held_rows, held_y)["auc"]
        except KeyError:                      # an older feature set: compare on AUC alone below
            metrics["previous_auc"] = None
    beats_formula = metrics["auc"] is not None and (metrics["formula_auc"] is None or metrics["auc"] >= metrics["formula_auc"])
    beats_prev = metrics.get("previous_auc") is None or metrics["auc"] >= metrics["previous_auc"] - 0.005
    if not (beats_formula and beats_prev):
        reason = "the formula still predicts better" if not beats_formula else "the current model is better"
        return {"status": "kept", "reason": reason, "metrics": metrics}
    final = Logistic(model.names).fit(rows, y)          # every example, now that it has earned its place
    final.fit_offsets(rows, y, ex["groups"])
    return {"status": "trained", "metrics": metrics, "model": final}


def predict(model: Logistic, visits: dict[str, list[Visit]], ref, window_days: int, sector: str,
            business_id: str | None = None) -> dict[str, float]:
    """{customer id: chance to return within the window} as of `ref` (visits on `ref` count)."""
    cut = ref + timedelta(seconds=1)
    cad = cohort_cadence(visits, cut, float(window_days))
    out = {}
    for cid, vs in visits.items():
        f = snapshot(vs, cut, cad, sector or "", len(visits))
        if f is not None:
            out[cid] = model.predict(f, business_id)
    return out


def explain(model: Logistic | None, h: History, customer_id: str) -> dict | None:
    """One customer's facts, chance to return and the reasons in plain words (None if unknown)."""
    vs = h.visits.get(customer_id)
    if not vs:
        return None
    cut = reference_now(h.visits) + timedelta(seconds=1)
    cad = cohort_cadence(h.visits, cut, float(h.window_days))
    fx = facts(vs, cut, cad)
    out = {"window_days": int(h.window_days), "facts": fx, "source": "formula", "risks": [], "strengths": []}
    if model is None:
        return out
    f = snapshot(vs, cut, cad, h.sector or "", len(h.visits))
    p = model.predict(f, h.business_id)
    risks, strengths = reasons(model, f, fx)
    out.update({"source": "model", "return_chance": int(round(100 * p)), "churn_risk": 100 - int(round(100 * p)),
                "risks": risks, "strengths": strengths})
    return out


_SAME = {"log_due": "quiet", "log_recency": "quiet", "rel_recency": "quiet"}


def reasons(model: Logistic, f: dict, fx: dict) -> tuple[list[str], list[str]]:
    """The features pushing this customer away from (risks) and toward (strengths) a return."""
    risks, strengths, seen = [], [], set()
    for key, push in model.contributions(f):
        if abs(push) < 0.05 or key.startswith("sector:") or key == "log_biz_size":
            continue
        group = _SAME.get(key, key)
        if group in seen:
            continue
        words = _words(group, push < 0, fx, f)
        if words:
            seen.add(group)
            (risks if push < 0 else strengths).append(words)
    return risks[:3], strengths[:3]


def _words(group: str, against: bool, fx: dict, f: dict) -> str:
    rec, usual, n = fx.get("recency_days"), fx.get("usual_gap_days"), fx.get("visits")
    v90, vprev = fx.get("visits_90"), fx.get("visits_prev_90")
    if group == "quiet":
        return (f"No visit for {rec} days; they usually come every {usual:g} days" if against
                else f"Visited {rec} days ago, within their usual rhythm")
    if group == "trend":
        if against and vprev and v90 < vprev:
            return f"{v90} visits in the last 3 months, down from {vprev}"
        if not against and v90 > vprev:
            return f"Coming more often: {v90} visits in the last 3 months"
        return ""
    if group in ("log_freq", "one_visit"):
        if against:
            return "Only one visit so far" if n == 1 else f"Only {n} visits so far"
        return f"A regular: {n} visits"
    if group == "spend_trend":
        return "Spending less per visit than before" if against else "Spending more per visit than before"
    if group == "gap_cv":
        return "Their visits are irregular" if against else "Comes at a steady rhythm"
    if group == "log_tenure":
        return "A newer customer" if against else "A long-time customer"
    if group == "weekend_share":
        return ""
    if group == "log_avg_bill":
        return "Smaller bills than most" if against else "Bigger bills than most"
    return ""
