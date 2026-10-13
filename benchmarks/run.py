"""Vertiq's benchmark suite: synthetic datasets with a planted truth, so every result can be checked
against the answer, and every learned model is compared with the obvious baseline on entities it
never saw. Each scenario runs over several seeds; results are mean ± standard deviation.

These are synthetic benchmarks. They prove the machinery finds a real signal, ignores noise and
refuses to ship a model that doesn't beat the baseline. They do not prove accuracy on your data;
the evaluation gate (vertiq.harness) measures that on your own held-out entities every time.

Run:  python benchmarks/run.py [--seeds 5] [--out benchmarks/results/core.json]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vertiq import churn, insights, plugins, uplift, value  # noqa: E402
from vertiq.contract import Dataset, Entity, Event  # noqa: E402
from vertiq.features import Visit  # noqa: E402
from vertiq.tasks import predict_event, predict_value  # noqa: E402

NOW = datetime(2026, 9, 30)


# ---- 1. customers: who comes back --------------------------------------------------------------
def churn_customers(seed: int) -> dict:
    """Two cafés, 250 customers each, with irregular timing (each gap is 0.5x to 1.8x the customer's
    rhythm). 60% eventually stop; two in three of those slow down first (gaps stretch 2.5x in their last
    70 days), the rest stop without warning. Slowing down is the signal a recency formula can't see."""
    rnd = random.Random(seed)
    hists = []
    for b in range(2):
        visits = {}
        for i in range(250):
            gap = rnd.choice([7, 10, 14, 21])
            stop = NOW + timedelta(days=999) if rnd.random() < 0.4 else NOW - timedelta(days=rnd.randint(20, 400))
            warns = rnd.random() < 2 / 3
            t, vs = NOW - timedelta(days=600 - rnd.randint(0, 100)), []
            while t < NOW and t < stop:
                vs.append(Visit(t, rnd.uniform(200, 900)))
                slow = 2.5 if warns and (stop - t).days < 70 else 1
                t += timedelta(days=max(1, round(gap * slow * rnd.uniform(0.5, 1.8))))
            if vs:
                visits[f"c{i}"] = vs
        hists.append(churn.History(f"b{b}", "cafe", 30, 90, visits))
    out = churn.train(churn.examples(hists))
    m = out["metrics"]
    return {"status": out["status"], "model_auc": m["auc"], "baseline_auc": m["formula_auc"],
            "model_brier": m["brier"], "baseline_brier": m["formula_brier"], "held_out": m["held_out"]}


# ---- 2. machines: will it break down (and what a plugin adds) ----------------------------------
def _plant(seed: int, machines: int = 240) -> Dataset:
    """Preventive maintenance every ~30 days; 35% of machines skip some. Each skip in the last 60
    days triples the weekly breakdown chance."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(machines):
        mid = f"M{i}"
        ents.append(Entity(mid, group=f"plant-{i % 3}"))
        skipper = rnd.random() < 0.35
        t, skips = NOW - timedelta(days=720), []
        while t < NOW:
            done = not (skipper and rnd.random() < 0.6)
            evs.append(Event(mid, t, "pm", 0.0, {"done": done}))
            if not done:
                skips.append(t)
            t += timedelta(days=30)
        d = NOW - timedelta(days=720)
        while d < NOW:
            recent = sum(1 for s in skips if timedelta(0) <= d - s <= timedelta(days=60))
            if rnd.random() < 0.02 * (3 ** min(recent, 2)):
                evs.append(Event(mid, d, "breakdown", rnd.uniform(500, 3000)))
            d += timedelta(days=7)
    return Dataset(ents, evs)


def machines(seed: int) -> dict:
    ds = _plant(seed)
    plugins.clear()
    plain = predict_event(ds, target="breakdown", window_days=30)

    @plugins.feature("plants")
    def skipped_pm_60(entity, before, cut):          # the domain knowledge: skipped maintenance matters
        return {"skipped_pm_60": sum(1 for e in before if e.type == "pm" and not e.attrs.get("done")
                                     and e.at >= cut - timedelta(days=60))}
    smart = predict_event(ds, target="breakdown", window_days=30, pack="plants")
    plugins.clear()
    pm, sm = plain.report()["metrics"], smart.report()["metrics"]
    return {"baseline_auc": sm["baseline_auc"], "generic_auc": pm["auc"], "plugin_auc": sm["auc"],
            "generic_status": plain.status, "plugin_status": smart.status, "held_out": sm["entities_held_out"]}


# ---- 3. patients: how long the next stay is ----------------------------------------------------
def patients(seed: int) -> dict:
    """Length of stay grows with how many admissions a patient had in the last 90 days."""
    rnd = random.Random(seed)
    ents, evs = [], []
    by: dict[str, list] = {}
    for i in range(300):
        pid = f"P{i}"
        ents.append(Entity(pid, group=f"hospital-{i % 2}"))
        t = NOW - timedelta(days=700 - rnd.randint(0, 60))
        mine = by.setdefault(pid, [])
        while t < NOW:
            recent = sum(1 for e in mine if e.at >= t - timedelta(days=90))
            e = Event(pid, t, "admission", max(1.0, rnd.gauss(2 + 2.5 * recent, 0.8)))
            mine.append(e)
            evs.append(e)
            t += timedelta(days=rnd.choice([20, 35, 60, 120]))
    task = predict_value(Dataset(ents, evs), target="admission", window_days=120)
    m = task.report()["metrics"]
    return {"status": task.status, "model_mae_days": m["mae"], "baseline_mae_days": m["baseline_mae"], "r2": m["r2"]}


# ---- 4. pure noise: nothing to learn -----------------------------------------------------------
def noise(seed: int) -> dict:
    """A memoryless process: every entity has the same 4% chance of an event each week, independent of
    its past. Nothing in the history predicts the future, so the honest AUC is 0.5. Five random datasets
    per seed, each scored with the default gate (the model must not be worse than the baseline) and with
    the 90% confidence check (the lower end of the gain interval must be above zero)."""
    out = {"datasets": 0, "promoted_default": 0, "promoted_confidence_90": 0, "shipped_worse_than_baseline": 0}
    aucs = []
    for k in range(5):
        rnd = random.Random(seed * 10 + k)
        ents, evs = [Entity(f"E{i}") for i in range(300)], []
        for i in range(300):
            d = NOW - timedelta(days=700)
            while d < NOW:
                if rnd.random() < 0.04:
                    evs.append(Event(f"E{i}", d, "ping"))
                d += timedelta(days=7)
        ds = Dataset(ents, evs)
        t = predict_event(ds, target="ping", window_days=60)
        m = t.report().get("metrics") or {}
        out["datasets"] += 1
        out["promoted_default"] += t.status == "trained"
        out["shipped_worse_than_baseline"] += t.status == "trained" and m["auc"] < m["baseline_auc"]
        out["promoted_confidence_90"] += predict_event(ds, target="ping", window_days=60, confidence=0.9).status == "trained"
        aucs.append(m.get("auc") or 0.5)
    out["model_auc"] = round(sum(aucs) / len(aucs), 4)
    return out


# ---- 5. uplift: who a message actually brings back ---------------------------------------------
def _send_world(seed: int, n: int = 800):
    """One business. A send goes out at S; 25% are held out as control. Baseline return chance in the
    14 days after: 30%. Planted: a message adds +30 points, but only for customers overdue against
    their own rhythm; for everyone else it does nothing. Half the messages carry an offer, and an offer
    shrinks the bill of the visit it brings by 20%."""
    rnd = random.Random(seed)
    S = NOW - timedelta(days=30)
    visits, sends, truth = {}, [], {}
    for i in range(n):
        cid = f"c{i}"
        gap = rnd.choice([7, 10, 14, 21])
        last_gap_mult = rnd.choice([0.5, 0.8, 1.0, 2.0, 3.0])        # how overdue they are at S
        t = S - timedelta(days=int(gap * last_gap_mult)) - timedelta(days=gap * rnd.randint(4, 12))
        vs = []
        usual = rnd.uniform(250, 900)
        while t <= S - timedelta(days=int(gap * last_gap_mult)):
            vs.append(Visit(t, usual * rnd.uniform(0.8, 1.2)))
            t += timedelta(days=gap)
        overdue = (S - vs[-1].at).days > 1.5 * gap
        messaged = rnd.random() >= 0.25
        offer = messaged and rnd.random() < 0.5
        p = 0.30 + (0.30 if messaged and overdue else 0.0)
        if rnd.random() < p:
            vs.append(Visit(S + timedelta(days=rnd.randint(1, 13)), usual * (0.8 if offer else 1.0) * rnd.uniform(0.9, 1.1)))
        visits[cid] = vs
        sends.append(uplift.Send(cid, S, messaged, offer, "winback"))
        truth[cid] = overdue
    h = churn.History("shop", "cafe", 30, 90, visits)
    return uplift.examples([(h, 14, sends)]), truth


def uplift_targeting(seed: int) -> dict:
    ex, _truth = _send_world(seed)
    out = uplift.train(ex)
    m = out["metrics"]
    bill = value.train_bill(ex)
    return {"status": out["status"], "lift_top_half": m.get("observed_lift_top_half"),
            "lift_bottom_half": m.get("observed_lift_bottom_half"), "planted_lift_overdue": 0.30,
            "offer_effect_pct": (bill.get("metrics") or {}).get("offer_effect_pct"), "planted_offer_effect_pct": -20.0}


# ---- 6. insights: a real pattern, and no false ones in noise ------------------------------------
def _first_item_world(seed: int, planted: bool):
    rnd = random.Random(seed)
    visits, first = {}, {}
    menu = ["Filter Coffee", "Masala Chai", "Cold Coffee", "Veg Puff", "Brownie", "Idli", "Dosa", "Lime Soda"]
    for i in range(600):
        cid = f"c{i}"
        start = NOW - timedelta(days=rnd.randint(60, 300))
        items = rnd.sample(menu, 2)
        p = 0.45
        if planted and "Filter Coffee" in items:
            p = 0.80
        vs = [Visit(start, rnd.uniform(150, 600))]
        if rnd.random() < p:
            vs.append(Visit(start + timedelta(days=rnd.randint(3, 40)), rnd.uniform(150, 600)))
        visits[cid], first[cid] = vs, items
    return insights.find(visits, 45, first_items=first)


def insight_discovery(seed: int) -> dict:
    found = _first_item_world(seed, planted=True)
    null = _first_item_world(seed + 1000, planted=False)
    hit = any("Filter Coffee" in i["title"] and "come back" in i["title"] for i in found["insights"])
    false_null = sum(1 for i in null["insights"] if i.get("kind") in ("opportunity", "risk", "pattern")
                     and "First-timers who order" in i.get("title", ""))
    return {"found_planted": hit, "tested": found["tested"], "false_item_findings_in_noise": false_null}


# ---- 7. mapping: the data repair never damages a correct mapping -------------------------------
def mapping_repair(seed: int) -> dict:
    sys.path.insert(0, str(ROOT / "lm"))
    from make_data import D, example
    from vertiq.mapping import Mapping, repair
    r = random.Random(seed)
    changed = n = 0
    for dom in D:
        for _ in range(140):
            _p, t = example(dom, r)
            rows = t.pop("_rows")
            m = Mapping.parse(t)
            changed += repair(m, rows)[0].as_dict() != m.as_dict()
            n += 1
    return {"correct_mappings": n, "changed_by_repair": changed}


# ---- 8. non-linear effects: what the boosted learner adds -------------------------------------
def interactions(seed: int) -> dict:
    """400 assets with an average operating temperature. Failure risk is low inside a 40-60 degree band and
    high outside it on either side: a U-shaped effect a linear model cannot express. The temperature is
    given to every learner as a plugin feature."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(400):
        temp = rnd.uniform(20, 80)
        ents.append(Entity(f"A{i}", attrs={"temp": temp}))
        risk = 0.10 if (temp < 40 or temp > 60) else 0.015
        d = NOW - timedelta(days=720)
        while d < NOW:
            if rnd.random() < risk:
                evs.append(Event(f"A{i}", d, "breakdown", 1000.0))
            d += timedelta(days=7)
        evs.append(Event(f"A{i}", NOW - timedelta(days=730), "install"))
    ds = Dataset(ents, evs)
    out = {}
    for learner in ("logistic", "boosted", "auto"):
        plugins.clear()

        @plugins.feature("assets")
        def temperature(entity, before, cut):
            return {"temp": float(entity.attrs.get("temp", 50.0))}
        t = predict_event(ds, target="breakdown", window_days=30, pack="assets", learner=learner)
        m = t.report()["metrics"]
        out[f"{learner}_auc"] = m["auc"]
        if learner == "auto":
            out["auto_chose_boosted"] = m["learner"] == "BoostedClassifier"
    plugins.clear()
    out["baseline_auc"] = m["baseline_auc"]
    return out


# ---- 9. survival: time-to-event on rhythmic data ------------------------------------------------
def renewal_survival(seed: int) -> dict:
    """400 customers with individual purchase rhythms (5 to 40 days); half eventually stop."""
    rnd = random.Random(seed)
    ents, evs = [], []
    for i in range(400):
        gap = rnd.choice([5, 10, 20, 40])
        ents.append(Entity(f"C{i}", group=f"g{i % 2}"))
        t = NOW - timedelta(days=600)
        stop = NOW + timedelta(days=999) if rnd.random() < 0.5 else NOW - timedelta(days=rnd.randint(30, 400))
        while t < min(NOW, stop):
            evs.append(Event(f"C{i}", t, "order", 100.0))
            t += timedelta(days=max(1, round(rnd.gauss(gap, gap * 0.3))))
    ds = Dataset(ents, evs)
    plain = predict_event(ds, target="order", window_days=21).report()["metrics"]
    surv = predict_event(ds, target="order", window_days=21, survival=True).report()["metrics"]
    return {"baseline_auc": plain["baseline_auc"], "plain_auc": plain["auc"], "survival_auc": surv["auc"],
            "survival_better": surv["auc"] > plain["auc"]}


# ---- 10. uplift on any action ---------------------------------------------------------------------
def action_uplift(seed: int) -> dict:
    """1,500 machines; a maintenance programme reduces 30-day failure from 50% to 25% for old machines
    and does nothing for new ones; 30% are held out as the control group."""
    from vertiq.contract import Action
    from vertiq.tasks import predict_uplift
    rnd = random.Random(seed)
    ents, evs, acts = [], [], []
    for i in range(1500):
        mid, old = f"M{i}", rnd.random() < 0.5
        ents.append(Entity(mid, attrs={"old": old}))
        evs.append(Event(mid, NOW - timedelta(days=800), "install"))       # every asset has an install record
        for k in range(1, 6):
            if rnd.random() < (0.5 if old else 0.3):
                evs.append(Event(mid, NOW - timedelta(days=60 + 70 * k), "breakdown"))
        when, treated = NOW - timedelta(days=45), rnd.random() >= 0.3
        acts.append(Action(mid, when, treated, "programme"))
        p = 0.25 if (treated and old) else (0.5 if old else 0.2)
        if rnd.random() < p:
            evs.append(Event(mid, when + timedelta(days=rnd.randint(1, 29)), "breakdown"))
    plugins.clear()

    @plugins.feature("fleet")
    def age(entity, before, cut):
        return {"old": 1.0 if entity.attrs.get("old") else 0.0}
    task = predict_uplift(Dataset(ents, evs, acts), target="breakdown", window_days=30, pack="fleet")
    plugins.clear()
    m = task.report()["metrics"]
    out = {"status": task.status, "average_effect": m["average_effect"]["effect"],
           "lift_top_half": m["lift_top_half"], "lift_bottom_half": m["lift_bottom_half"]}
    out["estimated_effect_old"] = out["estimated_effect_new"] = None
    if task.status == "trained":
        for key, want in (("estimated_effect_old", True), ("estimated_effect_new", False)):
            sc = [s["uplift"] for e in ents if e.attrs["old"] is want if (s := task.score(e.id))][:200]
            out[key] = round(sum(sc) / len(sc), 4) if sc else None
    return out


# ---- 11. matched cohorts without randomisation ----------------------------------------------------
def matched_cohort(seed: int) -> dict:
    """1,000 entities; treatment is more likely when a covariate is high, and the covariate also raises
    the outcome. True effect +1.0; the naive comparison is confounded."""
    from vertiq.causal import matched_att
    rnd = random.Random(seed)
    recs = []
    for _ in range(1000):
        x = rnd.gauss(0, 1)
        t = 1 if rnd.random() < 1 / (1 + math.exp(-1.5 * x)) else 0
        recs.append({"t": t, "x": x, "y": 2.0 * x + 1.0 * t + rnd.gauss(0, 0.3)})
    r = matched_att(recs, treatment="t", outcome="y", features=["x"])
    naive = (sum(z["y"] for z in recs if z["t"]) / sum(z["t"] for z in recs)
             - sum(z["y"] for z in recs if not z["t"]) / sum(1 - z["t"] for z in recs))
    return {"true_effect": 1.0, "matched_att": r["att"], "naive_difference": round(naive, 4),
            "interval_covers_truth": r["interval_90"][0] <= 1.0 <= r["interval_90"][1]}


# ---- 12. sequential testing -----------------------------------------------------------------------
def sequential(seed: int) -> dict:
    """Baseline rate 10%, smallest worthwhile rate 15%, alpha 0.05, beta 0.2. 300 streams with no
    improvement and 300 with a 15% rate; compared with the observations a fixed-sample test of the same
    rate against the same baseline would need."""
    from vertiq.sequential import SequentialTest
    rnd = random.Random(seed)

    def runs(rate):
        dec, n = [], []
        for _ in range(300):
            s = SequentialTest(0.10, 0.15, alpha=0.05, beta=0.2)
            dec.append(s.run(1 if rnd.random() < rate else 0 for _ in range(20000)))
            n.append(s.n)
        return dec, n
    d0, n0 = runs(0.10)
    d1, n1 = runs(0.15)
    # the fair comparison is a fixed-sample test of one rate against the same known baseline (one sample),
    # at the same one-sided error rates the sequential test uses
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    p0, p1 = 0.10, 0.15
    fixed = math.ceil(((z(0.95) * math.sqrt(p0 * (1 - p0)) + z(0.8) * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2)
    return {"false_alarm_rate": round(sum(d == "accept_h1" for d in d0) / 300, 4),
            "detection_rate": round(sum(d == "accept_h1" for d in d1) / 300, 4),
            "mean_observations": round((sum(n0) + sum(n1)) / 600, 1), "fixed_sample_equivalent": fixed}


# ---- 13. drift --------------------------------------------------------------------------------------
def drift_detection(seed: int) -> dict:
    """Feature rows from the same distribution, and with one feature shifted by one standard deviation."""
    from vertiq import drift
    rnd = random.Random(seed)
    ref = [{"x": rnd.gauss(0, 1), "y": rnd.gauss(0, 1)} for _ in range(2000)]
    same = [{"x": rnd.gauss(0, 1), "y": rnd.gauss(0, 1)} for _ in range(2000)]
    moved = [{"x": rnd.gauss(1, 1), "y": rnd.gauss(0, 1)} for _ in range(2000)]
    a, b = drift.check(ref, same), drift.check(ref, moved)
    return {"stable_verdict": a["verdict"], "shifted_verdict": b["verdict"], "shifted_feature": b["features"][0]["feature"],
            "shifted_psi": b["features"][0]["psi"], "stable_max_psi": a["features"][0]["psi"]}


# ---- 14. lagged sequences ---------------------------------------------------------------------------
def lagged_sequences(seed: int) -> dict:
    """500 customers, 6 orders each; 60% of Apparel orders are followed by Cosmetics 18 to 22 days later.
    A matched dataset with random categories checks for false patterns."""
    from vertiq import sequences

    def world(s, planted):
        rnd = random.Random(s)
        ents, evs = [], []
        for c in range(500):
            ents.append(Entity(f"C{c}"))
            t, last = NOW - timedelta(days=500), None
            for _ in range(6):
                cat = rnd.choice(["Apparel", "Grocery", "Home", "Cosmetics"])
                if planted and last == "Apparel" and rnd.random() < 0.6:
                    cat, t = "Cosmetics", t + timedelta(days=rnd.randint(18, 22))
                else:
                    t += timedelta(days=rnd.randint(5, 40))
                evs.append(Event(f"C{c}", t, "order", 50.0, {"Category": cat}))
                last = cat
        return Dataset(ents, evs)
    found = sequences.find(world(seed, True), attr="Category")["patterns"]
    null = sequences.find(world(seed + 1000, False), attr="Category")["patterns"]
    top = found[0] if found else {}
    return {"found_planted": (top.get("from"), top.get("to")) == ("Apparel", "Cosmetics"),
            "median_lag_days": top.get("median_lag_days"), "false_patterns_in_random": len(null)}


SCENARIOS = [("churn_customers", churn_customers), ("machines", machines), ("patients", patients),
             ("noise", noise), ("uplift_targeting", uplift_targeting), ("insight_discovery", insight_discovery),
             ("mapping_repair", mapping_repair), ("interactions", interactions), ("renewal_survival", renewal_survival),
             ("action_uplift", action_uplift), ("matched_cohort", matched_cohort), ("sequential", sequential),
             ("drift_detection", drift_detection), ("lagged_sequences", lagged_sequences)]


def _summary(runs: list[dict]) -> dict:
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        if all(isinstance(v, bool) for v in vals):
            out[k] = f"{sum(vals)}/{len(vals)}"
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = {"mean": round(mean(vals), 4), "sd": round(pstdev(vals), 4)}
        else:
            out[k] = sorted({str(v) for v in vals})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "core.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    started = time.time()
    results = {}
    for name, fn in SCENARIOS:
        if a.only and name != a.only:
            continue
        t0 = time.time()
        runs = [fn(s) for s in range(1, a.seeds + 1)]
        results[name] = {"seeds": a.seeds, "summary": _summary(runs), "runs": runs, "seconds": round(time.time() - t0, 1)}
        print(f"{name:20} {json.dumps(results[name]['summary'])}")
    doc = {"generated": datetime.utcnow().strftime("%Y-%m-%d"), "python": platform.python_version(),
           "machine": platform.machine(), "seconds": round(time.time() - started, 1), "scenarios": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
