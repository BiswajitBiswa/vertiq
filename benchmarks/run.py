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
    its past. Nothing in the history predicts the future, so the honest AUC is 0.5, and any model
    that ships must not be worse than the baseline."""
    rnd = random.Random(seed)
    ents, evs = [Entity(f"E{i}") for i in range(300)], []
    for i in range(300):
        d = NOW - timedelta(days=700)
        while d < NOW:
            if rnd.random() < 0.04:
                evs.append(Event(f"E{i}", d, "ping"))
            d += timedelta(days=7)
    t = predict_event(Dataset(ents, evs), target="ping", window_days=60)
    m = t.report().get("metrics") or {}
    return {"status": t.status, "model_auc": m.get("auc"), "baseline_auc": m.get("baseline_auc"),
            "shipped_worse_than_baseline": t.status == "trained" and m["auc"] < m["baseline_auc"]}


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


SCENARIOS = [("churn_customers", churn_customers), ("machines", machines), ("patients", patients),
             ("noise", noise), ("uplift_targeting", uplift_targeting), ("insight_discovery", insight_discovery),
             ("mapping_repair", mapping_repair)]


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
