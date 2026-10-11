"""Build the docs portal (docs/*.html, docs/guides/*.html) from one layout, the benchmark results and
the sector guides, so every page shares one design and every number comes from a results file.

Run:  python docs/build.py      (pure Python; GitHub Pages serves the docs/ folder as built)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))
from guides import EVIDENCE, GUIDES, snippet  # noqa: E402

CORE = json.loads((ROOT / "benchmarks" / "results" / "core.json").read_text())
LM = json.loads((ROOT / "benchmarks" / "results" / "lm.json").read_text())
REPO = "https://github.com/ideatorlabs/vertiq"
SITE = "https://ideatorlabs.github.io/vertiq/"
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700"
         "&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap")
NAV = [("index.html", "Overview"), ("architecture.html", "Architecture"), ("learning.html", "How it learns"),
       ("lm.html", "Vertiq LM"), ("benchmarks.html", "Benchmarks"), ("guides/index.html", "Sector guides")]
E = html.escape


def S(name):            # a benchmark summary
    return CORE["scenarios"][name]["summary"]


def mean(name, key, nd=3):
    return f"{S(name)[key]['mean']:.{nd}f}"


def page(path: str, title: str, body: str, desc: str) -> None:
    depth = path.count("/")
    up = "../" * depth
    cur = ' aria-current="page"'
    nav = "".join(f'<a href="{up}{href}"{cur if href == path else ""}>{label}</a>' for href, label in NAV)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{E(title)}</title><meta name="description" content="{E(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="icon" href="{up}assets/mark.svg" type="image/svg+xml">
<link rel="stylesheet" href="{FONTS}"><link rel="stylesheet" href="{up}assets/site.css">
</head><body>
<header class="bar"><div class="bar-in"><a class="mark" href="{up}index.html"><img src="{up}assets/mark.svg" alt="" width="28" height="28">vertiq</a><nav class="nav" aria-label="Sections">{nav}</nav></div></header>
<main>
{body}
</main>
<footer><div><span>Vertiq {E(CORE.get('version', '0.1.0'))} · Apache-2.0 · pure Python, no dependencies</span><a href="{REPO}">Source</a><span>Benchmarks generated {E(CORE['generated'])}</span></div></footer>
</body></html>
"""
    out = DOCS / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)


def arrow_defs(i: str) -> str:
    return (f'<defs><marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>'
            f'<marker id="b{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="ah acc"/></marker></defs>')


def box(x, y, w, h, title, lines, cls="box", kicker=None):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" class="{cls}"/>']
    ty = y + 26
    if kicker:
        out.append(f'<text x="{x + 14}" y="{y + 22}" class="k">{E(kicker)}</text>')
        ty = y + 44
    out.append(f'<text x="{x + 14}" y="{ty}" class="t">{E(title)}</text>')
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 14}" y="{ty + 22 + 18 * i}" class="{"s" if ln.startswith("~") else ""}">{E(ln.lstrip("~"))}</text>')
    return "".join(out)


# ---- diagrams ----------------------------------------------------------------------------------
def fig_system() -> str:
    xs = [16, 196, 376, 556, 736, 916]
    w, y, h = 158, 96, 176
    boxes = [("Your data", ["any export:", "CSV, POS, ERP,", "EHR, CRM, sheets"], "box"),
             ("Mapping", ["Vertiq LM proposes", "the rows settle names,", "types and dates"], "box"),
             ("Contract", ["Entity and group", "Event: when, type,", "value, attributes", "Action: held out?"], "box"),
             ("Tasks", ["predict_event", "predict_value", "uplift · affinity", "insights · rules"], "box"),
             ("Evaluation gate", ["held-out entities", "must beat the", "baseline and the", "current model"], "gate"),
             ("Results", ["scores with reasons", "ranked actions", "tested insights", "rule flags"], "box")]
    g = [arrow_defs("s")]
    g.append('<rect x="182" y="50" width="716" height="372" rx="14" fill="none" class="ln dash"/>')
    g.append('<text x="196" y="74" class="k">INSIDE VERTIQ: NO DATABASE, NO NETWORK, NO WEB FRAMEWORK</text>')
    for (t, ls, c), x in zip(boxes, xs):
        g.append(box(x, y, w, h, t, ls, c))
    for i in range(5):
        x1, x2 = xs[i] + w, xs[i + 1] - 2
        acc = i == 4
        g.append(f'<line x1="{x1}" y1="{y + h / 2}" x2="{x2}" y2="{y + h / 2}" class="ln{" acc" if acc else ""}" marker-end="url(#{"b" if acc else "a"}s)"/>')
    g.append(f'<text x="{xs[5] + w / 2}" y="{y - 10}" class="lbl" text-anchor="middle">ships only if better</text>')
    g.append(box(556, 320, 158, 86, "Your knowledge", ["@feature · @rule", "your own learner"], "box dash"))
    g.append(f'<line x1="635" y1="320" x2="635" y2="{y + h + 2}" class="ln" marker-end="url(#as)"/>')
    g.append('<text x="645" y="300" class="lbl">adds</text>')
    g.append(box(736, 320, 158, 86, "Baseline kept", ["~when the model", "~didn't earn its place"], "box dash"))
    g.append(f'<line x1="815" y1="{y + h}" x2="815" y2="318" class="ln dash" marker-end="url(#as)"/>')
    g.append('<text x="825" y="300" class="lbl">otherwise</text>')
    return (f'<figure><div class="scroll"><svg viewBox="0 0 1090 440" role="img" aria-label="Your data is mapped to one contract, '
            f'general tasks build candidate models, and the evaluation gate ships a model only if it beats the baseline on held-out entities.">'
            f'{"".join(g)}</svg></div><figcaption>A host maps its records to one contract. General tasks learn from them, your plugins add what '
            f'the data can\'t say, and the evaluation gate decides: a model reaches you only if it beats the obvious rule, and the model you '
            f'already have, on entities it never saw.</figcaption></figure>')


def fig_cutoffs() -> str:
    g = [arrow_defs("c")]
    g.append('<line x1="40" y1="150" x2="1040" y2="150" class="ln"/>')
    g.append('<text x="40" y="186" class="lbl">two years ago</text><text x="1040" y="186" class="lbl" text-anchor="end">today</text>')
    for x in (90, 150, 230, 300, 360, 470, 520, 610, 690, 760, 905, 980):
        g.append(f'<circle cx="{x}" cy="150" r="6" class="hot"/>')
    for cx, lab in ((440, "cutoff 3"), (620, "cutoff 2"), (800, "cutoff 1")):
        g.append(f'<line x1="{cx}" y1="70" x2="{cx}" y2="220" class="ln dash"/><text x="{cx}" y="62" class="k" text-anchor="middle">{lab}</text>')
    g.append('<rect x="40" y="96" width="760" height="22" rx="4" class="hot"/>')
    g.append('<text x="52" y="112" class="s">features at cutoff 1: only the events before it</text>')
    g.append('<rect x="800" y="232" width="180" height="22" rx="4" class="box"/>')
    g.append('<text x="810" y="248" class="s">label: did it happen in N days?</text>')
    g.append('<path d="M800 226 V246" class="ln"/>')
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 270" role="img" aria-label="At each cutoff, features use only events before it; the label is whether the event happened within the next N days.">'
            f'{"".join(g)}</svg></div><figcaption>Vertiq steps back through its own history. At each cutoff it builds features from what was '
            'known then, and labels the example with what happened in the next window. Every example is a test the model could have failed, '
            'and nothing from the future leaks in.</figcaption></figure>')


def fig_uplift() -> str:
    g = [arrow_defs("u")]
    g.append(box(16, 70, 200, 96, "A campaign", ["~sent to a list"], "box"))
    g.append(box(300, 20, 230, 88, "Messaged", ["~most of the list"], "box"))
    g.append(box(300, 128, 230, 88, "Held out on purpose", ["~the control group"], "hot"))
    g.append(box(620, 70, 230, 96, "Came back?", ["~same window for both"], "box"))
    g.append(box(930, 70, 140, 96, "Lift", ["~messaged minus", "~held out"], "gate"))
    g.append('<path d="M216 118 H258 V64 H298" class="ln" marker-end="url(#au)"/><path d="M258 118 V172 H298" class="ln" marker-end="url(#au)"/>')
    g.append('<path d="M530 64 H575 V110 H618" class="ln" marker-end="url(#au)"/><path d="M530 172 H575 V126 H618" class="ln" marker-end="url(#au)"/>')
    g.append('<line x1="850" y1="118" x2="928" y2="118" class="ln acc" marker-end="url(#bu)"/>')
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 230" role="img" aria-label="A control group held out of every campaign lets Vertiq learn who a message actually brings back.">'
            f'{"".join(g)}</svg></div><figcaption>Uplift is learned as cause, not correlation: a share of every campaign is held out, and the '
            'model learns for whom the gap between messaged and held-out guests is large. In the benchmark the half it ranks highest gains '
            f'{float(S("uplift_targeting")["lift_top_half"]["mean"]) * 100:+.1f} points; the other half gains '
            f'{float(S("uplift_targeting")["lift_bottom_half"]["mean"]) * 100:+.1f}.</figcaption></figure>')


def fig_lm() -> str:
    g = [arrow_defs("l")]
    steps = [("1 · GENERATE", "Tables", ["15 domains, 6 naming", "styles, 5 languages,", "every date format"]),
             ("2 · VERIFY", "Answers load", ["every answer loads", "its own rows, or", "it is thrown away"]),
             ("3 · FINE-TUNE", "LoRA on a Mac", ["Qwen 3.5 4B, 4-bit", "M1 Pro, 16 GB"]),
             ("4 · SCORE", "114 held-out", ["4 domains never", "seen in training"]),
             ("5 · PUNISH", "Own mistakes", ["wrong answers become", "rejected pairs: ORPO"])]
    xs = [16, 230, 444, 658, 872]
    for (k, t, ls), x in zip(steps, xs):
        g.append(box(x, 40, 192, 140, t, ls, "gate" if k.startswith("4") else "box", kicker=k))
    for i in range(4):
        g.append(f'<line x1="{xs[i] + 192}" y1="110" x2="{xs[i + 1] - 2}" y2="110" class="ln" marker-end="url(#al)"/>')
    g.append('<path d="M968 180 V226 H540 V182" class="ln acc" marker-end="url(#bl)"/>')
    g.append('<text x="754" y="246" class="lbl" text-anchor="middle">retrain; kept only if it scores better</text>')
    g.append(box(16, 270, 300, 110, "Teacher", ["~Qwen 3.5 9B (or an open API model)", "~writes new tables; kept only if", "~they load and it agrees with itself"], "box dash", kicker="6 · TEACH"))
    g.append('<path d="M166 270 V182" class="ln dash" marker-end="url(#al)"/>')
    g.append(box(444, 270, 620, 110, "At use: the data settles what it can", ["~column names snapped to real columns, roles held to their value types (time holds dates,",
                                                                          "~the entity isn't a date, the value is a number), the date format decided by the rows"], "hot", kicker="7 · REPAIR"))
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 400" role="img" aria-label="Vertiq LM pipeline: generate tables, verify them, fine-tune, score on held-out domains, punish its own mistakes, learn from a verified teacher, and repair answers with the data at use time.">'
            f'{"".join(g)}</svg></div><figcaption>The teacher is never trusted on its word: its tables must load with its own mapping, and '
            'it must map them blind and agree with itself. Closed models (OpenAI, Anthropic, Google) are refused as teachers.</figcaption></figure>')


def fig_mapping(g: dict) -> str:
    m = g["mapping"]
    pairs = [(m["entity"], "Entity · id"), (m["time"], "Event · when")]
    if m["type"]:
        pairs.append((m["type"], "Event · type"))
    if m["value"]:
        pairs.append((m["value"], "Event · value"))
    if m["group"]:
        pairs.append((m["group"], "Entity · group"))
    pairs += [(a, "Event · attributes") for a in m["attrs"]]
    h = 30 + 44 * len(pairs)
    roles = list(dict.fromkeys(r for _c, r in pairs))
    ry = {r: 30 + 44 * i + 22 for i, r in enumerate(roles)}
    out = [arrow_defs("m" + g["slug"][:3])]
    out.append('<text x="16" y="18" class="k">YOUR COLUMNS</text><text x="620" y="18" class="k">VERTIQ CONTRACT</text>')
    for i, (col, role) in enumerate(pairs):
        y = 30 + 44 * i
        out.append(f'<rect x="16" y="{y}" width="300" height="34" rx="7" class="box"/><text x="30" y="{y + 22}" class="m" style="font-family:var(--mono);font-size:13px">{E(col)}</text>')
        out.append(f'<path d="M316 {y + 17} C 470 {y + 17}, 470 {ry[role]}, 618 {ry[role]}" class="ln" marker-end="url(#am{g["slug"][:3]})"/>')
    for r in roles:
        out.append(f'<rect x="620" y="{ry[r] - 17}" width="240" height="34" rx="7" class="hot"/><text x="634" y="{ry[r] + 5}">{E(r)}</text>')
    out.append(f'<text x="880" y="{ry[roles[1]] + 5}" class="s">{E(m["date_format"])}</text>')
    return (f'<figure><div class="scroll"><svg viewBox="0 0 1080 {h + 10}" role="img" aria-label="How {E(g["name"])} columns map to the Vertiq contract.">'
            f'{"".join(out)}</svg></div><figcaption>Map once: Vertiq LM can propose this from the header and three rows, and the rows '
            'settle what they can (names, value types, the date format) before anything is trusted.</figcaption></figure>')


def bars(rows: list[tuple[str, list[tuple[str, float, str]]]], vmax: float, unit: str, label: str) -> str:
    """Grouped horizontal bars: rows of (name, [(series, value, class)])."""
    per = max(len(s) for _n, s in rows)
    rh = 20 * per + 18
    h = 24 + rh * len(rows) + 30
    x0, x1 = 300, 1000
    X = lambda v: x0 + (x1 - x0) * v / vmax      # noqa: E731
    g = []
    for k in range(0, 6):
        v = vmax * k / 5
        g.append(f'<line x1="{X(v):.1f}" y1="16" x2="{X(v):.1f}" y2="{h - 26}" class="grid-l"/>'
                 f'<text x="{X(v):.1f}" y="{h - 8}" class="s" text-anchor="middle">{v:.0f}{unit}</text>')
    for i, (name, series) in enumerate(rows):
        y = 24 + rh * i
        g.append(f'<text x="{x0 - 12}" y="{y + 10 * per + 4}" text-anchor="end">{E(name)}</text>')
        for j, (sname, v, cls) in enumerate(series):
            yy = y + 20 * j
            g.append(f'<rect x="{x0}" y="{yy}" width="{max(1.5, X(v) - x0):.1f}" height="15" rx="3" class="{cls}"><title>{E(sname)}: {v:.1f}{unit}</title></rect>'
                     f'<text x="{X(v) + 6:.1f}" y="{yy + 12}" class="s">{v:.1f}{unit}</text>')
    return f'<svg viewBox="0 0 1080 {h}" role="img" aria-label="{E(label)}">{"".join(g)}</svg>'


# ---- pages -------------------------------------------------------------------------------------
def lm_run(name):
    return next(r for r in LM["runs"] if r["run"] == name)


def build_index():
    best = lm_run("v2-sft")
    body = f"""
<section class="read">
  <div class="kicker">Local decision models</div>
  <h1>Decision models for any business, trained on its own outcomes</h1>
  <p class="lede">Vertiq predicts what happens next to a customer, a machine, a patient or an account. It measures what actually
  worked against a control group, and it only ships a model that beats the obvious rule on records it never saw. Pure Python, no
  dependencies; it runs on a laptop, a server or a till.</p>
</section>
<section aria-label="Benchmark highlights">
  <div class="stats">
    <div class="stat"><b>{mean('machines', 'plugin_auc')}</b><span>AUC predicting breakdowns with one domain plugin, against {mean('machines', 'baseline_auc')} for the recency rule</span></div>
    <div class="stat"><b>{S('uplift_targeting')['offer_effect_pct']['mean']:+.1f}%</b><span>offer effect on the bill recovered, planted at −20%</span></div>
    <div class="stat"><b>{S('noise')['shipped_worse_than_baseline'].split('/')[0]} of {CORE['scenarios']['noise']['seeds']}</b><span>models shipped on pure noise: the gate kept the baseline</span></div>
    <div class="stat"><b>{best['overall']['loads_repaired'] * 100:.1f}%</b><span>of held-out tables Vertiq LM maps so every row loads, after the data repair (4B model, on a Mac)</span></div>
  </div>
</section>
<section>
  <h2>Three ideas</h2>
  <div class="grid">
    <div class="card"><h3>Map once</h3><p>Any sector's records become entities, events and actions. Vertiq LM proposes the mapping from a header and three rows; the rows correct it.</p></div>
    <div class="card"><h3>Ask general questions</h3><p>Will it happen within N days? How much will the next one be worth? Who does a message bring back? The same tasks answer them in every sector.</p></div>
    <div class="card"><h3>Earn its place</h3><p>Every model is tested on entities it never saw, against the obvious rule and the model you already run. If it doesn't win, you keep what you had.</p></div>
  </div>
</section>
<section>
  <h2>Quick start</h2>
  <pre><code>{E('''pip install git+https://github.com/ideatorlabs/vertiq

from vertiq.mapping import Mapping, apply, repair
from vertiq.tasks import predict_event

m, notes = repair(Mapping.parse({"entity": "Equipment No", "time": "Work Order Date", "type": "Order Type",
                                 "value": "Cost", "group": "Plant", "attrs": [], "date_format": "%Y-%m-%d"}), rows)
ds, problems = apply(m, rows)
task = predict_event(ds, target="breakdown", window_days=30)
print(task.status, task.report()["metrics"])   # trained, kept (baseline wins) or learning (too little data)
print(task.score("EQ-602"))                    # {"chance": 0.31, "raises": [...], "lowers": [...]}''')}</code></pre>
</section>
<section>
  <h2>Where to go next</h2>
  <div class="grid">
    <a class="card" href="architecture.html"><h3>Architecture</h3><p>The contract, the tasks, the gate, and the boundary that keeps Vertiq independent of any product.</p></a>
    <a class="card" href="learning.html"><h3>How it learns</h3><p>No peeking at the future, held-out entities, control groups, tested insights.</p></a>
    <a class="card" href="lm.html"><h3>Vertiq LM</h3><p>An open 4B model fine-tuned on a Mac to map any table, punished for its own mistakes.</p></a>
    <a class="card" href="benchmarks.html"><h3>Benchmarks</h3><p>Seven scenarios with a planted truth, five seeds each, and every LM run.</p></a>
    <a class="card" href="guides/index.html"><h3>Sector guides</h3><p>{len(GUIDES)} sectors: how their records map, which questions apply, what has been tested.</p></a>
  </div>
</section>"""
    page("index.html", "Vertiq", body, "Decision models for any business, trained on its own outcomes.")


def build_architecture():
    mods = [("mapping", "Any table to the contract; repair() lets the rows settle names, value types and dates"),
            ("contract", "Entity, Event, Action, Dataset: the one shape every task reads"),
            ("tasks", "predict_event and predict_value for any event type, with reasons for each score"),
            ("harness", "The evaluation gate: held-out entities, baseline and previous-model comparison"),
            ("plugins", "@feature, @rule and register_learner: domain knowledge without forking"),
            ("rules", "repeat, overdue, exceeds, too_short: audit checks set with data, not code"),
            ("churn · uplift · value", "Customer return, who a message brings back, bill size and campaign revenue"),
            ("affinity · offers", "Bought together, next item, refill rhythms, spend-more offers"),
            ("insights", "Patterns tested with false-discovery control before they are shown"),
            ("inventory", "Reorder, slow movers, new arrivals, margins"),
            ("learn", "Logistic and ridge models, AUC, Brier, calibration, Benjamini-Hochberg"),
            ("verticals", "Profiles: windows, nouns and currency per vertical"),
            ("dataset · research · trends", "Training rows for a language model, cited tactics, trend matching")]
    rows = "".join(f"<tr><td><code>{E(a)}</code></td><td>{E(b)}</td></tr>" for a, b in mods)
    body = f"""
<section class="read"><div class="kicker">Architecture</div><h1>One contract, general tasks, one gate</h1>
<p class="lede">Vertiq never reads a database or calls a network. A host maps its records once, calls functions, and stores the
models (plain dicts) wherever it likes. That boundary is a test: <code>tests/test_boundary.py</code> fails if the package imports a web
framework, an ORM or an HTTP client.</p></section>
<section>{fig_system()}</section>
<section><h2>Modules</h2><div class="tbl"><table><thead><tr><th>Module</th><th>What it does</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section class="two">
  <div class="card"><h3>Extend without forking</h3><p>Register features and rules under a pack name and pass <code>pack=</code> to a task. A feature
  sees only events before the cutoff, so it can't leak the future. Any class with fit, predict, to_dict and from_dict can be a learner.</p></div>
  <div class="card"><h3>Run anywhere</h3><p>No dependencies at all, Python 3.10+. Training on a few hundred thousand rows takes seconds on a laptop; scoring at a
  till is a few dictionary lookups.</p></div>
  <div class="card"><h3>Embed it in any product</h3><p>A host keeps only glue on its side: reading its database into the contract and serving results. Nothing in this package knows which product embeds it.</p></div>
</section>"""
    page("architecture.html", "Vertiq architecture", body, "How Vertiq maps any data to one contract and ships only models that earn their place.")


def build_learning():
    body = f"""
<section class="read"><div class="kicker">How it learns</div><h1>Every model is a test it could have failed</h1>
<p class="lede">Three rules run through every task: features use only the past, a share of entities is held out entirely, and a model
replaces the baseline only when it wins on those held-out entities.</p></section>
<section><h2>No peeking at the future</h2>{fig_cutoffs()}</section>
<section class="two">
  <div class="card"><h3>Held out by entity</h3><p>About a fifth of entities, chosen by a hash of their id, never appear in training. A machine or a guest
  is never in both halves, so the score says how the model does on someone new.</p></div>
  <div class="card"><h3>Against the obvious rule</h3><p>For events, the baseline is recency: the more recently it happened, the likelier again. For values,
  it is "same as last time". Vertiq reports both, and ships only when it wins.</p></div>
  <div class="card"><h3>Against the model you run</h3><p>A retrained model must also be no worse than the current one on the same held-out entities,
  so a bad week of data can't replace a good model.</p></div>
</section>
<section><h2>Cause, not correlation</h2>{fig_uplift()}</section>
<section class="read"><h2>Insights are tested before they are shown</h2>
<p>Vertiq checks many patterns at once (the day of a first visit, the size of a first bill, what was ordered first). Each candidate gets
a p-value, and the family is filtered with Benjamini-Hochberg at a 10% false-discovery rate, so the more places it looks, the stronger a
pattern must be. Groups smaller than 20 are never compared. In the benchmark it found the planted pattern in
{S('insight_discovery')['found_planted']} seeds and reported {S('insight_discovery')['false_item_findings_in_noise']['mean']:.0f} false
item patterns in noise.</p></section>
<section class="note"><b>Consent.</b> The host decides who may train a model. Train only on people who agreed, never on anyone who opted out; scoring a business's own customers is use, not training.</section>"""
    page("learning.html", "How Vertiq learns", body, "No peeking, held-out entities, control groups and tested insights.")


def build_lm():
    runs_v1 = [r for r in LM["runs"] if r["test_set"] == "v1"]
    runs_v2 = [r for r in LM["runs"] if r["test_set"] == "v2"]

    def table(runs):
        best = max(r["overall"]["exact_repaired"] for r in runs)
        body = ""
        for r in runs:
            o, u = r["overall"], r["unseen"]
            body += (f'<tr class="{"best" if o["exact_repaired"] == best else ""}"><td>{E(r["model"])}</td>'
                     f'<td class="n">{o["loads"] * 100:.1f}%</td><td class="n">{o["exact"] * 100:.1f}%</td>'
                     f'<td class="n">{o["loads_repaired"] * 100:.1f}%</td><td class="n">{o["exact_repaired"] * 100:.1f}%</td>'
                     f'<td class="n">{u["exact"] * 100:.1f}%</td><td class="n">{u["exact_repaired"] * 100:.1f}%</td></tr>')
        return ('<div class="tbl"><table><thead><tr><th>Model</th><th class="n">Loads</th><th class="n">Exact</th><th class="n">Loads, repaired</th>'
                '<th class="n">Exact, repaired</th><th class="n">Unseen exact</th><th class="n">Unseen, repaired</th></tr></thead>'
                f'<tbody>{body}</tbody></table></div>')
    chart = bars([(r["model"], [("exact", r["overall"]["exact"] * 100, "bar1"), ("exact after repair", r["overall"]["exact_repaired"] * 100, "bar2")])
                  for r in runs_v1], 50, "%", "Exact mappings by model, raw and after the data repair")
    body = f"""
<section class="read"><div class="kicker">Vertiq LM</div><h1>An open 4B model that maps any table, trained on a Mac</h1>
<p class="lede">Vertiq LM's first job is the step every sector needs: reading an unfamiliar export and saying which column is the entity, the time,
the event type, the value and the group. It is Qwen 3.5 4B (Apache-2.0), fine-tuned with LoRA on an M1 Pro with 16 GB, scored on tables from
domains it never saw.</p></section>
<section>{fig_lm()}</section>
<section><h2>Results on the original test (114 tables)</h2>{table(runs_v1)}
<figure>{chart}<figcaption>Grey: exactly right as written. Blue: exactly right after the data repair. The repair never changed a correct mapping in
{S('mapping_repair')['correct_mappings']['mean']:.0f} checks per seed, over {CORE['scenarios']['mapping_repair']['seeds']} seeds.</figcaption></figure></section>
<section><h2>Results on the realistic test (114 tables)</h2>
<p class="read">Real audit columns, amounts in local styles (R$ 1.267,81, € 4 350,50, ₹1,267). The earlier model was given its own prompt format, so
the comparison is fair.</p>{table(runs_v2)}</section>
<section class="two">
  <div class="card"><h3>What worked</h3><p>Showing the model what each column holds (read from the values by code) lifted never-seen domains
  from {lm_run('pref-on-v2')['unseen']['exact'] * 100:.1f}% to {lm_run('v2-sft')['unseen']['exact'] * 100:.1f}% exactly right. Letting the rows decide the date
  format and the value types lifted every model.</p></div>
  <div class="card"><h3>What didn't, yet</h3><p>DPO ran out of memory on 16 GB because it keeps a reference copy; ORPO replaced it. The round-3 ORPO step
  also ran out of memory with other apps open and is waiting for a free machine. Constrained decoding was measured and dropped: it would fix
  2 of 88 remaining errors.</p></div>
  <div class="card"><h3>Reproduce</h3><p><code>lm/make_data.py</code> → <code>lm/train.sh</code> → <code>lm/evaluate.py</code> →
  <code>lm/collect_failures.py</code> → <code>lm/pref.sh</code>; <code>lm/round3.sh</code> runs the realistic round end to end.</p></div>
</section>"""
    page("lm.html", "Vertiq LM", body, "A Qwen 3.5 4B model fine-tuned on a Mac to map any table to Vertiq's contract.")


def build_benchmarks():
    def pm(name, key, nd=3, scale=1.0, unit=""):
        v = S(name)[key]
        return f"{v['mean'] * scale:.{nd}f}{unit} <span class='s'>± {v['sd'] * scale:.{nd}f}</span>"
    rows = [
        ("Customer return", "Two cafés, 500 customers, irregular rhythms; two in three leavers slow down first",
         f"AUC {pm('churn_customers', 'model_auc')} vs {pm('churn_customers', 'baseline_auc')}; Brier {pm('churn_customers', 'model_brier')} vs {pm('churn_customers', 'baseline_brier')}",
         "Ranks about as well as the recency formula; its probabilities are about 30% more accurate"),
        ("Machine breakdowns", "240 machines; each skipped maintenance in 60 days triples the breakdown chance",
         f"AUC {pm('machines', 'baseline_auc')} baseline → {pm('machines', 'generic_auc')} general → {pm('machines', 'plugin_auc')} with a plugin",
         "The general task finds signal; one line of domain knowledge adds more"),
        ("Length of stay", "300 patients; stays grow with recent admissions",
         f"Error {pm('patients', 'model_mae_days', 2)} days vs {pm('patients', 'baseline_mae_days', 2)} for same-as-last-time",
         "Value prediction beats the obvious rule"),
        ("Pure noise", "300 entities, a memoryless 4% weekly event: nothing to learn",
         f"Models shipped: {S('noise')['shipped_worse_than_baseline'].split('/')[0]} of {CORE['scenarios']['noise']['seeds']}; status {', '.join(S('noise')['status'])}",
         "The gate refuses to ship a model that doesn't beat the baseline"),
        ("Who a message brings back", "800 customers; a message adds 30 points only for the overdue; 25% held out",
         f"Lift in the top-ranked half {pm('uplift_targeting', 'lift_top_half', 1, 100, ' pts')}; bottom half {pm('uplift_targeting', 'lift_bottom_half', 1, 100, ' pts')}",
         "Uplift ranking separates who the message helps"),
        ("Offer effect on the bill", "Half the messages carry an offer that shrinks the bill by 20%",
         f"Measured {pm('uplift_targeting', 'offer_effect_pct', 1, 1, '%')} (planted −20%)", "The bill model recovers the planted effect"),
        ("Tested insights", "600 first-timers; one first-order item doubles returns; a twin dataset with none",
         f"Planted pattern found {S('insight_discovery')['found_planted']}; false item patterns in noise {S('insight_discovery')['false_item_findings_in_noise']['mean']:.0f}",
         "False-discovery control holds"),
        ("Mapping repair safety", "2,100 correct mappings across 15 domains",
         f"Changed by the repair: {S('mapping_repair')['changed_by_repair']['mean']:.0f}", "The repair never damages a correct answer"),
    ]
    tr = "".join(f"<tr><td><b>{E(a)}</b><br><span class='s' style='color:var(--muted)'>{E(b)}</span></td><td>{c}</td><td>{E(d)}</td></tr>" for a, b, c, d in rows)
    m = S("machines")
    chart = bars([("Recency baseline", [("AUC", m["baseline_auc"]["mean"] * 100, "bar0")]),
                  ("General task", [("AUC", m["generic_auc"]["mean"] * 100, "bar1")]),
                  ("With one plugin", [("AUC", m["plugin_auc"]["mean"] * 100, "bar2")])], 100, "", "Breakdown prediction AUC (x100)")
    body = f"""
<section class="read"><div class="kicker">Benchmarks</div><h1>Planted truths, five seeds each</h1>
<p class="lede">Every scenario is synthetic with a known answer, so each result can be checked against it. Means ± standard deviation over
{CORE['scenarios']['machines']['seeds']} seeds; the whole suite runs in {CORE['seconds']:.0f} seconds on {E(CORE['machine'])} with Python {E(CORE['python'])}.</p></section>
<section><div class="tbl"><table><thead><tr><th>Scenario</th><th>Result</th><th>What it shows</th></tr></thead><tbody>{tr}</tbody></table></div></section>
<section><h2>What a plugin adds</h2><figure><div class="scroll">{chart}</div><figcaption>Breakdown prediction on held-out machines (AUC × 100). One
feature that counts skipped maintenance carries knowledge the event log doesn't state.</figcaption></figure></section>
<section class="note"><b>What these don't show.</b> Synthetic benchmarks prove the machinery: it finds a real signal, ignores noise and refuses to ship
a worse model. They do not prove accuracy on your data; the evaluation gate measures that on your own held-out records every time it trains.
Public datasets (Online Retail II, the Hillstrom email uplift test, NASA C-MAPSS engines) are next.</section>
<section class="read"><h2>Run them</h2><pre><code>python benchmarks/run.py --seeds 5        # writes benchmarks/results/core.json
python benchmarks/collect_lm.py --lm lm   # summarises scored Vertiq LM runs into results/lm.json
python docs/build.py                      # rebuilds this portal from both</code></pre></section>"""
    page("benchmarks.html", "Vertiq benchmarks", body, "Seven synthetic scenarios with planted truths, and every Vertiq LM run.")


def pills(g):
    seen, out = set(), []
    for lvl, _t in g["evidence"]:
        if lvl not in seen:
            seen.add(lvl)
            label, cls = EVIDENCE[lvl]
            out.append(f'<span class="pill {cls}">{label}</span>')
    return " ".join(out)


def build_guides():
    cards = "".join(f'<a class="card" href="{g["slug"]}.html"><h3>{E(g["name"])}</h3><p>{E(g["blurb"])}</p><div>{pills(g)}</div></a>' for g in GUIDES)
    body = f"""
<section class="read"><div class="kicker">Sector guides</div><h1>One engine, every sector's own questions</h1>
<p class="lede">Vertiq has no sector-specific code. Each guide shows how a sector's records map to the contract, which general tasks answer its
questions, what domain knowledge to add as a plugin, and what has actually been tested.</p></section>
<section><div class="grid">{cards}</div></section>
<section class="read"><h2>Evidence labels</h2><ul>
<li><span class="pill ok">benchmarked</span> a synthetic benchmark with a planted truth covers this case</li>
<li><span class="pill ok">mapping tested</span> Vertiq LM's mapping test includes this sector's tables</li>
<li><span class="pill plain">unit tested</span> covered by the package's unit tests</li>
<li><span class="pill wait">not yet tested</span> no evidence yet; the evaluation gate still measures it on your data</li></ul></section>"""
    page("guides/index.html", "Vertiq sector guides", body, "How each sector's records map to Vertiq and which questions it can answer.")
    for i, g in enumerate(GUIDES):
        prev, nxt = GUIDES[i - 1], GUIDES[(i + 1) % len(GUIDES)]
        qs = [f"<li><b>{E(g['event']['q'])}</b> <code>predict_event(target=\"{E(g['event']['target'])}\", window_days={g['event']['window']})</code>"
              + (f"<br><span style='color:var(--muted)'>{E(g['event']['note'])}</span>" if g['event'].get('note') else "") + "</li>"]
        if g["value"]:
            qs.append(f"<li><b>{E(g['value']['q'])}</b> <code>predict_value(target=\"{E(g['value']['target'])}\", window_days={g['value']['window']})</code></li>")
        mods = "".join(f"<li><code>{E(a)}</code>: {E(b)}</li>" for a, b in g["modules"])
        rules = "".join(f"<li><code>rules.{E(r)}</code>: {E(w)}</li>" for r, w in g["rules"])
        ev = "".join(f"<li><span class='pill {EVIDENCE[l][1]}'>{EVIDENCE[l][0]}</span> {E(t)}</li>" for l, t in g["evidence"])
        cau = "".join(f"<li>{E(c)}</li>" for c in g["cautions"])
        body = f"""
<section class="read"><div class="kicker"><a href="index.html">Sector guides</a> / {E(g['name'])}</div><h1>{E(g['name'])}</h1>
<p class="lede">{E(g['blurb'])}</p><div>{pills(g)}</div></section>
<section><h2>Your records in Vertiq's terms</h2>
<dl class="map"><dt>entity</dt><dd>{E(g['entity'])}</dd><dt>group</dt><dd>{E(g['group'])}</dd>
<dt>profile</dt><dd>{E(g['profile'] or 'generic (set your own windows with verticals.register)')}</dd></dl>
{fig_mapping(g)}</section>
<section class="read"><h2>Questions you can ask</h2><ul>{''.join(qs)}</ul></section>
<section><h2>Working example</h2><pre><code>{E(snippet(g))}</code></pre>
<p class="read" style="color:var(--muted)">This exact code runs in the test suite (<code>tests/test_guides.py</code>) against generated rows in these column names.</p></section>
<section class="two">
  <div class="card"><h3>Domain knowledge: <code>{E(g['plugin']['name'])}</code></h3><p>{E(g['plugin']['why'])}</p></div>
  <div class="card"><h3>Checks</h3><ul>{rules or '<li>None suggested yet.</li>'}</ul></div>
  {f'<div class="card"><h3>Also useful</h3><ul>{mods}</ul></div>' if mods else ''}
</section>
<section class="two">
  <div class="card"><h3>Before you rely on it</h3><ul>{cau}</ul></div>
  <div class="card"><h3>Evidence so far</h3><ul style="list-style:none;padding:0;display:grid;gap:8px">{ev}</ul></div>
</section>
<section class="read"><p><a href="{prev['slug']}.html">← {E(prev['name'])}</a> · <a href="{nxt['slug']}.html">{E(nxt['name'])} →</a></p></section>"""
        page(f"guides/{g['slug']}.html", f"Vertiq for {g['name'].lower()}", body, g["blurb"])


def build_readme():
    """README.md, with every number read from the same results files as the portal."""
    def mm(name, key, nd=3, scale=1.0):
        v = S(name)[key]
        return f"{v['mean'] * scale:.{nd}f} ± {v['sd'] * scale:.{nd}f}"
    core = [
        ("Customer return", f"AUC {mm('churn_customers', 'model_auc')}, Brier {mm('churn_customers', 'model_brier')}",
         f"recency formula: AUC {mm('churn_customers', 'baseline_auc')}, Brier {mm('churn_customers', 'baseline_brier')}"),
        ("Machine breakdowns", f"AUC {mm('machines', 'generic_auc')}; with one plugin {mm('machines', 'plugin_auc')}", f"recency: AUC {mm('machines', 'baseline_auc')}"),
        ("Length of stay", f"error {mm('patients', 'model_mae_days', 2)} days", f"same as last time: {mm('patients', 'baseline_mae_days', 2)} days"),
        ("Pure noise", f"models shipped: {S('noise')['shipped_worse_than_baseline'].split('/')[0]} of {CORE['scenarios']['noise']['seeds']}", "baseline kept every time"),
        ("Who a message brings back", f"top-ranked half +{mm('uplift_targeting', 'lift_top_half', 1, 100)} pts, bottom half {mm('uplift_targeting', 'lift_bottom_half', 1, 100)} pts", "planted: +30 pts for the overdue only"),
        ("Offer effect on the bill", f"{mm('uplift_targeting', 'offer_effect_pct', 1)}%", "planted: −20%"),
        ("Tested insights", f"planted pattern found {S('insight_discovery')['found_planted']}; false patterns in noise {S('insight_discovery')['false_item_findings_in_noise']['mean']:.0f}", "Benjamini-Hochberg at 10%"),
        ("Mapping repair safety", f"correct mappings changed: {S('mapping_repair')['changed_by_repair']['mean']:.0f} of {S('mapping_repair')['correct_mappings']['mean']:.0f} per seed", "must be 0"),
    ]
    core_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in core)

    def lm_rows(test):
        return "\n".join(f"| {r['model']} | {r['overall']['loads'] * 100:.1f}% | {r['overall']['exact'] * 100:.1f}% | "
                         f"{r['overall']['exact_repaired'] * 100:.1f}% | {r['unseen']['exact'] * 100:.1f}% |"
                         for r in LM["runs"] if r["test_set"] == test)
    guides = "\n".join(f"- [{g['name']}]({SITE}guides/{g['slug']}.html): {g['blurb']}" for g in GUIDES)
    text = f"""<p align="center"><img src="docs/assets/logo.svg" alt="Vertiq" width="290"></p>

<p align="center"><b>Decision models for any business, trained on its own outcomes.</b><br>
<a href="{SITE}">Docs portal</a> · <a href="{SITE}benchmarks.html">Benchmarks</a> · <a href="{SITE}guides/">Sector guides</a></p>

Vertiq predicts what happens next to a customer, a machine, a patient or an account, measures what actually worked
against a control group, and only ships a model that beats the obvious rule on records it never saw. It is pure Python
with no dependencies, so it runs on a laptop, a server or a till. Vertiq LM, an open 4B model fine-tuned on a Mac, maps
any sector's export to Vertiq's contract.

- **Map once.** Any table becomes entities, events and actions (`vertiq.mapping`, `vertiq.contract`).
- **Ask general questions.** Will it happen within N days, how much will the next one be worth, who does a message bring
  back (`vertiq.tasks`, `uplift`, `affinity`, `insights`, `rules`).
- **Earn its place.** Every model is scored on held-out entities against the baseline and the current model
  (`vertiq.harness`); if it doesn't win, you keep what you had.

## Benchmarks

Synthetic scenarios with a planted truth, {CORE['scenarios']['machines']['seeds']} seeds each, mean ± sd
(`python benchmarks/run.py`, {CORE['seconds']:.0f} s on {CORE['machine']}, Python {CORE['python']}).

| Scenario | Vertiq | Reference |
|---|---|---|
{core_rows}

These prove the machinery finds real signal, ignores noise and refuses to ship a worse model. They don't prove accuracy on
your data; the evaluation gate measures that on your own held-out records every time it trains. Public datasets (Online
Retail II, Hillstrom email uplift, NASA C-MAPSS) are next.

### Vertiq LM: mapping unseen tables

Qwen 3.5 4B (Apache-2.0, 4-bit MLX), LoRA on an Apple M1 Pro with 16 GB. 114 held-out tables over 15 domains; 4 domains
(energy, insurance, manufacturing, retail sales) never appear in training. "Repaired" = after `vertiq.mapping.repair`,
which lets the rows settle column names, value types and the date format.

Original test set:

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
{lm_rows('v1')}

Realistic test set (real audit columns, amounts in local styles):

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
{lm_rows('v2')}

## Quick start

```bash
pip install git+https://github.com/ideatorlabs/vertiq
```

```python
from vertiq.mapping import Mapping, apply, repair
from vertiq.tasks import predict_event

rows = [...]   # your export as a list of dicts, e.g. list(csv.DictReader(open("work_orders.csv")))
m, notes = repair(Mapping.parse({{"entity": "Equipment No", "time": "Work Order Date", "type": "Order Type",
                                 "value": "Cost", "group": "Plant", "attrs": [], "date_format": "%Y-%m-%d"}}), rows)
ds, problems = apply(m, rows)
task = predict_event(ds, target="breakdown", window_days=30)
print(task.status, task.report()["metrics"])   # trained, kept (the baseline wins) or learning (too little data)
print(task.score("EQ-602"))                    # the chance, and what raises or lowers it
```

## Sector guides

Vertiq has no sector-specific code. Each guide shows how a sector's records map, which questions apply, the domain
knowledge worth adding as a plugin, and what has been tested. Every guide's example runs in `tests/test_guides.py`.

{guides}

## Docs portal

Live at **{SITE}**: architecture, how it learns, Vertiq LM, benchmarks and the sector guides. `docs/` is a static site
built by `python docs/build.py` from the results files, and GitHub Pages serves it from `main`.

## Layout

| Path | What |
|---|---|
| `vertiq/` | the package: contract, mapping, tasks, harness, plugins, rules, churn, uplift, value, affinity, insights, inventory, offers |
| `tests/` | unit tests, the boundary test (no framework, ORM or HTTP imports) and the guide examples |
| `benchmarks/` | `run.py` (planted-truth scenarios), `collect_lm.py`, `results/*.json` |
| `lm/` | Vertiq LM: data generation, LoRA fine-tuning, scoring, preference training, the verified teacher ([lm/README.md](lm/README.md)) |
| `docs/` | the portal and its builder |

## Status

Version 0.1.0.

## Licence

Apache License 2.0, see [LICENSE](LICENSE). The Qwen 3.5 model weights Vertiq LM fine-tunes are not in this repository;
they are downloaded separately and carry their own Apache-2.0 licence.
"""
    (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    for old in ("diagrams.html", "teach-loop.html"):
        (DOCS / old).unlink(missing_ok=True)
    build_index(), build_architecture(), build_learning(), build_lm(), build_benchmarks(), build_guides(), build_readme()
    (DOCS / ".nojekyll").write_text("")
    print("built", len(list(DOCS.rglob("*.html"))), "pages")
