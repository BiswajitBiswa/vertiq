"""Build the docs portal (docs/*.html, docs/guides/*.html) from one layout, the benchmark results and
the sector guides, so every page shares one design and every number comes from a results file.

Run:  python docs/build.py      (pure Python; GitHub Pages serves the docs/ folder as built)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

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
    """Every page carries its own styles and logo, so a page saved on its own still renders."""
    depth = path.count("/")
    up = "../" * depth
    cur = ' aria-current="page"'
    nav = "".join(f'<a href="{up}{href}"{cur if href == path else ""}>{label}</a>' for href, label in NAV)
    css = (DOCS / "assets" / "site.css").read_text()
    mark = (DOCS / "assets" / "mark.svg").read_text().strip()
    logo = mark.replace("<svg ", '<svg width="28" height="28" aria-hidden="true" ', 1).replace(' role="img" aria-label="Vertiq"', "")
    icon = "data:image/svg+xml," + quote(mark)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{E(title)}</title><meta name="description" content="{E(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="icon" href="{icon}" type="image/svg+xml">
<link rel="stylesheet" href="{FONTS}">
<style>
{css}</style>
</head><body>
<header class="bar"><div class="bar-in"><a class="mark" href="{up}index.html">{logo}vertiq</a><nav class="nav" aria-label="Sections">{nav}</nav></div></header>
<main>
{body}
</main>
<footer><div><span>Vertiq {E(CORE.get('version', '0.1.0'))} · Apache-2.0 · pure Python, no dependencies</span><a href="{REPO}">Source on GitHub</a><span>Benchmarks generated {E(CORE['generated'])}</span></div></footer>
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


PLUGIN_EXAMPLE = '''from datetime import timedelta
from vertiq import plugins
from vertiq.tasks import predict_event

@plugins.feature("plants")
def skipped_maintenance(entity, before, cut):
    # `before` holds only events earlier than the cutoff, so a plugin cannot see the future
    return {"skipped_pm_60": sum(1 for e in before
                                 if e.type == "pm" and e.attrs.get("PM Done") == "no"
                                 and e.at >= cut - timedelta(days=60))}

task = predict_event(ds, target="breakdown", window_days=30, pack="plants")'''

PAINS = [
    ("Every source exports data differently",
     "Mapping a new client's tables to your schema is weeks of manual work, and it is redone for every new source.",
     "Vertiq LM proposes the mapping from a header and three rows. The rows then settle what they can prove: real column names, value types and the date format."),
    ("Campaigns take credit for customers who were coming anyway",
     "Most attribution compares people who got a message with people who didn't, so it rewards targeting the already loyal.",
     "Uplift is learned from held-out control groups, so Vertiq ranks people by how much the action changes their behaviour."),
    ("A retrained model can quietly be worse",
     "Models are refreshed on a schedule and promoted without a fair test against what was already running.",
     "The evaluation gate scores every candidate on entities it never saw, against the obvious rule and the current model. It ships only if it wins."),
    ("Dashboards surface patterns that are noise",
     "Slice data enough ways and something always looks significant.",
     "Insights carry a p-value and are filtered with Benjamini-Hochberg false-discovery control before anyone sees them."),
    ("Domain experts know things the data doesn't say",
     "A maintenance engineer knows skipped servicing matters; a clinician knows which codes are chronic. Generic models never hear it.",
     "Plugins add that knowledge as features and rules, in a few lines, without forking the engine."),
    ("Data can't leave the building",
     "Sending customer or patient records to a hosted model is often not allowed, and heavy ML stacks don't fit at the edge.",
     "The core is pure Python with no dependencies and no network calls. Vertiq LM runs locally on a Mac."),
]

QUESTIONS = [
    ("Will it happen?", "A customer returning, a machine failing, a patient readmitted, an invoice paid late.", "tasks.predict_event"),
    ("How much will it be worth?", "The next bill, repair cost, length of stay or renewal.", "tasks.predict_value"),
    ("Who should we act on?", "The people an action actually changes, not the ones who would act anyway.", "uplift"),
    ("What goes together?", "Items bought together, each customer's next item, refill rhythms.", "affinity"),
    ("What in this data is real?", "Patterns tested before they are shown.", "insights"),
    ("What broke the rules?", "Repeats, overdue work, parts beyond the bill of materials, empty notes.", "rules"),
]


def fig_loop() -> str:
    g = [arrow_defs("r")]
    xs, w, h = [16, 286, 556, 826], 230, 116
    top = [("TEACHER", "Writes harder cases", ["~a larger open model,", "~or an API model you choose"], "box"),
           ("VERIFY", "The data checks it", ["~kept only if it loads and", "~the teacher, mapping it", "~blind, agrees with itself"], "box"),
           ("STUDENT", "Learns", ["~Vertiq LM, or a", "~decision model"], "box"),
           ("GATE", "Earns its place", ["~beats the baseline and", "~the current model on", "~held-out records"], "gate")]
    for (k, t, ls, c), x in zip(top, xs):
        g.append(box(x, 30, w, h, t, ls, c, kicker=k))
    for i in range(3):
        g.append(f'<line x1="{xs[i] + w}" y1="88" x2="{xs[i + 1] - 2}" y2="88" class="ln" marker-end="url(#ar)"/>')
    g.append(box(826, 226, w, h, "Become lessons", ["~the model's own wrong", "~answers are punished", "~in the next round"], "box", kicker="MISTAKES"))
    g.append(box(286, 226, w, h, "New outcomes arrive", ["~bills, breakdowns,", "~renewals, returns"], "box dash", kicker="OUTCOMES"))
    g.append('<line x1="941" y1="146" x2="941" y2="224" class="ln acc" marker-end="url(#br)"/>')
    g.append('<text x="951" y="190" class="lbl">after it ships</text>')
    g.append('<path d="M826 284 H686 V148" class="ln" marker-end="url(#ar)"/>')
    g.append('<text x="756" y="276" class="lbl" text-anchor="middle">next round</text>')
    g.append('<path d="M516 284 H626 V148" class="ln dash" marker-end="url(#ar)"/>')
    g.append('<text x="571" y="276" class="lbl" text-anchor="middle">each cycle</text>')
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 360" role="img" aria-label="A teacher writes harder cases, the data verifies them, '
            'the student learns, the gate ships it only if it beats the current model, and its mistakes become the next round\'s lessons.">'
            f'{"".join(g)}</svg></div><figcaption>Each round starts from the model that won the last one. The teacher is never trusted on its word, '
            'the student is punished for its own mistakes, and nothing ships unless it beats what is already running. Closed models whose terms '
            'forbid training on their output are refused as teachers.</figcaption></figure>')


def build_index():
    best = lm_run("v2-sft")
    pains = "".join(f'<div class="card"><h3>{E(t)}</h3><p>{E(p)}</p><p class="fix">{E(f)}</p></div>' for t, p, f in PAINS)
    qs = "".join(f'<div class="card"><h3>{E(q)}</h3><p>{E(d)}</p><p><code>{E(m)}</code></p></div>' for q, d, m in QUESTIONS)
    sectors = " · ".join(f'<a href="guides/{g["slug"]}.html">{E(g["name"])}</a>' for g in GUIDES)
    body = f"""
<section class="hero">
  <div class="kicker">Open source · Apache-2.0 · pure Python</div>
  <h1>Decisions you can trust, learned from your own outcomes</h1>
  <p class="lede">Vertiq predicts what happens next to a customer, a machine, a patient or an account, measures which actions actually
  changed the result, and replaces a model only when the new one wins on records it has never seen. No dependencies, no network calls,
  and a local language model that maps any sector's data.</p>
  <p class="cta"><a class="btn" href="{REPO}">View on GitHub</a><a class="btn ghost" href="benchmarks.html">See the benchmarks</a></p>
</section>
<section aria-label="Benchmark highlights">
  <div class="stats">
    <div class="stat"><b>{mean('machines', 'plugin_auc')}</b><span>AUC on breakdowns with one domain plugin, against {mean('machines', 'baseline_auc')} for the recency rule</span></div>
    <div class="stat"><b>{S('uplift_targeting')['offer_effect_pct']['mean']:+.1f}%</b><span>offer effect on the bill recovered, planted at −20%</span></div>
    <div class="stat"><b>{S('noise')['shipped_worse_than_baseline'].split('/')[0]} of {CORE['scenarios']['noise']['seeds']}</b><span>models shipped on pure noise; the gate kept the baseline</span></div>
    <div class="stat"><b>{best['overall']['loads_repaired'] * 100:.1f}%</b><span>of held-out tables mapped so every row loads (4B model, on a laptop)</span></div>
  </div>
</section>
<section>
  <div class="kicker">The problem</div>
  <h2>Most teams have the data. What they lack is a decision they can defend.</h2>
  <div class="grid pains">{pains}</div>
</section>
<section>
  <div class="kicker">What it answers</div>
  <h2>Six questions, the same in every sector</h2>
  <div class="grid">{qs}</div>
</section>
<section>
  <div class="kicker">Recursive improvement</div>
  <h2>A teacher, a student and a gate</h2>
  <p>Vertiq gets better by repeating one loop. A stronger model writes harder examples, the data verifies every one, the student learns,
  and the gate decides whether the result replaces what is running. The student's own mistakes become the next round's lessons, and
  new outcomes arrive every cycle. On the mapping task this loop took a 4B model from {lm_run('base')['overall']['exact'] * 100:.1f}% to
  {lm_run('v2-sft-on-v1')['overall']['exact'] * 100:.1f}% exactly right on the same held-out tables.</p>
  {fig_loop()}
</section>
<section class="two">
  <div>
    <div class="kicker">Plugins</div>
    <h2>Add what your experts know</h2>
    <p>Features, rules, learners and vertical profiles all register under a pack name. A feature sees only events before the cutoff, so
    it cannot leak the future, and the gate still decides whether it helped. In the benchmark this plugin lifted breakdown AUC from
    {mean('machines', 'generic_auc')} to {mean('machines', 'plugin_auc')}.</p>
    <ul><li><code>@plugins.feature(pack)</code> domain features</li><li><code>@plugins.rule(pack)</code> and <code>vertiq.rules</code> audit checks</li>
    <li><code>plugins.register_learner(name, cls)</code> bring your own model, for example gradient boosting</li>
    <li><code>verticals.register(...)</code> windows and vocabulary for a sector</li><li>teacher endpoints for Vertiq LM, any OpenAI-compatible server</li></ul>
  </div>
  <pre><code>{E(PLUGIN_EXAMPLE)}</code></pre>
</section>
<section>
  <div class="kicker">Trust</div>
  <h2>Built to be audited</h2>
  <div class="grid">
    <div class="card"><h3>Runs where your data is</h3><p>No network calls and no dependencies in the core. Vertiq LM runs on a Mac; teachers are optional.</p></div>
    <div class="card"><h3>No look-ahead</h3><p>Features come only from events before each cutoff. A plugin is handed the same restricted history.</p></div>
    <div class="card"><h3>Every score explained</h3><p>Each prediction lists what raises and lowers it, in words, and every flag carries its evidence.</p></div>
    <div class="card"><h3>Reproducible</h3><p>Benchmarks, docs and README are generated from committed results; CI fails if they drift.</p></div>
  </div>
</section>
<section>
  <div class="kicker">Sector guides</div>
  <h2>One engine, {len(GUIDES)} worked guides</h2>
  <p>{sectors}</p>
</section>
<section class="hero">
  <h2>Try it on your own export</h2>
  <pre><code>pip install git+https://github.com/ideatorlabs/vertiq</code></pre>
  <p class="cta"><a class="btn" href="guides/index.html">Pick your sector</a><a class="btn ghost" href="architecture.html">Read the architecture</a></p>
</section>"""
    page("index.html", "Vertiq", body, "Decisions you can trust, learned from your own outcomes. Open source, pure Python.")


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
        ("Pure noise", f"models shipped: {S('noise')['shipped_worse_than_baseline'].split('/')[0]} of {CORE['scenarios']['noise']['seeds']}", "the baseline must be kept"),
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
    pains = "\n".join(f"| {t} | {f} |" for t, _p, f in PAINS)
    qs = "\n".join(f"| {q} | `{m}` | {d} |" for q, d, m in QUESTIONS)
    guides = "\n".join(f"- [{g['name']}]({SITE}guides/{g['slug']}.html): {g['blurb']}" for g in GUIDES)
    best = lm_run("v2-sft")
    text = f"""<p align="center"><img src="docs/assets/logo.svg" alt="Vertiq" width="290"></p>

<p align="center"><b>Decisions you can trust, learned from your own outcomes.</b></p>

<p align="center">
<a href="https://github.com/ideatorlabs/vertiq/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/ideatorlabs/vertiq/actions/workflows/tests.yml/badge.svg"></a>
<a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-2346d3"></a>
<img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-2346d3">
<img alt="no dependencies" src="https://img.shields.io/badge/dependencies-0-1d7f55">
</p>

<p align="center"><a href="{SITE}">Docs</a> · <a href="{SITE}benchmarks.html">Benchmarks</a> · <a href="{SITE}guides/">Sector guides</a> · <a href="lm/README.md">Vertiq LM</a></p>

Vertiq is a Python library for business decision models. It predicts what happens next to a customer, a machine, a patient or an
account; it measures which actions actually changed the outcome, using control groups; and it replaces a model only when the new
one beats both a simple baseline and the current model on records it has never seen. The core has no dependencies and makes no
network calls. A companion model, Vertiq LM, maps an unfamiliar export from any sector into Vertiq's data contract, running
locally on a laptop.

## Why Vertiq

| The problem | What Vertiq does |
|---|---|
{pains}

## What it answers

| Question | API | Examples |
|---|---|---|
{qs}

Every task works on the same contract (entities, events, actions), so one engine serves every sector. There is no sector-specific
code; [sector guides]({SITE}guides/) show the mapping and the questions for each.

## How it works

```mermaid
flowchart LR
    A[Your export<br/>CSV, POS, ERP, EHR] --> B[Mapping<br/>Vertiq LM + data repair]
    B --> C[Contract<br/>Entity, Event, Action]
    C --> D[Tasks<br/>event, value, uplift,<br/>affinity, insights, rules]
    P[Your plugins<br/>features, rules, learners] --> D
    D --> G{{Evaluation gate<br/>held-out entities}}
    G -- beats baseline and current model --> R[Scores with reasons,<br/>ranked actions, flags]
    G -- otherwise --> K[Current model kept]
```

1. **Map.** A host maps its records once. Vertiq LM proposes the mapping; `vertiq.mapping.repair` lets the rows settle column names,
   value types and the date format, and it never invents a column.
2. **Learn.** Tasks step back through history to several cutoffs and build features only from what was known at each one.
3. **Gate.** About a fifth of entities, chosen by a hash of their id, never appear in training. A candidate ships only if it beats
   the obvious rule and the model already running on those entities (`vertiq.harness`).
4. **Explain.** Every score lists what raises and lowers it; every rule flag carries its evidence.

## Recursive improvement

```mermaid
flowchart LR
    T[Teacher<br/>writes harder cases] --> V[Verify<br/>data checks every one]
    V --> S[Student learns]
    S --> G{{Gate}}
    G -- ships --> M[Its mistakes]
    M -- punished next round --> S
    O[New outcomes] -- each cycle --> S
```

Each round starts from the model that won the last one. For Vertiq LM, a larger open model (or an OpenAI-compatible endpoint you
configure) writes new tables; a table is kept only if it loads with its own mapping and the teacher, mapping it blind, agrees with
itself. The student's own wrong answers become rejected examples for preference training (ORPO). Closed models whose terms forbid
training on their output are refused as teachers. On the mapping task this loop took a 4B model from
{lm_run('base')['overall']['exact'] * 100:.1f}% to {lm_run('v2-sft-on-v1')['overall']['exact'] * 100:.1f}% exactly right on the same held-out tables.
The decision models follow the same rule: retrained on new outcomes each cycle, promoted only through the gate.

## Install

```bash
pip install git+https://github.com/ideatorlabs/vertiq        # Python 3.10+, no dependencies
pip install "vertiq[lm] @ git+https://github.com/ideatorlabs/vertiq"   # Vertiq LM tooling, Apple Silicon
```

## Quick start

```python
import csv
from vertiq.mapping import Mapping, apply, repair
from vertiq.tasks import predict_event

rows = list(csv.DictReader(open("work_orders.csv")))
m, notes = repair(Mapping.parse({{"entity": "Equipment No", "time": "Work Order Date", "type": "Order Type",
                                 "value": "Cost", "group": "Plant", "attrs": [], "date_format": "%Y-%m-%d"}}), rows)
ds, problems = apply(m, rows)            # rows that don't load are listed, never guessed

task = predict_event(ds, target="breakdown", window_days=30)
print(task.status)                        # trained, kept (the baseline won) or learning (too little data)
print(task.report()["metrics"])           # AUC, Brier, calibration, baseline AUC, held-out entities
print(task.score("EQ-602"))               # {{"chance": ..., "raises": [...], "lowers": [...]}}
```

## Extend with plugins

```python
{PLUGIN_EXAMPLE}
```

| Extension point | Contract |
|---|---|
| `@plugins.feature(pack)` | `fn(entity, events_before, cutoff) -> {{name: number}}`; sees only events before the cutoff |
| `@plugins.rule(pack)`, `vertiq.rules` | `repeat`, `overdue`, `exceeds`, `too_short`, configured with data |
| `plugins.register_learner(name, cls)` | required: `fit(rows, y, **kw) -> self`, `predict(row, group=None)`, `to_dict()`, `from_dict(d)`; optional: `fit_offsets`, `contributions` |
| `verticals.register(Vertical(...))` | windows, nouns and currency for a sector |
| Vertiq LM teachers | `TEACHER_API_KEY`, `TEACHER_BASE_URL`, `TEACHER_MODEL` for any OpenAI-compatible server |

A plugin can only add signal. Whether it helped is still decided by the gate on held-out entities.

## Guarantees

- **No look-ahead.** Features, including plugin features, are built only from events before each cutoff.
- **Held out by entity.** No entity appears in both training and evaluation.
- **No silent regressions.** A candidate replaces the current model only if it is no worse on the same held-out entities.
- **No invented data.** The mapping repair snaps to real columns or drops the role; rows that fail are named.
- **No network, no I/O.** The package never opens a connection or a file. `tests/test_boundary.py` fails on framework, ORM or HTTP imports.
- **Reproducible.** Benchmarks, the docs and this README are generated from committed results; CI fails if they drift.

## Benchmarks

Synthetic scenarios with a planted truth, {CORE['scenarios']['machines']['seeds']} seeds each, mean ± sd
(`python benchmarks/run.py`, {CORE['seconds']:.0f} s on {CORE['machine']}, Python {CORE['python']}).

| Scenario | Vertiq | Reference |
|---|---|---|
{core_rows}

### Vertiq LM: mapping unseen tables

Qwen 3.5 4B (Apache-2.0, 4-bit MLX), LoRA on an Apple M1 Pro with 16 GB. 114 held-out tables over 15 domains; 4 domains
(energy, insurance, manufacturing, retail sales) never appear in training. "Repaired" means after `vertiq.mapping.repair`.

Original test set:

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
{lm_rows('v1')}

Realistic test set (real audit columns, amounts in local styles):

| Model | Loads | Exact | Exact, repaired | Never-seen domains, exact |
|---|---|---|---|---|
{lm_rows('v2')}

## Where Vertiq fits

- **Not a general ML library.** For custom modelling use scikit-learn, LightGBM or XGBoost. Any of them can run inside Vertiq's
  tasks and gate through `register_learner`.
- **Not an experiment tracker.** MLflow and similar tools record and serve models. Vertiq decides whether a retrained model should be
  promoted at all, and the two work together.
- **Not a general extraction library.** Schema-forcing libraries make an LLM fill a schema. Vertiq LM is trained for one job, mapping a
  table to a contract, and every answer is checked against the table's own rows.

## Known limitations

- The benchmarks are synthetic. They show the machinery finds planted signal, ignores noise and refuses worse models; they do not
  show accuracy on your data. Public datasets are next on the roadmap.
- Vertiq LM gets {best['overall']['exact_repaired'] * 100:.0f}% of unseen tables exactly right after repair and loads
  {best['overall']['loads_repaired'] * 100:.0f}%. Treat its mapping as a proposal for a person to confirm.
- Uplift needs a held-out control group on every campaign. Without one, Vertiq falls back to rules and says so.
- The built-in learners are regularised logistic and linear models, chosen for speed and explanations. Register a stronger learner
  when the data supports it.
- Version 0.1. The API may change before 1.0; changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Roadmap

- Benchmarks on public data: Online Retail II (repeat purchase), the Hillstrom email test (uplift), NASA C-MAPSS (engine failure).
- A release on PyPI.
- Vertiq LM: finish preference training on the realistic set, try a 9B student, publish the adapter.
- Generic uplift on any `Action` in the contract, beyond customer messages.
- A gradient-boosting learner plugin with explanations.

## Sector guides

{guides}

## Project layout

| Path | Contents |
|---|---|
| `vertiq/` | contract, mapping, tasks, harness, plugins, rules, churn, uplift, value, affinity, insights, inventory, offers |
| `tests/` | unit tests, the boundary test, and every sector guide's example |
| `benchmarks/` | `run.py`, `collect_lm.py`, `results/*.json` |
| `lm/` | Vertiq LM: data generation, LoRA fine-tuning, scoring, preference training, the verified teacher |
| `docs/` | the portal and `build.py`, which also writes this README |

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through
[SECURITY.md](SECURITY.md). To cite Vertiq, use [CITATION.cff](CITATION.cff).

## Licence

Apache License 2.0, see [LICENSE](LICENSE). The Qwen 3.5 weights Vertiq LM fine-tunes are not in this repository; they are
downloaded separately under their own Apache-2.0 licence.
"""
    (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    for old in ("diagrams.html", "teach-loop.html"):
        (DOCS / old).unlink(missing_ok=True)
    build_index(), build_architecture(), build_learning(), build_lm(), build_benchmarks(), build_guides(), build_readme()
    (DOCS / ".nojekyll").write_text("")
    print("built", len(list(DOCS.rglob("*.html"))), "pages")
