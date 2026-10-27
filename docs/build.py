"""Build the documentation site (docs/*.html, docs/guides/*.html) and README.md from one layout, the
benchmark results and the industry guides. Every page shares one design, and every figure is read from
a results file.

Run:  python docs/build.py      (pure Python; GitHub Pages serves the docs/ folder as built)
"""
from __future__ import annotations

import html
import re
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))
from guides import GUIDES, snippet  # noqa: E402

CORE = json.loads((ROOT / "benchmarks" / "results" / "core.json").read_text())
LM = json.loads((ROOT / "benchmarks" / "results" / "lm.json").read_text())
REAL = json.loads((ROOT / "benchmarks" / "results" / "real.json").read_text())
NARR = json.loads((ROOT / "benchmarks" / "results" / "narrate.json").read_text())
REAL_NAMES = {"rules": "Rule-based proposer (no model)", "base": "Qwen 3.5 4B, no training", "v2-pref": "Vertiq LM (round 3 + preference training)"}


def real_rows():
    """(method, share of roles correct, exact tables) on the held-out real exports, after validation."""
    return [(REAL_NAMES.get(k, k), f"{v['roles_correct'] * 100:.0f}%", v["exact"]) for k, v in REAL["summary"].items()]
PUBLIC = json.loads((ROOT / "benchmarks" / "results" / "public.json").read_text())
SECT = json.loads((ROOT / "benchmarks" / "results" / "sectors.json").read_text())


def sector_rows():
    """(sector, data, size, question, what was promoted, patterns) per sector of benchmarks/sectors."""
    rows = []
    for x in SECT["sectors"]:
        ev = [p for p in x["predictions"] if p["task"] == "event"]
        won = [p for p in ev if p["status"] == "trained"]
        tried = [p for p in ev if p["model"] is not None]
        if won:
            res = "Promoted: " + "; ".join(f"{p['window_days']} days, AUC {p['model']:.3f} vs {p['baseline']:.3f}" for p in won)
            rest = [p for p in ev if p["status"] != "trained" and p["model"] is not None]
            if rest:
                res += ". Kept the rule: " + ", ".join(f"{p['window_days']} days" for p in rest)
        elif tried:
            b = max(tried, key=lambda p: p["model"])
            res = f"Not promoted: best AUC {b['model']:.3f} vs {b['baseline']:.3f}; the current rule stays"
        else:
            res = "Not enough data to train"
        val = [p for p in x["predictions"] if p["task"] == "value" and p["status"] == "trained"]
        if val:
            fmt = (lambda v: f"{v:,.0f}") if val[0]["baseline"] >= 100 else (lambda v: f"{v:.2f}")
            res += f". Next value promoted: average error {fmt(val[0]['model'])} vs {fmt(val[0]['baseline'])}"
        pat = x["patterns"] or {}
        rows.append((x["title"], x["dataset"], f"{x['records']:,} records, {x['entities']:,} entities", x["question"], res,
                     f"{len(pat.get('found') or [])} of {pat.get('tested') or 0}"))
    return rows
REPO = "https://github.com/BiswajitBiswa/vertiq"
SITE = "https://biswajitbiswa.github.io/vertiq/"
FONTS = ("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600"
         "&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap")
NAV = [("index.html", "Overview"), ("start.html", "Get started"), ("architecture.html", "Architecture"), ("learning.html", "Methodology"),
       ("lm.html", "Vertiq LM"), ("benchmarks.html", "Benchmarks"), ("guides/index.html", "Industries"), ("scope.html", "Scope & roadmap")]
E = html.escape
RUNS = CORE["scenarios"]["machines"]["seeds"]
SUITE = f"{CORE['wall_seconds']:.0f} seconds on a laptop using {CORE['jobs']} cores"   # wall-clock time of the parallel run


def S(name):
    return CORE["scenarios"][name]["summary"]


def mean(name, key, nd=3):
    return f"{S(name)[key]['mean']:.{nd}f}"


def total(name, key):
    """A count summed over every run of a scenario (e.g. models promoted across all random datasets)."""
    return sum(r[key] for r in CORE["scenarios"][name]["runs"])


def NOISE(key):
    return f"{total('noise', key)} of {total('noise', 'datasets')}"


def public_rows():
    """(learner, AUC, 90% interval, gain interval over the recency baseline, seconds) for Online Retail II."""
    r = PUBLIC["runs"]
    names = [("linear", "Linear (logistic)"), ("linear_survival", "Linear with the survival feature"),
             ("boosted", "Boosted trees"), ("auto", f"Auto (chose {r['auto']['learner'].replace('Classifier', '').lower()})")]
    out = [(label, f"{r[k]['auc']:.3f}", f"{r[k]['intervals']['score_ci'][0]:.3f}–{r[k]['intervals']['score_ci'][1]:.3f}",
            f"+{r[k]['intervals']['gain_ci'][0]:.3f} to +{r[k]['intervals']['gain_ci'][1]:.3f}", f"{r[k]['seconds']:.0f} s") for k, label in names]
    ext = r.get("external", {})
    if isinstance(ext.get("sklearn_logistic_auc"), float):
        out.append(("scikit-learn logistic regression (reference)", f"{ext['sklearn_logistic_auc']:.3f}", "", "", ""))
    return out


def lm_run(name):
    return next(r for r in LM["runs"] if r["run"] == name)


# ---- shared content ----------------------------------------------------------------------------
CHALLENGES = [
    ("Schema onboarding",
     "Each source system exports data in a different structure. Mapping it to an internal model is manual work, repeated for every new source.",
     "Vertiq LM proposes a column mapping from the header and sample rows. Column names, value types and the date format are then validated against the data before use."),
    ("Attribution bias",
     "Comparing contacted and non-contacted customers credits an intervention with outcomes that would have occurred without it.",
     "Uplift models are trained against randomised hold-out groups and rank entities by the expected change an intervention produces."),
    ("Unvalidated retraining",
     "Scheduled retraining can promote a model that performs worse than the model already in production.",
     "Each candidate is evaluated on held-out entities against a rule-based baseline and the production model, and is promoted only if it performs better."),
    ("False discoveries",
     "Testing many segments surfaces patterns that occur by chance.",
     "Candidate insights are reported only after Benjamini–Hochberg false-discovery-rate control."),
    ("Domain knowledge",
     "Engineers, clinicians and analysts hold knowledge that is not recorded in event data.",
     "Feature and rule plugins encode that knowledge without modifying the decision engine. The evaluation gate measures their contribution."),
    ("Data residency",
     "Regulated data often cannot be sent to external services, and full ML platforms are heavy to deploy at the edge.",
     "The decision engine has no third-party dependencies and makes no network calls. Vertiq LM runs locally on Apple Silicon."),
]

CAPABILITIES = [
    ("Event prediction", "Probability that an event occurs for an entity within a time window, such as customer return, equipment failure, readmission or late payment.", "tasks.predict_event"),
    ("Value prediction", "Expected value of the next event, such as order value, repair cost, length of stay or renewal amount.", "tasks.predict_value"),
    ("Intervention effects", "Effect of any recorded action, such as a maintenance programme, outreach call or carrier change, estimated against a control group, with the entities it helps most.", "tasks.predict_uplift"),
    ("Effects without a control group", "Matched-cohort and doubly robust estimates when an intervention was not randomised, with balance and overlap diagnostics and an interval.", "causal"),
    ("Time to next event", "Probability of the next event within a window given the time elapsed, personalised to each entity's rhythm.", "survival"),
    ("Affinity and sequences", "Co-purchase associations, the next likely product, replenishment intervals and tested A-then-B patterns with their typical lag.", "affinity · sequences"),
    ("Tested insights", "Segment-level patterns reported only after false-discovery-rate control.", "insights"),
    ("Monitoring", "Input and score drift, accuracy on new outcomes, and sequential tests of one rate or two live arms that can be checked after every observation.", "drift · sequential"),
    ("Pattern search", "Finds patterns without being told where to look: every attribute, group, weekday, part of day and item group is compared with the value and with every other feature, under false-discovery control, and written as sentences.", "patterns"),
    ("One-command report", "From exported tables (CSV or Excel, read by content rather than file name, header found below titles) to a written report: Vertiq proposes the mapping, recognises line-item and one-row-per-entity tables, joins tables by shared keys, checks everything, and runs every analysis that applies.", "report · tables"),
    ("Several possible outcomes", "Which of several events comes next (a repair, a replacement or nothing; an upgrade or a cancellation), with a probability for each.", "category.predict_category"),
    ("Planning under a budget", "Who should receive which action, given each action's measured effect, its cost and the value of the outcome, within a budget.", "plan"),
    ("Checks on proposed actions", "Allow, hold or block an action proposed by an agent or a person, from the measured effect of that action and rule flags; unmeasured actions are held for a person.", "guard"),
    ("Why a number changed", "Splits a change between two periods into groups whose own rate moved and groups whose share moved, with false-discovery control.", "diagnose"),
    ("What happened", "The last period against the one before: counts, values, active, new and quiet entities and the groups that moved, as figures and sentences.", "summary · narrate"),
    ("Automatic tuning", "Proposes candidate features from a fixed menu (counts, recency and shares by event type and attribute value), tries them and learner settings one change at a time, keeps those that improve validation folds of training entities consistently, and lets the gate decide on held-out entities.", "autotune · menu"),
    ("Rule-based checks", "Repeat events, overdue tasks, consumption above specification and incomplete records.", "rules"),
    ("Plain-language descriptions", "Predictions, gate decisions and effect estimates written as sentences from the result's own fields; an optional local model can rewrite them, and a rewrite with any number not in the description is rejected.", "narrate"),
]

PLUGIN_EXAMPLE = '''from datetime import timedelta
from vertiq import plugins
from vertiq.tasks import predict_event

@plugins.feature("plants")
def skipped_maintenance(entity, before, cut):
    # `before` contains only events recorded before the training cutoff
    return {"skipped_pm_60": sum(1 for e in before
                                 if e.type == "pm" and e.attrs.get("PM Done") == "no"
                                 and e.at >= cut - timedelta(days=60))}

task = predict_event(ds, target="breakdown", window_days=30, pack="plants")'''

STACK_ROLE = ("Vertiq is a specialised model system, not a general-purpose assistant. Where foundation models generate text, Vertiq "
              "makes and verifies operational decisions: its local model maps source data to a common contract, deterministic "
              "validation checks the mapping, and each new generation of its decision models is promoted only when it measurably "
              "outperforms the generation in production.")

COMPONENTS = [
    ("Vertiq LM", "A 4-billion-parameter language model fine-tuned locally to map any source table to Vertiq's data contract. It runs "
     "on Apple Silicon, so source data does not leave the machine."),
    ("Decision engine", "Event prediction, value prediction, uplift, affinity, tested insights and rule-based monitoring, trained on an "
     "organisation's own outcomes and embedded directly in its applications."),
    ("RSI loop", "Teacher generation, validation, training, evaluation and error feedback. Each promoted generation becomes the starting "
     "point for the next, and every generation passes the evaluation gate."),
]

STACK = [
    ("No dependency footprint", "The decision engine uses only the Python standard library, so it can be vendored into an existing "
     "service, run on a server or embedded in an edge device without adding packages to the environment."),
    ("Local schema mapping", "Vertiq LM runs on Apple Silicon and maps source tables without sending data to an external API. "
     "Its output is validated against the rows before it is used."),
    ("Promotion control", "The evaluation gate compares every candidate with a rule-based baseline and the production model on "
     "held-out entities, with an optional minimum margin (min_gain)."),
    ("Measured outcomes", "Uplift is estimated against randomised control groups, so reported effects reflect the intervention "
     "rather than pre-existing behaviour."),
]

METHODS = ("The methods are established and deliberately transparent: regularised logistic and linear regression with "
           "per-group offsets, gradient-boosted trees, two-model and boosted uplift estimators, matching and augmented inverse-probability "
           "weighting for observational effects, sequential tests, Benjamini–Hochberg false-discovery "
           "control and rule-based checks. Vertiq's contribution is the structure around them: point-in-time features, "
           "entity-level holdout, control-group measurement, the promotion gate and local mapping. Stronger estimators can be "
           "registered through register_learner and are evaluated by the same gate.")


RSI_TEXT = ("Vertiq improves its models through recursive self-improvement (RSI) under evaluation control. In each round a teacher "
            "model generates additional training examples, and every example is validated against its own data before it is used. "
            "The student model is trained and evaluated on held-out records; it is promoted only if it outperforms the production model, "
            "and the promoted model becomes the starting point for the next round. Its incorrect outputs are retained as negative examples "
            "for the following round, and new outcome data is added at every cycle. No iteration reaches production without passing "
            "the evaluation gate.")


def principles():
    """The three design principles, each tied to the code that implements it. Figures come from the results."""
    offer = S("uplift_targeting")["offer_effect_pct"]["mean"]
    return [
        ("Structural intelligence", "Language-model output is constrained by verification against the data.", [
            ("Local schema resolution", "Vertiq LM, a 4-billion-parameter model fine-tuned with LoRA, proposes a mapping from the column "
             "headers and sample rows. It runs on the user's machine; source data is not sent to an external service."),
            ("Deterministic validation", "vertiq.mapping.repair checks the proposal against the rows: names are matched to existing "
             "columns, roles are checked against value types and the date format is inferred. It never introduces a column that does not exist."),
            ("Bounded data contract", "Every record is expressed as an entity, a timestamp, an event type, a value and attributes. Rows "
             "that fail validation are reported to the caller, so models train only on records that loaded.")]),
        ("Causal measurement", "Intervention effects are estimated against control groups rather than inferred from correlation.", [
            ("Control-group uplift", "vertiq.uplift compares treated entities with a randomly held-out group over the same window and "
             f"ranks entities by estimated incremental effect. In the benchmark it estimated a planted −20% offer effect at {offer:+.1f}%."),
            ("Configurable horizons", "window_days sets the prediction horizon, cutoffs the number of historical training points and "
             "history the event types used as inputs, so each task matches the operating cycle of the industry."),
            ("Small footprint", "The decision engine has no third-party dependencies and no network layer. It runs wherever Python 3.10 runs; "
             f"the full benchmark suite completes in {SUITE}.")]),
        ("Defensive deployment", "A candidate model is treated as unproven until it has been evaluated.", [
            ("Held-out evaluation", "Candidates are scored on entities excluded from training, using features built only from events "
             "recorded before each cutoff."),
            ("Champion–challenger gate", "A candidate must outperform the rule-based baseline and the production model, and its gain over "
             "the baseline must be clear at 90% confidence by default. min_gain sets a required margin: an AUC margin for event tasks "
             "and a relative error reduction for value tasks."),
            ("Segment check", "A candidate that wins on average but is clearly worse than the baseline for a segment of the "
             "held-out entities is not promoted. Segments come from the data (entity groups, entity-table labels, length of history), "
             "each with at least 30 held-out entities, and the confidence level is corrected for the number checked."),
            ("Explicit rejection", "A rejected candidate returns status \"kept\" with the reason, such as the AUC it reached against the "
             "required threshold. The production model remains in service.")]),
    ]


MATRIX = [
    ("Structural intelligence", "Local mapping proposal, validated against the rows",
     "Fine-tune Vertiq LM on your naming conventions (lm/); vertiq.mapping.repair",
     "Source data stays on premises; new sources are onboarded without hand-written mappings"),
    ("Causal measurement", "Control-group uplift and point-in-time tasks",
     "window_days, cutoffs and history in vertiq.tasks; a hold-out share on every campaign",
     "Intervention effects are measured rather than assumed; horizons match each industry's cycle"),
    ("Defensive deployment", "Entity-level holdout with champion–challenger comparison",
     "min_gain in vertiq.tasks and vertiq.harness",
     "Weaker or unstable candidates are not promoted; the production model stays in service"),
]


def principles_html() -> str:
    cols = ""
    for title, lead, items in principles():
        lis = "".join(f"<li><b>{E(h)}.</b> {E(t)}</li>" for h, t in items)
        cols += f'<div class="card principle"><h3>{E(title)}</h3><p class="lead">{E(lead)}</p><ul>{lis}</ul></div>'
    rows = "".join(f"<tr><td><b>{E(a)}</b></td><td>{E(b)}</td><td>{E(c)}</td><td>{E(d)}</td></tr>" for a, b, c, d in MATRIX)
    return (f'<div class="grid principles">{cols}</div>'
            '<div class="tbl"><table><thead><tr><th>Principle</th><th>Mechanism</th><th>Configuration</th><th>Operational value</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def principles_md() -> str:
    out = []
    for title, lead, items in principles():
        out.append(f"### {title}\n\n{lead}\n")
        out += [f"- **{h}.** {t}" for h, t in items]
        out.append("")
    out.append("| Principle | Mechanism | Configuration | Operational value |\n|---|---|---|---|")
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in MATRIX]
    return "\n".join(out)


def scope_items(best: dict) -> tuple[list[str], list[str]]:
    """Limitations and roadmap, shared by the README and scope.html (Markdown inline syntax)."""
    lims = [
        f"Most benchmarks use synthetic data. They verify that the methods detect planted signal, reject noise and retain the baseline when appropriate; one public dataset (Online Retail II) has been run. None measures accuracy on a specific organisation's data.",
        f'The gate requires the gain over the baseline to be clear at 90% confidence (`confidence=0.9`, the default). On small datasets this also holds back some real improvements until more data arrives: with 240 assets and a real signal, 7 of 15 candidates were promoted, against 13 of 15 with the check off. `confidence=None` turns the check off; the candidate then only has to not be worse.',
        f"Plain-language descriptions (`narrate`) are written from templates by default. The trained rewrite model gives real rewrites for {NARR['runs'][1]['accepted_rewrites_not_copies']} of {NARR['runs'][1]['n']} held-out descriptions, and every rewrite is checked: each number must be the template's and stay next to what it described. The checks catch invented and moved numbers, not every misleading sentence.",
        f"After validation, Vertiq LM maps {best['overall']['exact_repaired'] * 100:.0f}% of held-out tables exactly ({best['unseen']['exact_repaired'] * 100:.0f}% from domains it was not trained on) and loads {best['overall']['loads_repaired'] * 100:.0f}% without errors. Proposed mappings should be reviewed before production use.",
        f'Uplift estimation requires a randomised control group. Without one, `causal.matched_att` gives an estimate whose validity depends on every confounder being observed; `design.next_step` reports the experiment size needed to settle the question.',
        f'On the synthetic benchmarks, boosted trees improve on linear models where effects are non-linear and do not where effects are additive. Across 100 to 6,400 entities the default tree settings were within 0.01 AUC of the best of five alternatives. `learner="auto"` chooses by 3-fold cross-validation on the training entities; with a few hundred entities or fewer its choice is close to random when the two learners are within a point or two of each other.',
        f"Doubly robust estimates are consistent if either the propensity or the outcome model is right, but they vary more than the outcome model alone (standard deviation {S('doubly_robust_effect')['doubly_robust']['sd']:.2f} across benchmark runs); large weights from extreme propensities are clipped and counted in the result.",
        f'The rule-based mapping proposer gets 57% of roles right on the held-out real exports and Vertiq LM 83%; both are checked by validation, which flags what it cannot settle (for example two complete amount columns, or day/month order), but meaning (which amount is the one that matters) still needs a person or the model.',
        f'Tables with one row per entity (customer lists, asset registers) often hold totals and dates computed when the file was exported; Vertiq uses them to describe, never as prediction features.',
        f'`plan` and `guard` rely on uplift tasks, so they are only as good as the recorded actions and control groups; an action that was never measured is excluded from plans and held by the check. `guard` does not decide on its own: it allows only measured, clearly helpful actions.',
        f'`menu` proposes features from a fixed set of operations; anything outside it (external data, formulas specific to a sector) is added as a plugin by a person and judged by the same gate.',
        f'`autotune` scores every trial on the same training entities, so many trials can still favour a lucky change; it keeps a change only when most validation folds agree, and the held-out gate makes the final decision.',
        f'Version 0.2. The API may change before 1.0; changes are recorded in [CHANGELOG.md](CHANGELOG.md).',
    ]
    road = [
        f'Further public benchmarks: M5 (value prediction), NASA C-MAPSS (equipment failure), KKBOX (churn), and Hillstrom and Criteo (randomised uplift); protocols in [proposal 0001](proposals/0001-statistical-validation.md), part C.',
        f'Vertiq LM: a training round on tables with real-export clutter (`lm/make_data.py --messy`), scored on the synthetic and the real held-out sets; evaluate a 9B student and publish the adapter with a `vertiq map` command.',
        f'`autotune` for value tasks and uplift; menu operations for group-level signals and external series (weather, prices) joined by date and location; menu items suggested by a language model for a sector, checked by `menu.parse`.',
        f'Turning free-text fields (support tickets, maintenance notes) into validated events.',
    ]
    return lims, road


def md_inline(text: str) -> str:
    """Markdown inline code and links to HTML; links to repository files go to GitHub."""
    out = E(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)

    def link(m):
        url = m.group(2) if m.group(2).startswith("http") else f"{REPO}/blob/main/{m.group(2)}"
        return f'<a href="{url}">{m.group(1)}</a>'
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, out)


# ---- layout ------------------------------------------------------------------------------------
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
<footer><div><span>Vertiq {E(CORE.get('version', '0.2.0'))} · Apache License 2.0</span><a href="{REPO}">Source on GitHub</a><a href="{up}benchmarks.html">Benchmarks</a><a href="{up}guides/index.html">Industry guides</a></div></footer>
</body></html>
"""
    out = DOCS / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)


# ---- icons (48 x 48, stroked with the current colour) ------------------------------------------
def _gear() -> str:
    teeth = []
    for k in range(8):
        a = math.pi * k / 4
        x1, y1 = 24 + 12.5 * math.cos(a), 24 + 12.5 * math.sin(a)
        x2, y2 = 24 + 17 * math.cos(a), 24 + 17 * math.sin(a)
        teeth.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    return '<circle cx="24" cy="24" r="12.5"/><circle cx="24" cy="24" r="5"/>' + "".join(teeth)


ICONS = {
    "bag": '<path d="M11 16h26l-2 24H13z"/><path d="M18 16v-3a6 6 0 0 1 12 0v3"/>',
    "hotel": '<rect x="11" y="8" width="26" height="34"/><line x1="6" y1="42" x2="42" y2="42"/><rect x="21" y="33" width="6" height="9"/>'
             '<path d="M16 14h4M28 14h4M16 21h4M28 21h4M16 28h4M28 28h4"/>',
    "bank": '<path d="M6 18L24 8l18 10z"/><line x1="8" y1="40" x2="40" y2="40"/><line x1="6" y1="44" x2="42" y2="44"/>'
            '<path d="M12 21v16M20 21v16M28 21v16M36 21v16"/>',
    "shield": '<path d="M24 6l15 6v10c0 10-7 17-15 20C16 39 9 32 9 22V12z"/><polyline points="17,24 22,29 31,19"/>',
    "cross": '<path d="M19 8h10v11h11v10H29v11H19V29H8V19h11z"/>',
    "gear": _gear(),
    "truck": '<rect x="5" y="14" width="24" height="17"/><path d="M29 19h8l5 6v6H29z"/><circle cx="13" cy="34" r="3.5"/><circle cx="35" cy="34" r="3.5"/>',
    "tower": '<line x1="24" y1="18" x2="24" y2="42"/><line x1="17" y1="42" x2="24" y2="22"/><line x1="31" y1="42" x2="24" y2="22"/>'
             '<circle cx="24" cy="15" r="2.5"/><path d="M17 9a10 10 0 0 0 0 12M31 9a10 10 0 0 1 0 12M12 5a16 16 0 0 0 0 20M36 5a16 16 0 0 1 0 20"/>',
    "bolt": '<path d="M27 5L11 27h11l-3 16 18-24H26z"/>',
    "cloud": '<path d="M15 36a8 8 0 0 1-1-15.9 11 11 0 0 1 21-1.6A8.5 8.5 0 0 1 35 36z"/>',
    "boxes": '<rect x="8" y="24" width="15" height="14"/><rect x="25" y="24" width="15" height="14"/><rect x="16" y="10" width="15" height="14"/>',
    "doc": '<path d="M12 6h18l8 8v28H12z"/><path d="M30 6v8h8"/><line x1="18" y1="34" x2="30" y2="22"/><circle cx="19" cy="23" r="2"/><circle cx="29" cy="33" r="2"/>',
    "cap": '<path d="M4 18l20-9 20 9-20 9z"/><path d="M12 22v9c3 3 8 5 12 5s9-2 12-5v-9"/><line x1="44" y1="18" x2="44" y2="30"/>',
}


def icon(key: str) -> str:
    return (f'<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[key]}</svg>')


def tile(g: dict, i: int, href: str) -> str:
    return f'<a class="tile t{i % 4 + 1}" href="{href}">{icon(g["icon"])}<span>{E(g["name"])}</span></a>'


# ---- diagrams ----------------------------------------------------------------------------------
def arrow_defs(i: str) -> str:
    return (f'<defs><marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>'
            f'<marker id="b{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="ah acc"/></marker></defs>')


def box(x, y, w, h, title, lines, cls="box", kicker=None):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>']
    ty = y + 26
    if kicker:
        out.append(f'<text x="{x + 14}" y="{y + 22}" class="k">{E(kicker)}</text>')
        ty = y + 44
    out.append(f'<text x="{x + 14}" y="{ty}" class="t">{E(title)}</text>')
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 14}" y="{ty + 22 + 18 * i}" class="s">{E(ln)}</text>')
    return "".join(out)


def fig_hero() -> str:
    """An example of the library's input and output, labelled as an example."""
    rows = [("EQ-602", "2026-09-04", "pm", "no"), ("EQ-602", "2026-09-19", "breakdown", "—"), ("EQ-118", "2026-09-20", "pm", "yes"),
            ("EQ-431", "2026-09-22", "pm", "no")]
    g = ['<rect x="0" y="0" width="560" height="430" rx="8" class="fills" opacity=".55"/>',
         '<rect x="22" y="26" width="330" height="200" rx="6" class="box"/>',
         '<text x="40" y="54" class="k">SOURCE EXPORT · WORK_ORDERS.CSV</text>',
         '<line x1="22" y1="68" x2="352" y2="68" class="grid-l"/>']
    hdr = ("Equipment No", "Date", "Order Type", "PM Done")
    xs = (40, 130, 220, 300)
    for x, h in zip(xs, hdr):
        g.append(f'<text x="{x}" y="90" class="s">{E(h)}</text>')
    for i, r in enumerate(rows):
        y = 118 + 26 * i
        for x, v in zip(xs, r):
            g.append(f'<text x="{x}" y="{y}" class="m">{E(v)}</text>')
    g += ['<rect x="196" y="214" width="342" height="196" rx="6" class="gate"/>',
          '<text x="216" y="242" class="k">EXAMPLE OUTPUT · predict_event</text>',
          '<text x="216" y="272" class="t">EQ-602 · failure within 30 days</text>',
          '<text x="216" y="318" style="font-family:var(--display);font-size:40px;fill:var(--head)">0.31</text>',
          '<rect x="306" y="300" width="210" height="10" rx="5" class="fills"/>',
          '<rect x="306" y="300" width="65" height="10" rx="5" class="fillo"/>',
          '<text x="216" y="350" class="s">Raises: skipped maintenance in the last 60 days</text>',
          '<text x="216" y="372" class="s">Lowers: time since the last failure</text>',
          '<text x="216" y="394" class="s">Model status: promoted (beat baseline on held-out assets)</text>']
    return ('<div class="hero-visual"><svg viewBox="0 0 560 430" role="img" aria-label="Example: a work-order export is mapped and scored; '
            f'the output shows a failure probability for one asset with the factors that raised and lowered it.">{"".join(g)}</svg></div>')


def fig_system() -> str:
    xs = [16, 196, 376, 556, 736, 916]
    w, y, h = 158, 96, 176
    boxes = [("Source data", ["CSV, POS, ERP,", "EHR and CRM", "exports"], "box"),
             ("Mapping", ["proposed by", "Vertiq LM, validated", "against the rows"], "box"),
             ("Data contract", ["entities and groups", "events: time, type,", "value, attributes", "actions and controls"], "box"),
             ("Analysis tasks", ["event prediction", "value prediction", "uplift, affinity", "insights, rules"], "box"),
             ("Evaluation gate", ["held-out entities;", "must outperform", "baseline and", "production model"], "gate"),
             ("Outputs", ["scores with factors", "ranked interventions", "tested insights", "rule flags"], "box")]
    g = [arrow_defs("s")]
    g.append('<rect x="182" y="50" width="716" height="372" rx="10" fill="none" class="ln dash"/>')
    g.append('<text x="196" y="74" class="k">VERTIQ LIBRARY BOUNDARY · NO DATABASE, NETWORK OR WEB FRAMEWORK ACCESS</text>')
    for (t, ls, c), x in zip(boxes, xs):
        g.append(box(x, y, w, h, t, ls, c))
    for i in range(5):
        x1, x2 = xs[i] + w, xs[i + 1] - 2
        acc = i == 4
        g.append(f'<line x1="{x1}" y1="{y + h / 2}" x2="{x2}" y2="{y + h / 2}" class="ln{" acc" if acc else ""}" marker-end="url(#{"b" if acc else "a"}s)"/>')
    g.append(f'<text x="{xs[5] + w / 2}" y="{y - 10}" class="lbl" text-anchor="middle">promoted when better</text>')
    g.append(box(556, 320, 158, 86, "Domain plugins", ["features, rules,", "custom learners"], "box dash"))
    g.append(f'<line x1="635" y1="320" x2="635" y2="{y + h + 2}" class="ln" marker-end="url(#as)"/>')
    g.append('<text x="645" y="300" class="lbl">extends</text>')
    g.append(box(736, 320, 158, 86, "Model retained", ["production model", "stays in service"], "box dash"))
    g.append(f'<line x1="815" y1="{y + h}" x2="815" y2="318" class="ln dash" marker-end="url(#as)"/>')
    g.append('<text x="825" y="300" class="lbl">otherwise</text>')
    return (f'<figure><div class="scroll"><svg viewBox="0 0 1090 440" role="img" aria-label="Source data is mapped to a common contract; '
            f'analysis tasks train candidate models; the evaluation gate promotes a candidate only if it outperforms the baseline and the production model.">'
            f'{"".join(g)}</svg></div><figcaption>Records are mapped once to a common data contract. Analysis tasks train candidate models, '
            'plugins add domain features, and the evaluation gate promotes a candidate only if it outperforms both the rule-based baseline '
            'and the production model on held-out entities.</figcaption></figure>')


def fig_cutoffs() -> str:
    g = [arrow_defs("c")]
    g.append('<line x1="40" y1="150" x2="1040" y2="150" class="ln"/>')
    g.append('<text x="40" y="186" class="lbl">earliest record</text><text x="1040" y="186" class="lbl" text-anchor="end">latest record</text>')
    for x in (90, 150, 230, 300, 360, 470, 520, 610, 690, 760, 905, 980):
        g.append(f'<circle cx="{x}" cy="150" r="6" class="hot"/>')
    for cx, lab in ((440, "CUTOFF 3"), (620, "CUTOFF 2"), (800, "CUTOFF 1")):
        g.append(f'<line x1="{cx}" y1="70" x2="{cx}" y2="220" class="ln dash"/><text x="{cx}" y="62" class="k" text-anchor="middle">{lab}</text>')
    g.append('<rect x="40" y="96" width="760" height="22" rx="4" class="hot"/>')
    g.append('<text x="52" y="112" class="s">features at cutoff 1: events recorded before the cutoff only</text>')
    g.append('<rect x="800" y="232" width="200" height="22" rx="4" class="box"/>')
    g.append('<text x="810" y="248" class="s">label: event within the window</text>')
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 270" role="img" aria-label="Training examples are built at successive cutoffs; features use only events before each cutoff.">'
            f'{"".join(g)}</svg></div><figcaption>Training examples are built at successive historical cutoffs. Features use only events '
            'recorded before each cutoff, and each label records whether the target event occurred in the following window.</figcaption></figure>')


def fig_uplift() -> str:
    g = [arrow_defs("u")]
    g.append(box(16, 70, 200, 96, "Campaign", ["target list"], "box"))
    g.append(box(300, 20, 230, 88, "Treatment group", ["receives the intervention"], "box"))
    g.append(box(300, 128, 230, 88, "Control group", ["randomly held out"], "hot"))
    g.append(box(620, 70, 230, 96, "Outcome", ["measured over the same window"], "box"))
    g.append(box(930, 70, 140, 96, "Uplift", ["treatment minus", "control"], "gate"))
    g.append('<path d="M216 118 H258 V64 H298" class="ln" marker-end="url(#au)"/><path d="M258 118 V172 H298" class="ln" marker-end="url(#au)"/>')
    g.append('<path d="M530 64 H575 V110 H618" class="ln" marker-end="url(#au)"/><path d="M530 172 H575 V126 H618" class="ln" marker-end="url(#au)"/>')
    g.append('<line x1="850" y1="118" x2="928" y2="118" class="ln acc" marker-end="url(#bu)"/>')
    top, bottom = S("uplift_targeting")["lift_top_half"]["mean"] * 100, S("uplift_targeting")["lift_bottom_half"]["mean"] * 100
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 230" role="img" aria-label="A randomised control group is withheld from each intervention to estimate uplift.">'
            f'{"".join(g)}</svg></div><figcaption>A randomly selected control group is withheld from each intervention. The uplift model '
            'estimates for which entities the outcome difference between treatment and control is largest. In the benchmark, entities '
            f'ranked in the top half showed a lift of {top:+.1f} percentage points; the bottom half showed {bottom:+.1f}.</figcaption></figure>')


def fig_rsi() -> str:
    g = [arrow_defs("r")]
    xs, w, h = [16, 286, 556, 826], 230, 116
    top = [("TEACHER", "Generates examples", ["open-weight model or a", "configured API endpoint"], "box"),
           ("VALIDATION", "Verifies examples", ["each example must load and", "match a blind re-mapping"], "box"),
           ("STUDENT", "Trains", ["Vertiq LM or a", "decision model"], "box"),
           ("EVALUATION GATE", "Promotes or retains", ["compared with the baseline", "and the production model", "on held-out records"], "gate")]
    for (k, t, ls, c), x in zip(top, xs):
        g.append(box(x, 30, w, h, t, ls, c, kicker=k))
    for i in range(3):
        g.append(f'<line x1="{xs[i] + w}" y1="88" x2="{xs[i + 1] - 2}" y2="88" class="ln" marker-end="url(#ar)"/>')
    g.append(box(826, 226, w, h, "Feed the next round", ["incorrect outputs are", "penalised in preference", "training (ORPO)"], "box", kicker="NEGATIVE EXAMPLES"))
    g.append(box(286, 226, w, h, "Added each cycle", ["transactions, failures,", "renewals, claims"], "box dash", kicker="OUTCOME DATA"))
    g.append('<line x1="941" y1="146" x2="941" y2="224" class="ln acc" marker-end="url(#br)"/>')
    g.append('<text x="951" y="190" class="lbl">after promotion</text>')
    g.append('<path d="M826 284 H686 V148" class="ln" marker-end="url(#ar)"/>')
    g.append('<text x="756" y="276" class="lbl" text-anchor="middle">next round</text>')
    g.append('<path d="M516 284 H626 V148" class="ln dash" marker-end="url(#ar)"/>')
    g.append('<text x="571" y="276" class="lbl" text-anchor="middle">each cycle</text>')
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 360" role="img" aria-label="Recursive self-improvement: a teacher generates examples, '
            'validation verifies them, the student trains, the evaluation gate promotes or retains, and incorrect outputs feed the next round.">'
            f'{"".join(g)}</svg></div><figcaption>Each round begins from the model promoted in the previous round. Teacher output is never used '
            'without validation, and providers whose terms prohibit training on generated output are excluded.</figcaption></figure>')


def fig_lm() -> str:
    g = [arrow_defs("l")]
    steps = [("1 · GENERATE", "Synthetic tables", ["15 domains, 6 naming", "conventions, 5 languages,", "all date formats"]),
             ("2 · VALIDATE", "Labels verified", ["each label must load", "its own rows; failures", "are discarded"]),
             ("3 · FINE-TUNE", "LoRA training", ["Qwen 3.5 4B, 4-bit", "Apple M1 Pro, 16 GB"]),
             ("4 · EVALUATE", "Held-out tables", ["114 tables; 4 domains", "excluded from training"]),
             ("5 · PREFERENCE", "Error correction", ["incorrect outputs used", "as negatives (ORPO)"])]
    xs = [16, 230, 444, 658, 872]
    for (k, t, ls), x in zip(steps, xs):
        g.append(box(x, 40, 192, 140, t, ls, "gate" if k.startswith("4") else "box", kicker=k))
    for i in range(4):
        g.append(f'<line x1="{xs[i] + 192}" y1="110" x2="{xs[i + 1] - 2}" y2="110" class="ln" marker-end="url(#al)"/>')
    g.append('<path d="M968 180 V226 H540 V182" class="ln acc" marker-end="url(#bl)"/>')
    g.append('<text x="754" y="246" class="lbl" text-anchor="middle">retrained; promoted only if accuracy improves</text>')
    g.append(box(16, 270, 300, 110, "Teacher model", ["Qwen 3.5 9B or an open API model;", "tables retained only if they load and", "match a blind re-mapping"], "box dash", kicker="6 · TEACHER"))
    g.append('<path d="M166 270 V182" class="ln dash" marker-end="url(#al)"/>')
    g.append(box(444, 270, 620, 110, "Validation at inference", ["column names matched to existing columns; roles checked against value types",
                                                                 "(timestamps contain dates, values are numeric); date format inferred from the rows"], "hot", kicker="7 · VALIDATE"))
    return ('<figure><div class="scroll"><svg viewBox="0 0 1080 400" role="img" aria-label="Vertiq LM training pipeline: generate, validate, fine-tune, evaluate, preference training, teacher model and validation at inference.">'
            f'{"".join(g)}</svg></div><figcaption>Teacher output is used only after validation: a generated table must load with its own '
            'mapping, and the teacher must reproduce the mapping without seeing it. OpenAI, Anthropic and Google models are excluded as '
            'teachers.</figcaption></figure>')


def fig_mapping(g: dict) -> str:
    m = g["mapping"]
    pairs = [(m["entity"], "Entity · identifier"), (m["time"], "Event · timestamp")]
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
    sid = "m" + g["slug"][:4]
    out = [arrow_defs(sid)]
    out.append('<text x="16" y="18" class="k">SOURCE COLUMNS</text><text x="620" y="18" class="k">VERTIQ DATA CONTRACT</text>')
    for i, (col, role) in enumerate(pairs):
        y = 30 + 44 * i
        out.append(f'<rect x="16" y="{y}" width="300" height="34" rx="5" class="box"/><text x="30" y="{y + 22}" class="m">{E(col)}</text>')
        out.append(f'<path d="M316 {y + 17} C 470 {y + 17}, 470 {ry[role]}, 618 {ry[role]}" class="ln" marker-end="url(#a{sid})"/>')
    for r in roles:
        out.append(f'<rect x="620" y="{ry[r] - 17}" width="240" height="34" rx="5" class="hot"/><text x="634" y="{ry[r] + 5}">{E(r)}</text>')
    out.append(f'<text x="880" y="{ry[roles[1]] + 5}" class="s">{E(m["date_format"])}</text>')
    return (f'<figure><div class="scroll"><svg viewBox="0 0 1080 {h + 10}" role="img" aria-label="Column mapping for {E(g["name"])}.">'
            f'{"".join(out)}</svg></div><figcaption>Vertiq LM can propose this mapping from the header and sample rows. Column names, value '
            'types and the date format are validated against the data before use.</figcaption></figure>')


def bars(rows, vmax: float, unit: str, label: str) -> str:
    per = max(len(s) for _n, s in rows)
    rh = 20 * per + 18
    h = 24 + rh * len(rows) + 30
    x0, x1 = 320, 1000
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
            g.append(f'<rect x="{x0}" y="{yy}" width="{max(1.5, X(v) - x0):.1f}" height="15" rx="2" class="{cls}"><title>{E(sname)}: {v:.1f}{unit}</title></rect>'
                     f'<text x="{X(v) + 6:.1f}" y="{yy + 12}" class="s">{v:.1f}{unit}</text>')
    return f'<svg viewBox="0 0 1080 {h}" role="img" aria-label="{E(label)}">{"".join(g)}</svg>'


# ---- pages -------------------------------------------------------------------------------------
def build_index():
    best = lm_run("v2-pref")
    challenges = "".join(f'<div class="card"><h3>{E(t)}</h3><p>{E(p)}</p><p class="fix">{E(f)}</p></div>' for t, p, f in CHALLENGES)
    caps = "".join(f'<div class="card"><h3>{E(t)}</h3><p>{E(d)}</p><p class="api">{E(a)}</p></div>' for t, d, a in CAPABILITIES)
    tiles = "".join(tile(g, i, f"guides/{g['slug']}.html") for i, g in enumerate(GUIDES))
    body = f"""
<section class="hero-split">
  <div>
    <div class="kicker">Self-improving decision intelligence · open source</div>
    <h1>A self-improving model system for operational decisions</h1>
    <p class="lede">Vertiq combines a local language model that reads any operational dataset with decision models trained on real
    outcomes. Both improve through recursive self-improvement: each generation trains its successor, and a generation reaches production
    only after it outperforms the current one on held-out data.</p>
    <p class="cta"><a class="btn" href="start.html">Get started</a><a class="btn ghost" href="{REPO}">View on GitHub</a><a class="btn ghost" href="#rsi">How RSI works</a></p>
  </div>
  {fig_hero()}
</section>
<section aria-label="Benchmark results">
  <div class="stats">
    <div class="stat"><b>{mean('machines', 'plugin_auc')}</b><span>AUC for equipment failure prediction with one domain plugin; rule baseline {mean('machines', 'baseline_auc')}</span></div>
    <div class="stat"><b>{S('uplift_targeting')['offer_effect_pct']['mean']:+.1f}%</b><span>estimated offer effect on transaction value; true effect −20%</span></div>
    <div class="stat"><b>{PUBLIC['runs']['linear']['auc']:.3f}</b><span>AUC predicting repeat orders for {PUBLIC['counts']['customers']:,} real customers (Online Retail II); recency baseline {PUBLIC['runs']['linear']['baseline_auc']:.3f}</span></div>
    <div class="stat"><b>{best['overall']['loads_repaired'] * 100:.1f}%</b><span>of held-out tables mapped by Vertiq LM without load errors after validation</span></div>
  </div>
  <p class="note-s">Synthetic benchmarks with known ground truth, {RUNS} runs each, and one public dataset. See <a href="benchmarks.html">benchmark methodology</a>.</p>
</section>
<section id="rsi">
  <div class="section-head"><div class="kicker">Recursive self-improvement</div><h2>Each generation trains the next. Only better generations ship.</h2>
  <p>{E(RSI_TEXT)}</p>
  <p>Applied to table mapping, this process raised the exact-match accuracy of a 4-billion-parameter model from
  {lm_run('base')['overall']['exact'] * 100:.1f}% to {lm_run('v2-pref-on-v1')['overall']['exact'] * 100:.1f}% on the same held-out set.</p></div>
  <div class="grid">{"".join(f'<div class="card"><h3>{E(t)}</h3><p>{E(d)}</p></div>' for t, d in COMPONENTS)}</div>
  {fig_rsi()}
</section>
<section>
  <div class="section-head"><div class="kicker">Role in an AI stack</div><h2>A specialised model system for decisions</h2>
  <p>{E(STACK_ROLE)}</p></div>
  <div class="grid">{"".join(f'<div class="card"><h3>{E(t)}</h3><p>{E(d)}</p></div>' for t, d in STACK)}</div>
</section>
<section>
  <div class="section-head"><div class="kicker">Challenges</div><h2>Common obstacles to reliable decision models</h2></div>
  <div class="grid">{challenges}</div>
</section>
<section>
  <div class="section-head"><div class="kicker">Capabilities</div><h2>Supported analyses</h2>
  <p>All analyses operate on one data contract of entities, events and actions, so the same system serves every industry.</p></div>
  <div class="grid">{caps}</div>
</section>
<section>
  <div class="section-head"><div class="kicker">Design principles</div><h2>Structural intelligence, causal measurement, defensive deployment</h2>
  <p>Each principle is implemented in a specific module and can be inspected and configured.</p></div>
  {principles_html()}
</section>
<section class="two">
  <div class="section-head">
    <div class="kicker">Extensibility</div>
    <h2>Domain logic through plugins</h2>
    <p>Feature functions, validation rules, custom learners and industry profiles are registered under a named pack. Feature functions
    receive only events recorded before each training cutoff, which prevents look-ahead bias, and the evaluation gate measures whether a
    plugin improves performance. In the equipment benchmark, one maintenance feature raised AUC from {mean('machines', 'generic_auc')} to
    {mean('machines', 'plugin_auc')}.</p>
    <ul><li><code>@plugins.feature(pack)</code>: domain features</li><li><code>@plugins.rule(pack)</code>, <code>vertiq.rules</code>: validation rules</li>
    <li><code>plugins.register_learner(name, cls)</code>: external estimators such as gradient boosting</li>
    <li><code>verticals.register(...)</code>: industry windows and terminology</li><li>Teacher endpoints: any OpenAI-compatible server</li></ul>
  </div>
  <pre><code>{E(PLUGIN_EXAMPLE)}</code></pre>
</section>
<section>
  <div class="section-head"><div class="kicker">Security and governance</div><h2>Controls built into the system</h2></div>
  <div class="grid">
    <div class="card"><h3>Local execution</h3><p>The decision engine performs no network or file I/O. Data is passed in by the host application, which retains control of storage and access.</p></div>
    <div class="card"><h3>Point-in-time correctness</h3><p>Features, including plugin features, are computed only from events recorded before each cutoff.</p></div>
    <div class="card"><h3>Explainability</h3><p>Each prediction lists the factors that raised and lowered it. Each rule flag includes its supporting evidence.</p></div>
    <div class="card"><h3>Reproducibility</h3><p>Benchmarks, documentation and the README are generated from committed results. Continuous integration fails if they diverge.</p></div>
  </div>
</section>
<section>
  <div class="section-head"><div class="kicker">Industries</div><h2>Implementation guides by industry</h2>
  <p>Each guide documents the data mapping, supported analyses, recommended plugins, validation rules and test coverage for one industry.</p></div>
  <div class="tiles">{tiles}</div>
</section>
<section class="section-head">
  <div class="kicker">Get started</div><h2>Install Vertiq</h2>
  <p>Python 3.10 or later; no third-party dependencies.</p>
  <pre><code>pip install git+https://github.com/BiswajitBiswa/vertiq</code></pre>
  <p class="cta"><a class="btn" href="guides/index.html">Industry guides</a><a class="btn ghost" href="{REPO}#quick-start">Quick start</a></p>
</section>"""
    page("index.html", "Vertiq", body, "Self-improving decision intelligence: a local language model and outcome-trained decision models, improved through recursive self-improvement under an evaluation gate.")


def build_architecture():
    mods = [("mapping", "Maps a source table to the data contract; repair() validates column names, value types and the date format"),
            ("contract", "Entity, Event, Action and Dataset: the common structure that every task reads"),
            ("tasks", "predict_event and predict_value for any event type, with contributing factors for each score"),
            ("harness", "Evaluation gate: entity-level holdout, baseline comparison and production-model comparison"),
            ("plugins", "Registration of domain features, rules and custom learners"),
            ("rules", "Declarative checks: repeat, overdue, exceeds and too_short"),
            ("churn · uplift · value", "Customer retention, intervention uplift, transaction value and campaign revenue"),
            ("affinity · offers", "Product affinity, next likely product, replenishment intervals and threshold offers"),
            ("insights", "Segment patterns filtered with false-discovery-rate control"),
            ("inventory", "Reorder points, slow-moving stock, new items and margins"),
            ("learn", "Regularised logistic and linear models, AUC, Brier score, calibration and Benjamini–Hochberg"),
            ("verticals", "Industry profiles: time windows, terminology and currency"),
            ("boost", "Gradient-boosted trees for non-linear effects, with path-based contributing factors"),
            ("survival", "Kaplan–Meier time-to-event estimates with censoring and per-entity rhythm"),
            ("causal", "Matched-cohort ATT and cross-fitted doubly robust (AIPW) ATE, with balance, overlap and intervals"),
            ("design", "Experiment sizing, minimum detectable effects and next-step recommendations"),
            ("sequential", "Wald's SPRT for one rate and a mixture SPRT for two live arms; a result can block a promotion"),
            ("autotune", "Gated search over task settings and candidate features, scored by cross-validation on training entities"),
            ("patterns", "Automatic search for tested patterns across all features of a dataset, with false-discovery control"),
            ("report", "python -m vertiq.report config.json: exported tables to a written report"),
            ("tables", "Mapping and join proposals from the values alone, checked by mapping.repair"),
            ("menu", "Candidate features as reviewable specs (count, since, share, sum, group rate, attribute); no generated code"),
            ("category", "Which of several outcomes comes next, one model per outcome, gated on macro AUC against the past mix"),
            ("plan", "Choice of action per entity under a budget from measured effects, costs and outcome values"),
            ("guard", "Allow, hold or block for proposed actions, from measured effects, intervals and rule flags"),
            ("diagnose", "Rate and mix effects of a change between two periods, by group, with false-discovery control"),
            ("summary", "The last period against the previous one, with chance-adjusted changes"),
            ("drift", "Population Stability Index for inputs and scores, and accuracy on new outcomes"),
            ("sequences", "Lagged A-then-B patterns between consecutive events, filtered for false discoveries"),
            ("profiles", "Precomputed scores and factors for serving, with staleness checks"),
            ("narrate", "Plain-language descriptions of results, and a check that rejects rewrites with numbers not in the result"),
            ("dataset · research · trends", "Training data export for language models, cited external tactics and trend matching")]
    rows = "".join(f"<tr><td><code>{E(a)}</code></td><td>{E(b)}</td></tr>" for a, b in mods)
    body = f"""
<section class="read"><div class="kicker">Architecture</div><h1>Architecture</h1>
<p class="lede">Vertiq has three components: Vertiq LM, the decision engine and the recursive self-improvement loop. The decision
engine runs inside the host application rather than as a separate service: the host loads its records into a common data contract,
calls Vertiq and stores the resulting models, which are plain dictionaries. The engine does not import web frameworks, ORMs or HTTP
clients; <code>tests/test_boundary.py</code> enforces this.</p></section>
<section>{fig_system()}</section>
<section><h2>Design principles</h2>{principles_html()}</section>
<section class="note"><b>Methods.</b> {E(METHODS)}</section>
<section><h2>Modules</h2><div class="tbl"><table><thead><tr><th>Module</th><th>Responsibility</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section class="grid">
  <div class="card"><h3>Extensibility</h3><p>Features and rules are registered under a pack name and selected with <code>pack=</code>. Custom learners
  require <code>fit</code>, <code>predict</code>, <code>to_dict</code> and <code>from_dict</code>; per-group offsets and contributing factors are optional.</p></div>
  <div class="card"><h3>Deployment footprint</h3><p>Python 3.10 or later with no third-party dependencies. The complete benchmark suite runs in
  {SUITE}.</p></div>
  <div class="card"><h3>Integration</h3><p>The host application owns data access, storage and serving. The decision engine has no knowledge of the product that embeds it.</p></div>
</section>"""
    page("architecture.html", "Vertiq architecture", body, "Vertiq architecture: data contract, analysis tasks, plugins and the evaluation gate.")


def build_learning():
    body = f"""
<section class="read"><div class="kicker">Methodology</div><h1>Training and evaluation methodology</h1>
<p class="lede">All tasks follow three rules: features use only information available at prediction time, a fixed share of entities is
excluded from training, and a candidate model replaces the baseline only when it performs better on those entities.</p></section>
<section><h2>Point-in-time features</h2>{fig_cutoffs()}</section>
<section class="grid">
  <div class="card"><h3>Entity-level holdout</h3><p>Approximately 20% of entities, selected by a hash of their identifier, are excluded from
  training. No entity appears in both training and evaluation data.</p></div>
  <div class="card"><h3>Rule-based baseline</h3><p>Event tasks are compared with a recency baseline and value tasks with the entity's previous
  value. Both scores are reported with every training run.</p></div>
  <div class="card"><h3>Champion–challenger comparison</h3><p>A retrained model must perform at least as well as the production model on the
  same held-out entities before it replaces it, and its gain over the baseline must be clear at 90% confidence, from a bootstrap over
  the held-out entities.</p></div>
  <div class="card"><h3>No harm hidden in an average</h3><p>The gain is also checked within each segment the data defines: entity groups,
  labels from an entity table, and length of history. A model clearly worse than the baseline in any segment with at least 30 held-out
  entities is not promoted, and the report names the segment. Nothing is configured per sector.</p></div>
</section>
<section><h2>Causal effect estimation</h2>{fig_uplift()}</section>
<section class="read"><h2>Statistically tested insights</h2>
<p>Vertiq evaluates many candidate patterns at once, such as the day of a first purchase, first-order value and first-order products. Each
candidate receives a p-value, and the set is filtered with the Benjamini–Hochberg procedure at a 10% false-discovery rate. Segments smaller
than 20 entities are excluded. In the benchmark, the planted pattern was detected in {S('insight_discovery')['found_planted']} runs, with
no false product patterns reported on random data.</p></section>
<section><h2>Recursive self-improvement</h2><p class="read">{E(RSI_TEXT)}</p>{fig_rsi()}</section>
<section class="read"><h2>Checking a mapping against the data</h2>
<p>A mapping, whether written by a person, proposed by Vertiq LM or proposed from the values, is checked against every row before use.
Summary and mostly empty rows are named and skipped. Identifiers that stand for "no value" ("-", "NA", a default number on many
rows) become one-off entities instead of merging records. A role whose column is empty, a placeholder or the wrong kind moves to the
clearly best-fitting column or is flagged; an amount column that is zero where another column records the same amount, or that amount
times a quantity, gives way to the complete one. Judgements that need volume wait for 30 rows. Across
{S('mapping_repair')['correct_mappings']['mean'] * CORE['scenarios']['mapping_repair']['seeds']:,.0f} known-correct mappings,
these checks changed {S('mapping_repair')['changed_by_repair']['mean']:.0f}.</p></section>
<section class="read"><h2>Finding patterns without being told where to look</h2>
<p>Every attribute value, group, weekday, part of day and item group becomes a feature. Each is compared with the record's value
and with every other feature, and all p-values are filtered together with Benjamini–Hochberg at 10%, so the more comparisons are
made, the stronger a pattern must be. A pattern must also be large enough to matter. On planted data the search found
{S('pattern_search')['planted_found']['mean']:.0f} of 2 patterns in every run and nothing in random data.</p></section>
<section class="read"><h2>Improving itself under the gate</h2>
<p>The tuning loop tries one change at a time (a learner, a setting, or a feature from a fixed menu of counts, recency and shares),
scores it by five-fold cross-validation on training entities, and keeps it only if the mean improves and most folds agree. The
final model must then beat the baseline and the untuned model on held-out entities. No generated code is run.</p></section>
<section class="grid">
  <div class="card"><h3>Effects of actions</h3><p>With a control group: uplift for any recorded action, with cross-fitted checks. Without one:
  matched cohorts and doubly robust estimation, with balance and overlap reported. Live comparisons use sequential tests that can be
  checked after every observation without inflating false alarms.</p></div>
  <div class="card"><h3>Plans and checks on actions</h3><p>Plans choose an action per entity within a budget from measured effects.
  Proposed actions, from a person or an automated agent, are allowed only when their measured effect is clear; unmeasured
  actions are held for a person.</p></div>
  <div class="card"><h3>Entity tables and leakage</h3><p>Customer lists and asset registers join events by key. Their fixed labels
  describe and predict; totals and dates computed when the file was exported describe only, since using them as features would
  leak the future.</p></div>
</section>
<section class="read"><h2>Testing on data it was not built for</h2>
<p>Synthetic benchmarks plant a known answer. Mapping is also tested on six public exports from six sectors, held out from all
training, with every acceptable answer listed per role; Vertiq LM gets {REAL['summary']['v2-pref']['roles_correct'] * 100:.0f}% of
roles right after validation. New behaviour found on real files becomes a general rule with a test on generated data, and the
check that no correct mapping is changed runs after every change.</p></section>
<section class="note"><b>Data governance.</b> The host application determines which records are eligible for training. Train only on
records with documented consent and exclude records whose owners have opted out. Scoring existing customers is handled separately from training.</section>"""
    page("learning.html", "Vertiq methodology", body, "Point-in-time features, entity-level holdout, control groups and recursive self-improvement.")


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
        return ('<div class="tbl"><table><thead><tr><th>Model</th><th class="n">Loads</th><th class="n">Exact</th><th class="n">Loads, validated</th>'
                '<th class="n">Exact, validated</th><th class="n">Unseen, exact</th><th class="n">Unseen, validated</th></tr></thead>'
                f'<tbody>{body}</tbody></table></div>')
    chart = bars([(r["model"], [("exact", r["overall"]["exact"] * 100, "bar1"), ("exact after validation", r["overall"]["exact_repaired"] * 100, "bar2")])
                  for r in runs_v1], 50, "%", "Exact mappings by model before and after validation")
    body = f"""
<section class="read"><div class="kicker">Vertiq LM</div><h1>Vertiq LM</h1>
<p class="lede">Vertiq LM maps an unfamiliar table to Vertiq's data contract. It identifies the entity, timestamp, event type, value and
grouping columns, the attributes to retain and the date format. It is Qwen 3.5 4B (Apache 2.0) fine-tuned with LoRA on an Apple M1 Pro
with 16 GB of memory, and it is evaluated on tables from domains excluded from training.</p></section>
<section>{fig_lm()}</section>
<section><h2>Results: original test set (114 tables)</h2>{table(runs_v1)}
<figure>{chart}<figcaption>Light bars: exact matches before validation. Dark bars: exact matches after validation. Validation changed no
correct mapping in {S('mapping_repair')['correct_mappings']['mean']:.0f} checks per run across {CORE['scenarios']['mapping_repair']['seeds']} runs.</figcaption></figure></section>
<section><h2>Results: real exports (held out)</h2>
<p class="read">Six public exports from different sectors ({", ".join(sorted({d['sector'] for k, d in json.loads((ROOT / 'benchmarks/real/labels.json').read_text()).items() if not k.startswith('_')}))}),
never used for training. Each role lists every acceptable answer; every proposal is validated before scoring, as in use.</p>
<div class="tbl"><table><thead><tr><th>Method</th><th>Roles correct</th><th>Tables exactly right</th></tr></thead><tbody>
{"".join(f"<tr><td>{E(a)}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in real_rows())}</tbody></table></div></section>
<section><h2>Results: plain-language rewrites (50 held-out descriptions)</h2>
<p class="read">A rewrite counts only if every number is the template's, each number stays next to what it described, and the rewrite is no
longer than the template; accepted rewrites that are near-copies add nothing. Training data: {NARR['training_examples']} rewrites by a local
9B teacher, kept from {NARR['teacher_tried']} attempts.</p>
<div class="tbl"><table><thead><tr><th>Model</th><th>Accepted</th><th>Real rewrites</th><th>Near-copies</th></tr></thead><tbody>
{"".join(f"<tr><td>{E(r['model'])}</td><td>{r['accepted']} of {r['n']}</td><td>{r['accepted_rewrites_not_copies']}</td><td>{r['copies']}</td></tr>" for r in NARR['runs'])}</tbody></table></div></section>
<section><h2>Results: production-like test set (114 tables)</h2>
<p class="read">Includes audit columns and locale-specific number formats (R$ 1.267,81, € 4 350,50, ₹1,267). The earlier model is evaluated
with its original prompt format.</p>{table(runs_v2)}</section>
<section class="grid">
  <div class="card"><h3>Findings</h3><p>Adding column types, inferred from the values, to the prompt raised exact matches on unseen domains from
  {lm_run('pref-on-v2')['unseen']['exact'] * 100:.1f}% to {lm_run('v2-sft')['unseen']['exact'] * 100:.1f}%. Inferring date formats and value types from the rows improved every model.</p></div>
  <div class="card"><h3>Preference round</h3><p>ORPO on the round 3 model's own mistakes raised exact matches after validation from
  {lm_run('v2-sft')['overall']['exact_repaired'] * 100:.1f}% to {lm_run('v2-pref')['overall']['exact_repaired'] * 100:.1f}%, and on unseen domains from
  {lm_run('v2-sft')['unseen']['exact_repaired'] * 100:.1f}% to {lm_run('v2-pref')['unseen']['exact_repaired'] * 100:.1f}%. To fit 16 GB, the 8% of preference
  examples longer than 460 tokens were left out. DPO was not used because it keeps a second copy of the model. Constrained
  decoding was evaluated and not adopted: it would correct 2 of 88 remaining errors.</p></div>
  <div class="card"><h3>Several Macs</h3><p><code>lm/mesh.py</code> sends LoRA sweeps (one job per setting) and evaluations split into
  shards to a job-queue head node; workers on other Apple Silicon Macs run them while those Macs are idle and on wall power. Loss,
  validation loss and peak memory stream to the head's dashboard. A job saves its adapter every 25 steps and resumes on another Mac
  if its own is needed. Only the vertiq package, the job scripts and the named data go to a worker.</p></div>
  <div class="card"><h3>Reproduction</h3><p><code>lm/make_data.py</code>, <code>lm/train.sh</code>, <code>lm/evaluate.py</code>,
  <code>lm/collect_failures.py</code> and <code>lm/pref.sh</code>; <code>lm/round3.sh</code> runs the production-like round end to end.</p></div>
</section>"""
    page("lm.html", "Vertiq LM", body, "Vertiq LM: an open-weight model fine-tuned locally to map source tables to Vertiq's data contract.")


def build_benchmarks():
    def pm(name, key, nd=3, scale=1.0, unit=""):
        v = S(name)[key]
        return f"{v['mean'] * scale:.{nd}f}{unit} <span class='s'>± {v['sd'] * scale:.{nd}f}</span>"
    rows = [
        ("Customer retention", "Two businesses, 500 customers with irregular purchase intervals; two thirds of churners slow down before leaving",
         f"AUC {pm('churn_customers', 'model_auc')} vs {pm('churn_customers', 'baseline_auc')}; Brier {pm('churn_customers', 'model_brier')} vs {pm('churn_customers', 'baseline_brier')}",
         "Ranking comparable to the recency rule; probability estimates about 30% more accurate (Brier score)"),
        ("Equipment failure", "240 assets; each skipped maintenance in the last 60 days triples failure probability",
         f"AUC {pm('machines', 'baseline_auc')} baseline, {pm('machines', 'generic_auc')} generic, {pm('machines', 'plugin_auc')} with a plugin",
         "The generic task detects the signal; one domain feature improves it further"),
        ("Length of stay", "300 patients; stay length increases with recent admissions",
         f"MAE {pm('patients', 'model_mae_days', 2)} days vs {pm('patients', 'baseline_mae_days', 2)} for the previous value",
         "Value prediction outperforms the previous-value baseline"),
        ("Random data", "300 entities with a memoryless 4% weekly event rate",
         f"Models promoted: {NOISE('promoted')} with the default gate; {NOISE('promoted_without_check')} with the confidence check off; "
         f"promoted models worse than the baseline: {total('noise', 'shipped_worse_than_baseline')}",
         "The default gate requires the gain to be clear at 90% confidence and keeps the baseline on most random datasets. "
         "Without that check, a model that only ties the baseline passes about half the time"),
        ("Intervention uplift", "800 customers; the intervention raises response by 30 points for overdue customers only; 25% held out",
         f"Top-ranked half {pm('uplift_targeting', 'lift_top_half', 1, 100, ' pts')}; bottom half {pm('uplift_targeting', 'lift_bottom_half', 1, 100, ' pts')}",
         "Uplift ranking separates responsive from non-responsive customers"),
        ("Offer effect on order value", "Half of interventions include an offer that reduces order value by 20%",
         f"Estimated {pm('uplift_targeting', 'offer_effect_pct', 1, 1, '%')}", "The order-value model recovers the true effect"),
        ("Insight detection", "600 new customers; one first-order product doubles the repeat rate; a matched dataset with no effect",
         f"Planted pattern detected {S('insight_discovery')['found_planted']}; false patterns on random data: {S('insight_discovery')['false_item_findings_in_noise']['mean']:.0f}",
         "False-discovery control holds"),
        ("Non-linear effects", "400 assets; failure risk is high outside a 40–60 degree operating band on either side",
         f"AUC linear {pm('interactions', 'logistic_auc')}, boosted {pm('interactions', 'boosted_auc')}, auto {pm('interactions', 'auto_auc')}",
         f"Boosted trees capture the U-shaped effect; auto chose them in {S('interactions')['auto_chose_boosted']} runs"),
        ("Time to next event", "400 customers with purchase rhythms of 5 to 40 days; half eventually stop",
         f"AUC {pm('renewal_survival', 'plain_auc')} without, {pm('renewal_survival', 'survival_auc')} with the survival feature",
         f"Better in {S('renewal_survival')['survival_better']} runs on rhythm-driven data"),
        ("Effect of any action", "1,500 machines; a programme lowers 30-day failure from 50% to 25% for old machines only; 30% held out",
         f"Estimated effect: old {pm('action_uplift', 'estimated_effect_old')}, new {pm('action_uplift', 'estimated_effect_new')} (true −0.25 and 0)",
         "The uplift model identifies which entities the action helps"),
        ("Effect without a control group", "1,000 entities; treatment is more likely where the outcome is already higher; true effect +1.0",
         f"Matched estimate {pm('matched_cohort', 'matched_att')}; naive difference {pm('matched_cohort', 'naive_difference')}",
         f"Matching removes the confounding; the 90% interval covered the truth in {S('matched_cohort')['interval_covers_truth']} runs"),
        ("Sequential test", "Baseline 10%, smallest worthwhile rate 15%, alpha 0.05, power 0.8; 600 streams per run",
         f"False alarms {pm('sequential', 'false_alarm_rate')}; detection {pm('sequential', 'detection_rate')}; "
         f"{S('sequential')['mean_observations']['mean']:.0f} observations on average against {S('sequential')['fixed_sample_equivalent']['mean']:.0f} for a fixed-sample test",
         "Error rates on target with fewer observations"),
        ("Drift", "Feature rows from the same distribution, and with one input shifted by one standard deviation",
         f"Largest PSI {pm('drift_detection', 'stable_max_psi')} when stable, {pm('drift_detection', 'shifted_psi')} when shifted",
         "Verdicts: stable, and retrain with the shifted input named"),
        ("Lagged sequences", "500 customers; 60% of Apparel orders are followed by Cosmetics 18 to 22 days later",
         f"Found {S('lagged_sequences')['found_planted']}; median lag {S('lagged_sequences')['median_lag_days']['mean']:.0f} days; "
         f"false patterns on random data {pm('lagged_sequences', 'false_patterns_in_random', 1)} per run",
         "Patterns and their timing are recovered"),
        ("Automatic tuning", "The 240-asset plant; the search may switch learner, add the survival feature and add any of four candidate "
         "features: skipped maintenance and three random numbers",
         f"Real feature kept {S('autotune_search')['real_feature_chosen']}; random features kept {S('autotune_search')['noise_features_chosen']['mean']:.1f} per run; "
         f"tuned model promoted {S('autotune_search')['tuned_promoted']}; AUC {pm('autotune_search', 'tuned_auc')} tuned vs {pm('autotune_search', 'untuned_auc')} untuned",
         "Changes are kept only when most validation folds improve; the gate promotes the tuned model only when the held-out gain is clear"),
        ("Two live arms", "Arm A converts at 10%; B at 10% (no difference) or 13%; checked after every observation",
         f"False alarms {pm('two_sample_sequential', 'false_alarm_rate')} against {pm('two_sample_sequential', 'false_alarm_rate_peeking_z_test')} for a z-test "
         f"checked every 100 pairs; detection {pm('two_sample_sequential', 'detection_rate')}, median {S('two_sample_sequential')['median_per_arm_when_detected']['mean']:,.0f} per arm",
         "Continuous monitoring without the inflated false-alarm rate of repeated checks"),
        ("Doubly robust effect", "3,000 entities; treatment more likely where the outcome is already higher; outcome curved (exp), true effect +1.0",
         f"AIPW {pm('doubly_robust_effect', 'doubly_robust')}; outcome model alone {pm('doubly_robust_effect', 'outcome_model_only')}; "
         f"naive {pm('doubly_robust_effect', 'naive_difference')}",
         f"Consistent although the outcome model is wrong; the 90% interval covered the truth in {S('doubly_robust_effect')['interval_covers_truth']} runs"),
        ("Planning under a budget", "2,000 machines, two actions with known effects (a call and a visit), budget 8,000 for 400 machines",
         f"Expected benefit {pm('budget_plan', 'planned_benefit', 0)} planned, {pm('budget_plan', 'random_benefit', 0)} random; "
         f"{S('budget_plan')['share_of_best']['mean'] * 100:.0f}% of the best possible plan",
         "The plan sends visits where they work and calls elsewhere"),
        ("Checks on proposed actions", "300 proposals: a helpful call, a harmful overtime action and an unmeasured part replacement",
         f"Harmful allowed: {total('agent_guard', 'harmful_allowed')} of {total('agent_guard', 'harmful')}; unmeasured held: "
         f"{total('agent_guard', 'unmeasured_held')} of {total('agent_guard', 'unmeasured')}; helpful allowed: "
         f"{total('agent_guard', 'helpful_allowed')} of {total('agent_guard', 'helpful')}, blocked {total('agent_guard', 'helpful_blocked')}",
         "Harm is blocked and the unknown goes to a person; helpful actions pass only in runs where their measured effect was clear, "
         "and are otherwise held for a person"),
        ("Features from the menu", "The 240-asset plant without the hand-written plugin",
         f"Skipped-maintenance count kept {S('menu_search')['kept_skipped_maintenance']}; AUC {pm('menu_search', 'tuned_auc')} tuned vs "
         f"{pm('menu_search', 'untuned_auc')} untuned; tuned promoted {S('menu_search')['tuned_promoted']}",
         "The search finds the feature a person wrote by hand"),
        ("Several possible outcomes", "600 machines; repair, replacement or nothing within 60 days",
         f"Macro AUC {pm('multi_outcome', 'macro_auc')} vs {pm('multi_outcome', 'baseline_macro_auc')} for each machine's past mix",
         "Recent repairs predict a replacement; the past mix does not"),
        ("Why a rate changed", "Plant B's rate doubles while plant C's share doubles",
         f"Only B named as a driver: {S('change_drivers')['drivers_exactly_B']}; C's mix effect {pm('change_drivers', 'c_mix_effect')}, "
         f"rate effect {pm('change_drivers', 'c_rate_effect')}",
         "Rate and mix effects separate exactly"),
        ("Pattern search", "400 orders with two planted patterns (online orders rarely include a drink; pizza orders are worth more), and 400 with none",
         f"Planted found {pm('pattern_search', 'planted_found', 1)} of 2; other findings {pm('pattern_search', 'other_findings', 1)} per run; "
         f"findings in random data {pm('pattern_search', 'findings_in_random_data', 1)}",
         f"{S('pattern_search')['tested']['mean']:.0f} comparisons per dataset under false-discovery control at 10%"),
        ("Mapping validation", "2,100 correct mappings across 15 domains",
         f"Correct mappings altered: {S('mapping_repair')['changed_by_repair']['mean']:.0f}", "Validation never alters a correct mapping"),
    ]
    tr = "".join(f"<tr><td><b>{E(a)}</b><br><span class='s'>{E(b)}</span></td><td>{c}</td><td>{E(d)}</td></tr>" for a, b, c, d in rows)
    m = S("machines")
    chart = bars([("Rule baseline", [("AUC", m["baseline_auc"]["mean"] * 100, "bar0")]),
                  ("Generic task", [("AUC", m["generic_auc"]["mean"] * 100, "bar1")]),
                  ("With domain plugin", [("AUC", m["plugin_auc"]["mean"] * 100, "bar2")])], 100, "", "Equipment failure AUC × 100")
    body = f"""
<section class="read"><div class="kicker">Benchmarks</div><h1>Benchmark results</h1>
<p class="lede">Each scenario uses synthetic data with a known ground truth, so results can be checked against the correct answer. Values
are means ± standard deviation over {RUNS} runs. The full suite runs in {SUITE} ({E(CORE['machine'])}, Python {E(CORE['python'])}).</p></section>
<section><div class="tbl"><table><thead><tr><th>Scenario</th><th>Result</th><th>Interpretation</th></tr></thead><tbody>{tr}</tbody></table></div></section>
<section><h2>Effect of a domain plugin</h2><figure><div class="scroll">{chart}</div><figcaption>Equipment failure prediction on held-out
assets (AUC × 100). A single feature counting skipped maintenance adds information that is not present in the event log.</figcaption></figure></section>
<section class="read"><h2>Real data: Online Retail II</h2>
<p>Repeat purchase on the UCI Online Retail II dataset (CC BY 4.0): {PUBLIC['counts']['lines']:,} invoice lines from a UK online
retailer, 2009 to 2011. Cancellations and lines without a customer are removed, and lines are combined into
{PUBLIC['counts']['orders']:,} orders from {PUBLIC['counts']['customers']:,} customers. The task predicts whether a customer orders again
within {PUBLIC['window_days']} days, with point-in-time features and 20% of customers held out. The baseline is the recency rule.
Recency baseline AUC: {PUBLIC['runs']['linear']['baseline_auc']:.3f}.</p></section>
<section><div class="tbl"><table><thead><tr><th>Learner</th><th>AUC</th><th>90% interval</th><th>Gain over baseline (90%)</th><th>Time</th></tr></thead>
<tbody>{"".join(f"<tr><td>{E(a)}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in public_rows())}</tbody></table></div>
<p class="note-s">One dataset and one task. The scikit-learn row uses the same features and split and confirms the linear learner's
implementation. Boosted trees score slightly higher; the automatic choice, made by 3-fold cross-validation on the training customers,
picked {E(PUBLIC['runs']['auto']['learner'])}.
Reproduce with <code>python benchmarks/public/online_retail.py --data online_retail_II.xlsx</code>.</p></section>
<section class="read"><h2>Real data across sectors</h2>
<p>The one-command report (<code>vertiq-report</code>) on public data from {len(SECT['sectors'])} sectors, nothing tuned per sector: each config
names only the files, the words to use and, where the outcome is a value in a column, the outcome ("Fail"). Vertiq proposes the
mapping, tests patterns, searches candidate inputs and tries 30, 60 and 90-day windows, each through the evaluation gate with its
confidence corrected for the three windows and its segment check. A prediction is promoted only when it beats the rule in use on
held-out entities.</p></section>
<section><div class="tbl"><table><thead><tr><th>Sector</th><th>Data</th><th>Size</th><th>Question</th><th>Result</th><th>Patterns kept</th></tr></thead>
<tbody>{"".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td><td>{E(c)}</td><td>{E(d)}</td><td>{E(e)}</td><td>{E(f)}</td></tr>" for a, b, c, d, e, f in sector_rows())}</tbody></table></div>
<p class="note-s">AUC against the rule in use (recency). "Patterns kept": comparisons that hold after false-discovery control, of all
tested. Full reports: <code>benchmarks/sectors/reports/</code>. Reproduce with <code>benchmarks/sectors/fetch.sh</code> and
<code>python benchmarks/sectors/run.py</code> ({SECT['wall_seconds']:.0f} seconds on {E(SECT['machine'])}).</p></section>
<section class="note"><b>Scope.</b> The synthetic benchmarks verify that the methods detect real signal, reject noise and do not promote
weaker models; the public datasets show the same pipeline on real records from several sectors. None measures accuracy on a specific
organisation's data; the evaluation gate measures that on held-out records at every training run.</section>
<section class="read"><h2>Reproduction</h2><pre><code>python benchmarks/run.py --seeds 5        # writes benchmarks/results/core.json
python benchmarks/collect_lm.py --lm lm   # summarises scored Vertiq LM runs into results/lm.json
python docs/build.py                      # rebuilds this site and the README</code></pre></section>"""
    page("benchmarks.html", "Vertiq benchmarks", body, "Benchmark results for Vertiq on synthetic data with known ground truth, and Vertiq LM evaluation results.")


def validation(g: dict) -> list[str]:
    items = [t for lvl, t in g["evidence"] if lvl != "none"]
    items.append("Every deployment: the evaluation gate measures each model on the organisation's own held-out records before promotion.")
    return items


def build_guides():
    tiles = "".join(tile(g, i, f"{g['slug']}.html") for i, g in enumerate(GUIDES))
    body = f"""
<section class="read"><div class="kicker">Industries</div><h1>Industry implementation guides</h1>
<p class="lede">Vertiq contains no industry-specific code. Each guide documents how an industry's records map to the data contract, which
analyses apply, which domain plugins to add and how results are validated.</p></section>
<section><div class="tiles">{tiles}</div></section>"""
    page("guides/index.html", "Vertiq industry guides", body, "Implementation guides for applying Vertiq in each industry.")
    for i, g in enumerate(GUIDES):
        prev, nxt = GUIDES[i - 1], GUIDES[(i + 1) % len(GUIDES)]
        qs = [f"<li><b>{E(g['event']['q'])}</b> <code>predict_event(target=\"{E(g['event']['target'])}\", window_days={g['event']['window']})</code>"
              + (f"<br><span style='color:var(--muted)'>{E(g['event']['note'])}</span>" if g['event'].get('note') else "") + "</li>"]
        if g["value"]:
            qs.append(f"<li><b>{E(g['value']['q'])}</b> <code>predict_value(target=\"{E(g['value']['target'])}\", window_days={g['value']['window']})</code></li>")
        mods = "".join(f"<li><code>{E(a)}</code>: {E(b)}</li>" for a, b in g["modules"])
        rules = "".join(f"<li><code>rules.{E(r)}</code>: {E(w)}</li>" for r, w in g["rules"])
        val = "".join(f"<li>{E(t)}</li>" for t in validation(g))
        cau = "".join(f"<li>{E(c)}</li>" for c in g["cautions"])
        body = f"""
<section class="guide-head">{tile(g, i, 'index.html')}
<div><div class="kicker"><a href="index.html">Industries</a></div><h1>{E(g['name'])}</h1><p class="lede">{E(g['blurb'])}</p></div></section>
<section><h2>Data mapping</h2>
<dl class="map"><dt>Entity</dt><dd>{E(g['entity'])}</dd><dt>Group</dt><dd>{E(g['group'])}</dd>
<dt>Profile</dt><dd>{E(g['profile'] or 'generic; configure time windows with verticals.register')}</dd></dl>
{fig_mapping(g)}</section>
<section class="read"><h2>Supported analyses</h2><ul>{''.join(qs)}</ul></section>
<section><h2>Reference implementation</h2><pre><code>{E(snippet(g))}</code></pre>
<p class="note-s">This code is executed by the test suite (<code>tests/test_guides.py</code>) against generated records with these column names.</p></section>
<section class="grid">
  <div class="card"><h3>Recommended plugin</h3><p><code>{E(g['plugin']['name'])}</code>: {E(g['plugin']['why'])}</p></div>
  <div class="card"><h3>Validation rules</h3><ul>{rules}</ul></div>
  {f'<div class="card"><h3>Related modules</h3><ul>{mods}</ul></div>' if mods else ''}
  <div class="card"><h3>Implementation notes</h3><ul>{cau}</ul></div>
  <div class="card"><h3>Validation</h3><ul>{val}</ul></div>
</section>
<section class="pager"><a href="{prev['slug']}.html">← {E(prev['name'])}</a><a href="{nxt['slug']}.html">{E(nxt['name'])} →</a></section>"""
        page(f"guides/{g['slug']}.html", f"Vertiq for {g['name'].lower()}", body, g["blurb"])


def build_start():
    """Get started: install, the one-command report, and Python use."""
    cfg = ('{"records": {"file": "orders.xlsx"},\n'
           ' "items": {"file": "order_lines.xlsx"},\n'
           ' "entity_table": {"file": "customers.xlsx"},\n'
           ' "noun": "orders", "currency": "₹", "entities": "customers"}')
    py = """from vertiq.report import read_table
from vertiq.tables import propose
from vertiq.mapping import repair, apply
from vertiq.tasks import predict_event
from vertiq import patterns, narrate

rows = read_table("orders.xlsx")                 # CSV or Excel; header found below titles
m, notes = propose(rows)                         # mapping proposed from the values
m, more = repair(m, rows)                        # checked against every row
ds, problems = apply(m, rows)                    # summary rows skipped, placeholder ids kept apart

for f in patterns.find(ds, noun="orders")["found"]:
    print(f["text"])                             # tested patterns, under false-discovery control

task = predict_event(ds, target="record", window_days=30)
print(narrate.gate(task.report()))               # promoted, kept, or why there is not enough data"""
    body = f"""
<section class="read"><div class="kicker">Get started</div><h1>Install and first report</h1>
<p class="lede">The decision engine is pure Python with no dependencies. One command turns exported tables into a written report;
the same steps are available as functions.</p></section>
<section class="read"><h2>Install</h2><pre><code>pip install vertiq                  # the decision engine
pip install "vertiq[excel]"         # also read .xlsx and .xls exports
pip install "vertiq[lm]"            # Vertiq LM tooling (Apple Silicon)</code></pre></section>
<section class="read"><h2>A report from exported files</h2>
<p>Write a small config naming the files; nothing else is required. Vertiq proposes the mapping from the values, finds how the files
join, checks everything against the rows, and writes what it can say and what it cannot say yet.</p>
<pre><code>{E(cfg)}</code></pre><pre><code>vertiq-report config.json --out report.md</code></pre>
<div class="grid">
  <div class="card"><h3>What the report contains</h3><ul><li>Loading notes: the mapping used, what was checked, rows skipped and why</li>
  <li>What happened: the last period against the one before</li><li>Patterns: every comparison tested, with false-discovery control</li>
  <li>Items bought together, when there are line items</li><li>Predictions that passed the evaluation gate, or the reason they did not</li></ul></div>
  <div class="card"><h3>Shapes it recognises</h3><ul><li>Records with a separate line-item file (orders and order lines)</li>
  <li>A single line-item file (one row per item)</li><li>One row per entity (a customer list, an asset register)</li>
  <li>Exports with title rows, totals rows and misnamed file types</li>
  <li>Large line-item exports: hundreds of thousands of lines and catalogues of thousands of products</li></ul></div>
</div></section>
<section class="read"><h2>In Python</h2><pre><code>{E(py)}</code></pre>
<p>Domain knowledge goes in as plugins (features, rules, learners), and every addition is judged by the same evaluation gate.
See the <a href="architecture.html">architecture</a> and the <a href="guides/index.html">industry guides</a>.</p></section>"""
    page("start.html", "Get started with Vertiq", body, "Install Vertiq and turn exported tables into a written report with one command.")


def build_scope():
    """Scope and roadmap: limitations and what comes next, from the same list as the README."""
    best = lm_run("v2-pref")
    lims, road = scope_items(best)
    body = f"""
<section class="read"><div class="kicker">Scope & roadmap</div><h1>What Vertiq does not do yet, and what comes next</h1>
<p class="lede">Every limitation below is measured or stated in the code; the README carries the same list.</p></section>
<section class="read"><h2>Limitations</h2><ul>{"".join(f"<li>{md_inline(x)}</li>" for x in lims)}</ul></section>
<section class="read"><h2>Roadmap</h2><ul>{"".join(f"<li>{md_inline(x)}</li>" for x in road)}</ul>
<p>Changes by release are in the <a href="{REPO}/blob/main/CHANGELOG.md">changelog</a>.</p></section>"""
    page("scope.html", "Vertiq scope and roadmap", body, "Measured limitations of Vertiq and its roadmap.")


def build_readme():
    """README.md, with every figure read from the same results files as the site."""
    def mm(name, key, nd=3, scale=1.0):
        v = S(name)[key]
        return f"{v['mean'] * scale:.{nd}f} ± {v['sd'] * scale:.{nd}f}"
    core = [
        ("Customer retention", f"AUC {mm('churn_customers', 'model_auc')}, Brier {mm('churn_customers', 'model_brier')}",
         f"Recency rule: AUC {mm('churn_customers', 'baseline_auc')}, Brier {mm('churn_customers', 'baseline_brier')}"),
        ("Equipment failure", f"AUC {mm('machines', 'generic_auc')}; with one plugin {mm('machines', 'plugin_auc')}", f"Recency rule: AUC {mm('machines', 'baseline_auc')}"),
        ("Length of stay", f"MAE {mm('patients', 'model_mae_days', 2)} days", f"Previous value: {mm('patients', 'baseline_mae_days', 2)} days"),
        ("Random data", f"Models promoted: {NOISE('promoted')} (default gate), {NOISE('promoted_without_check')} (confidence check off)", "Worse than the baseline: must be 0"),
        ("Intervention uplift", f"Top half {mm('uplift_targeting', 'lift_top_half', 1, 100)} pts; bottom half {mm('uplift_targeting', 'lift_bottom_half', 1, 100)} pts", "True effect: +30 pts for overdue customers only"),
        ("Offer effect on order value", f"{mm('uplift_targeting', 'offer_effect_pct', 1)}%", "True effect: −20%"),
        ("Insight detection", f"Planted pattern detected {S('insight_discovery')['found_planted']}; false patterns {S('insight_discovery')['false_item_findings_in_noise']['mean']:.0f}", "Benjamini–Hochberg at 10%"),
        ("Non-linear effects (U-shaped)", f"boosted AUC {mm('interactions', 'boosted_auc')}", f"linear AUC {mm('interactions', 'logistic_auc')}"),
        ("Time to next event", f"AUC {mm('renewal_survival', 'survival_auc')} with the survival feature", f"without: {mm('renewal_survival', 'plain_auc')}"),
        ("Effect of any action", f"old {mm('action_uplift', 'estimated_effect_old')}, new {mm('action_uplift', 'estimated_effect_new')}", "true: −0.25 and 0"),
        ("Effect without a control group", f"matched {mm('matched_cohort', 'matched_att')}", f"true 1.0; naive {mm('matched_cohort', 'naive_difference')}"),
        ("Sequential test", f"false alarms {mm('sequential', 'false_alarm_rate')}, detection {mm('sequential', 'detection_rate')}, "
                            f"{S('sequential')['mean_observations']['mean']:.0f} observations", f"targets 0.05 and 0.80; fixed sample {S('sequential')['fixed_sample_equivalent']['mean']:.0f}"),
        ("Drift", f"PSI {mm('drift_detection', 'shifted_psi')} when shifted", f"{mm('drift_detection', 'stable_max_psi')} when stable"),
        ("Lagged sequences", f"planted found {S('lagged_sequences')['found_planted']}, lag {S('lagged_sequences')['median_lag_days']['mean']:.0f} days", "true lag 18 to 22 days"),
        ("Automatic tuning", f"real feature kept {S('autotune_search')['real_feature_chosen']}, random features {S('autotune_search')['noise_features_chosen']['mean']:.1f} per run, "
                             f"AUC {mm('autotune_search', 'tuned_auc')}", f"untuned {mm('autotune_search', 'untuned_auc')}; promoted {S('autotune_search')['tuned_promoted']}"),
        ("Two live arms (mixture SPRT)", f"false alarms {mm('two_sample_sequential', 'false_alarm_rate')}, detection {mm('two_sample_sequential', 'detection_rate')}",
                                         f"z-test checked every 100 pairs: false alarms {mm('two_sample_sequential', 'false_alarm_rate_peeking_z_test')}"),
        ("Doubly robust effect", f"AIPW {mm('doubly_robust_effect', 'doubly_robust')}", f"true 1.0; outcome model alone {mm('doubly_robust_effect', 'outcome_model_only')}; naive {mm('doubly_robust_effect', 'naive_difference')}"),
        ("Planning under a budget", f"{S('budget_plan')['share_of_best']['mean'] * 100:.0f}% of the best possible benefit", f"random choice: {mm('budget_plan', 'random_benefit', 0)} vs planned {mm('budget_plan', 'planned_benefit', 0)}"),
        ("Checks on proposed actions", f"harmful allowed {total('agent_guard', 'harmful_allowed')} of {total('agent_guard', 'harmful')}, unmeasured held {total('agent_guard', 'unmeasured_held')} of {total('agent_guard', 'unmeasured')}",
                                       f"helpful allowed {total('agent_guard', 'helpful_allowed')} of {total('agent_guard', 'helpful')}"),
        ("Features from the menu", f"skipped-maintenance count kept {S('menu_search')['kept_skipped_maintenance']}, AUC {mm('menu_search', 'tuned_auc')}", f"untuned {mm('menu_search', 'untuned_auc')}"),
        ("Several possible outcomes", f"macro AUC {mm('multi_outcome', 'macro_auc')}", f"past mix {mm('multi_outcome', 'baseline_macro_auc')}"),
        ("Why a rate changed", f"only the true driver named: {S('change_drivers')['drivers_exactly_B']}", "rate and mix effects add up to the change"),
        ("Pattern search", f"planted found {mm('pattern_search', 'planted_found', 1)} of 2", f"findings in random data {mm('pattern_search', 'findings_in_random_data', 1)}"),
        ("Mapping validation", f"Correct mappings altered: {S('mapping_repair')['changed_by_repair']['mean']:.0f} of {S('mapping_repair')['correct_mappings']['mean']:.0f} per run", "Must be 0"),
    ]
    core_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in core)
    real_md = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in real_rows())
    narr_md = "\n".join(f"| {r['model']} | {r['accepted']} of {r['n']} | {r['accepted_rewrites_not_copies']} | {r['copies']} |" for r in NARR["runs"])
    public_md = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d, _e in public_rows())
    sector_md = "\n".join(f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in sector_rows())

    def lm_rows(test):
        return "\n".join(f"| {r['model']} | {r['overall']['loads'] * 100:.1f}% | {r['overall']['exact'] * 100:.1f}% | "
                         f"{r['overall']['exact_repaired'] * 100:.1f}% | {r['unseen']['exact'] * 100:.1f}% |"
                         for r in LM["runs"] if r["test_set"] == test)
    challenges = "\n".join(f"| {t} | {f} |" for t, _p, f in CHALLENGES)
    caps = "\n".join(f"| {t} | `{a}` | {d} |" for t, d, a in CAPABILITIES)
    guides = "\n".join(f"- [{g['name']}]({SITE}guides/{g['slug']}.html): {g['blurb']}" for g in GUIDES)
    best = lm_run("v2-pref")
    lims, road = scope_items(best)
    lims_md = "\n".join(f"- {x}" for x in lims)
    road_md = "\n".join(f"- {x}" for x in road)
    text = f"""<p align="center"><img src="https://raw.githubusercontent.com/BiswajitBiswa/vertiq/main/docs/assets/logo.svg" alt="Vertiq" width="290"></p>

<p align="center"><b>Self-improving decision intelligence: a local language model and outcome-trained decision models,<br>improved recursively under an evaluation gate.</b></p>

<p align="center">
<a href="https://github.com/BiswajitBiswa/vertiq/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/BiswajitBiswa/vertiq/actions/workflows/tests.yml/badge.svg"></a>
<a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/licence-Apache%202.0-56633f"></a>
<img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-56633f">
<img alt="No third-party dependencies" src="https://img.shields.io/badge/dependencies-none-7a8765">
</p>

<p align="center"><a href="{SITE}">Documentation</a> · <a href="{SITE}benchmarks.html">Benchmarks</a> · <a href="{SITE}guides/">Industry guides</a> · <a href="lm/README.md">Vertiq LM</a></p>

## Overview

Vertiq is an open-source, self-improving decision intelligence system with three components:

- **Vertiq LM**, a 4-billion-parameter language model fine-tuned locally to map any source table to a common data contract.
- **The decision engine**, which predicts customer, asset and account outcomes, estimates the effect of interventions against control
  groups, and reports tested patterns and rule violations. It is a dependency-free Python package that runs inside the host application.
- **Recursive self-improvement (RSI)**: each generation of a model trains its successor, and a generation is promoted only after it
  outperforms the one in production on held-out data.

## Recursive self-improvement (RSI)

```mermaid
flowchart LR
    T[Teacher model<br/>generates examples] --> V[Validation<br/>each example verified]
    V --> S[Student model trains]
    S --> G{{{{Evaluation gate}}}}
    G -- promoted --> M[Incorrect outputs]
    M -- negative examples, next round --> S
    O[New outcome data] -- each cycle --> S
```

{RSI_TEXT}

For Vertiq LM, the teacher is a larger open-weight model or an OpenAI-compatible endpoint configured by the user. A generated table is
retained only if it loads with its own mapping and the teacher reproduces the mapping without seeing it. Incorrect student outputs are
used as negative examples in preference training (ORPO). Providers whose terms prohibit training on generated output (OpenAI, Anthropic,
Google) are excluded. Applied to table mapping, this process raised exact-match accuracy of a 4-billion-parameter model from
{lm_run('base')['overall']['exact'] * 100:.1f}% to {lm_run('v2-pref-on-v1')['overall']['exact'] * 100:.1f}% on the same held-out set.

## Role alongside large language models

{STACK_ROLE}

| Property | Detail |
|---|---|
{chr(10).join(f"| {t} | {d} |" for t, d in STACK)}

## Challenges addressed

| Challenge | Approach |
|---|---|
{challenges}

## Capabilities

| Capability | API | Description |
|---|---|---|
{caps}

All analyses operate on one data contract of entities, events and actions. Vertiq contains no industry-specific code; the
[industry guides]({SITE}guides/) document the mapping and analyses for each industry.

## Design principles

{principles_md()}

## Architecture

```mermaid
flowchart LR
    A[Source data<br/>CSV, POS, ERP, EHR, CRM] --> B[Mapping<br/>Vertiq LM + validation]
    B --> C[Data contract<br/>Entity, Event, Action]
    C --> D[Analysis tasks<br/>event, value, uplift,<br/>affinity, insights, rules]
    P[Domain plugins<br/>features, rules, learners] --> D
    D --> G{{{{Evaluation gate<br/>held-out entities}}}}
    G -- outperforms baseline and production model --> R[Scores with factors,<br/>ranked interventions, flags]
    G -- otherwise --> K[Production model retained]
```

1. **Mapping.** The host application maps its records once. Vertiq LM proposes the mapping, and `vertiq.mapping.repair` validates
   column names, value types and the date format against the rows. It never introduces a column that does not exist.
2. **Training.** Tasks build examples at successive historical cutoffs, using only information available at each cutoff.
3. **Evaluation.** Approximately 20% of entities, selected by a hash of their identifier, are excluded from training. A candidate is
   promoted only if it outperforms the rule-based baseline and the production model on those entities (`vertiq.harness`).
4. **Explanation.** Each score lists the factors that raised and lowered it; each rule flag includes its supporting evidence.

## Requirements

| Component | Requirement |
|---|---|
| Decision engine (`vertiq`) | Python 3.10, 3.11 or 3.12; no third-party dependencies; any operating system |
| Vertiq LM tooling | Apple Silicon, 16 GB of memory or more, `mlx-lm` and `mlx-lm-lora` (installed with the `lm` extra) |

## Installation

```bash
pip install vertiq                  # the decision engine: pure Python, no dependencies
pip install "vertiq[excel]"         # also read .xlsx and .xls exports in vertiq-report
pip install "vertiq[lm]"            # Vertiq LM tooling (Apple Silicon)
```

## Quick start

```python
import csv
from vertiq.mapping import Mapping, apply, repair
from vertiq.tasks import predict_event

rows = list(csv.DictReader(open("work_orders.csv")))
m, notes = repair(Mapping.parse({{"entity": "Equipment No", "time": "Work Order Date", "type": "Order Type",
                                 "value": "Cost", "group": "Plant", "attrs": [], "date_format": "%Y-%m-%d"}}), rows)
ds, problems = apply(m, rows)            # rows that fail to load are reported, not imputed

task = predict_event(ds, target="breakdown", window_days=30)
print(task.status)                        # trained, kept (baseline retained) or learning (insufficient data)
print(task.report()["metrics"])           # AUC, Brier score, calibration, baseline AUC, held-out entities
print(task.score("EQ-602"))               # {{"chance": ..., "raises": [...], "lowers": [...]}}

from vertiq import narrate
print(narrate.gate(task.report()))        # "A model to predict whether breakdown happens within 30 days was promoted. ..."
print(narrate.prediction(task.score("EQ-602"), target="breakdown", window_days=30))
```

From exported files to a written report, with no mapping written by hand (Vertiq proposes and checks it):

```bash
vertiq-report config.json --out report.md          # or: python -m vertiq.report config.json
# config.json: {{"records": {{"file": "orders.xlsx"}}, "items": {{"file": "order_lines.xlsx"}}, "noun": "orders", "currency": "₹"}}
```

## Extending Vertiq

```python
{PLUGIN_EXAMPLE}
```

| Extension point | Interface |
|---|---|
| `@plugins.feature(pack)` | `fn(entity, events_before, cutoff) -> {{name: number}}`; receives only events recorded before the cutoff |
| `@plugins.rule(pack)`, `vertiq.rules` | `repeat`, `overdue`, `exceeds` and `too_short`, configured with parameters |
| `plugins.register_learner(name, cls)` | Required: `fit(rows, y, **kw) -> self`, `predict(row, group=None)`, `to_dict()`, `from_dict(d)`. Optional: `fit_offsets`, `contributions`. Built in: `logistic`, `linear`, `boosted`, `boosted_regression`, and `auto`, which chooses on training entities only |
| `verticals.register(Vertical(...))` | Time windows, terminology and currency for an industry |
| Vertiq LM teachers | `TEACHER_API_KEY`, `TEACHER_BASE_URL` and `TEACHER_MODEL` for any OpenAI-compatible server |

Plugins add features; the evaluation gate determines whether they improve performance on held-out entities.

## Design guarantees

- **No look-ahead.** Features, including plugin features, use only events recorded before each cutoff.
- **Entity-level holdout.** No entity appears in both training and evaluation data.
- **No unvalidated promotion.** A candidate replaces the production model only if it performs at least as well on the same held-out entities, or by the margin set with `min_gain`.
- **No imputed structure.** Mapping validation selects existing columns or removes the role; rows that fail to load are reported.
- **No network or file access.** The package opens no connections or files. `tests/test_boundary.py` rejects framework, ORM and HTTP imports.
- **Reproducibility.** Benchmarks, the documentation site and this README are generated from committed results; CI fails if they diverge.

## Benchmarks

Synthetic scenarios with known ground truth, {RUNS} runs each, mean ± standard deviation (`python benchmarks/run.py`,
{SUITE}, {CORE['machine']}, Python {CORE['python']}).

| Scenario | Vertiq | Reference |
|---|---|---|
{core_rows}

### Real data: Online Retail II

Repeat purchase within {PUBLIC['window_days']} days for {PUBLIC['counts']['customers']:,} customers ({PUBLIC['counts']['orders']:,} orders, UCI Online Retail II,
CC BY 4.0), 20% of customers held out (`python benchmarks/public/online_retail.py --data online_retail_II.xlsx`).
Recency baseline AUC {PUBLIC['runs']['linear']['baseline_auc']:.3f}.

| Learner | AUC | 90% interval | Gain over baseline (90%) |
|---|---|---|---|
{public_md}

### Real data across sectors

`vertiq-report` on public data from {len(SECT['sectors'])} sectors, nothing tuned per sector; windows of 30, 60 and 90 days, each through the
gate with its confidence corrected for the three, and its segment check (`benchmarks/sectors/fetch.sh`, then
`python benchmarks/sectors/run.py`; full reports in `benchmarks/sectors/reports/`).

| Sector | Data | Size | Question | Result | Patterns kept |
|---|---|---|---|---|---|
{sector_md}

### Real exports (held out)

Six public exports from different sectors, never used for training; a role is correct when it is one of the accepted answers
(`benchmarks/real/labels.json`; `python benchmarks/real/score.py`).

| Method | Roles correct | Tables exactly right |
|---|---|---|
{real_md}

### Plain-language rewrites

{NARR['held_out_descriptions']} held-out descriptions; a rewrite is accepted only if it keeps every number next to what it described and is no longer.

| Model | Accepted | Real rewrites | Near-copies |
|---|---|---|---|
{narr_md}

### Vertiq LM table mapping

Qwen 3.5 4B (Apache 2.0, 4-bit MLX), fine-tuned with LoRA on an Apple M1 Pro with 16 GB of memory. 114 held-out tables across 15
domains; 4 domains (energy, insurance, manufacturing, retail sales) are excluded from training. "Validated" means after
`vertiq.mapping.repair`.

Original test set:

| Model | Loads | Exact | Exact, validated | Unseen domains, exact |
|---|---|---|---|---|
{lm_rows('v1')}

Production-like test set (audit columns and locale-specific number formats):

| Model | Loads | Exact | Exact, validated | Unseen domains, exact |
|---|---|---|---|---|
{lm_rows('v2')}

## Verify the results

Every decision-engine figure can be reproduced from a clean checkout with Python and pytest; the benchmark suite runs in about {SUITE}. Vertiq LM figures require Apple Silicon and the model weights; see [lm/README.md](lm/README.md).

```bash
git clone https://github.com/BiswajitBiswa/vertiq && cd vertiq
python -m pip install pytest && python -m pytest -q     # unit tests and every industry guide's reference implementation
python benchmarks/run.py --seeds 5                      # regenerates benchmarks/results/core.json
python docs/build.py && git diff --stat                 # regenerates the site and this README; no diff means the figures match
```

## Methods

{METHODS}

## Positioning

- **Modelling libraries.** scikit-learn, LightGBM and XGBoost provide general estimators. Vertiq provides the task structure,
  point-in-time features and evaluation gate, and can run any of those estimators through `register_learner`.
- **Model registries.** MLflow and similar tools track and serve models. Vertiq decides whether a retrained model qualifies for
  promotion; the two are complementary.
- **Schema extraction.** General extraction libraries constrain a language model to a schema. Vertiq LM is trained for one task,
  mapping a table to a data contract, and each result is validated against the table's rows.

## Limitations

{lims_md}

## Roadmap

{road_md}

Further reading on recursive self-improvement is kept in [proposals/reading-list.md](proposals/reading-list.md), and an external assessment of the competitive landscape in [proposals/landscape-review.md](proposals/landscape-review.md).

## Industry guides

{guides}

## Repository structure

| Path | Contents |
|---|---|
| `vertiq/` | Decision engine: contract, mapping, tasks, harness, plugins, rules, churn, uplift, value, affinity, insights, inventory, offers |
| `tests/` | Unit tests, the dependency boundary test and every industry guide's reference implementation |
| `benchmarks/` | `run.py`, `collect_lm.py` and `results/*.json` |
| `lm/` | Vertiq LM: data generation, LoRA fine-tuning, evaluation, preference training and the teacher pipeline |
| `docs/` | Documentation site and `build.py`, which also generates this README |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues as described in [SECURITY.md](SECURITY.md). Citation metadata is in
[CITATION.cff](CITATION.cff).

## Licence

Apache License 2.0; see [LICENSE](LICENSE). The Qwen 3.5 weights used by Vertiq LM are not distributed in this repository and are
licensed separately under Apache 2.0.
"""
    (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    for old in ("diagrams.html", "teach-loop.html"):
        (DOCS / old).unlink(missing_ok=True)
    keep = {f"{g['slug']}.html" for g in GUIDES} | {"index.html"}
    for f in (DOCS / "guides").glob("*.html"):
        if f.name not in keep:
            f.unlink()
    build_index(), build_start(), build_architecture(), build_learning(), build_lm(), build_benchmarks(), build_guides(), build_scope(), build_readme()
    (DOCS / ".nojekyll").write_text("")
    print("built", len(list(DOCS.rglob("*.html"))), "pages")
