"""Plain-language descriptions of Vertiq's results: a prediction, a gate decision or an effect estimate,
written as sentences a manager can read.

Templates write every description from the result's own fields, so they are always correct. A language
model can optionally rewrite a description into a short paragraph (`describe(..., generate=...)`); the
rewrite is accepted only if every number in it appears in the result (`check`). Otherwise the template
is returned with the reasons. The model is any callable from prompt to text supplied by the host, for
example a local Qwen model (lm/narrate.py); this module imports nothing outside the standard library.
"""
from __future__ import annotations

import json
import re

PROMPT = """Rewrite the description below as one short paragraph for a business reader.
Rules: use only the numbers that appear in the facts or the description; do not calculate new numbers;
do not add causes, advice or certainty that the facts do not state; plain words, no headings, no lists.

Facts (JSON):
{facts}

Description:
{text}

Paragraph:"""


def _pct(x: float, nd: int = 0) -> str:
    return f"{x * 100:.{nd}f}%"


def _pts(x: float) -> str:
    return f"{x * 100:+.1f} points"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3g}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---- templates -----------------------------------------------------------------------------------
def prediction(score: dict, *, target: str, window_days: int) -> str:
    """One entity's score (Task.score) as a sentence: the prediction and the strongest factors."""
    if "chance" in score:
        s = f"{score['entity']}: {_pct(score['chance'])} chance of {target} within {window_days} days."
    else:
        s = f"{score['entity']}: the next {target} is expected to be worth about {score['value']:,.2f}."
    if score.get("raises"):
        s += f" Raised by {_join(score['raises'])}."
    if score.get("lowers"):
        s += f" Lowered by {_join(score['lowers'])}."
    return s


def gate(report: dict) -> str:
    """A Task.report() as sentences: whether a model was promoted, on what evidence, and if not, why."""
    m, status = report.get("metrics") or {}, report["status"]
    target, window = report.get("target", "the event"), report.get("window_days")
    what = (f"whether {target} happens within {window} days" if report.get("task") == "event"
            else f"the value of the next {target}")
    if status == "learning":
        have = report.get("rows", 0)
        return (f"Not enough data yet to predict {what}: {have:,} training rows, {report.get('need_rows', 0):,} needed. "
                f"The baseline rule stays in use.")
    held = m.get("entities_held_out", m.get("held_out"))
    if "auc" in m:
        ev = (f"On {held:,} held-out entities the model ranks outcomes with AUC {m['auc']:.3f}, "
              f"against {m['baseline_auc']:.3f} for the baseline rule.")
    else:
        ev = (f"On {held:,} held-out entities its average error is {m['mae']:,.2f}, "
              f"against {m['baseline_mae']:,.2f} for the baseline rule.")
    ci = (m.get("intervals") or {}).get("gain_ci")
    if ci:
        lvl = (m.get("intervals") or {}).get("level", 0.9)
        unit = "AUC" if "auc" in m else "relative error reduction"
        ev += f" At {_pct(lvl)} confidence the gain is between {ci[0]:+.3f} and {ci[1]:+.3f} ({unit})."
    if status == "trained":
        return f"A model to predict {what} was promoted. {ev}"
    return f"The model to predict {what} was not promoted; the current model or rule stays in use. {ev} Reason: {report.get('reason', 'not stated')}."


def effect(report: dict, *, action: str = "the action") -> str:
    """An uplift report (predict_uplift) as sentences: the average effect and what to do next."""
    m, target = report.get("metrics") or {}, report.get("target", "the event")
    if report["status"] == "learning":
        need = report.get("need", {})
        return (f"Not enough data yet to measure the effect of {action}: {report.get('treated', 0):,} treated and "
                f"{report.get('control', 0):,} control entities; at least {need.get('treated', 0):,} and {need.get('control', 0):,} are needed.")
    a = m["average_effect"]
    s = (f"Among entities that received {action}, {_pct(a['treated_rate'], 1)} had {target} within {report.get('window_days')} days, "
         f"against {_pct(a['control_rate'], 1)} in the control group: an average effect of {_pts(a['effect'])} ({_p(a['p_value'])}).")
    nxt = m.get("next_step") or {}
    if nxt.get("decision") == "effect detected":
        s += " The effect is statistically clear."
    elif nxt.get("needed"):
        n = nxt["needed"]
        s += (f" The effect is not yet clear. Confirming an effect of {_pts(n['effect'])} needs about {n['treated']:,} treated "
              f"and {n['control']:,} control entities.")
    if report["status"] == "trained":
        s += " The model ranks entities by their individual effect, so the action can go to those it helps most."
    else:
        s += " Individual effects could not be separated reliably; use the average effect."
    return s


# ---- checking a rewrite ----------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[tuple[float, int]]:
    """Every standalone number in `text` with its number of decimals ("C123" and "v2" are not numbers)."""
    out = []
    for raw in _NUM.findall(text):
        clean = raw.replace(",", "").replace("−", "-")
        if clean.lstrip("+-") == "":
            continue
        out.append((float(clean), len(clean.split(".")[1]) if "." in clean else 0))
    return out


def facts(*sources) -> set[float]:
    """All numbers in the given results (dicts, lists, strings), with their percentage forms."""
    found: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            for v in (float(x), float(x) * 100):
                found.update((v, abs(v)))
        elif isinstance(x, str):
            for v, _d in numbers(x):
                found.update((v, abs(v)))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
    for s in sources:
        walk(s)
    return found


def check(text: str, allowed: set[float]) -> list[str]:
    """Problems with a rewrite: every number must equal an allowed number at the precision written
    ("0.81" matches 0.8117; "4.5" matches nothing unless 4.5 is a fact)."""
    problems = []
    if not text.strip():
        return ["empty text"]
    for v, d in numbers(text):
        tol = 0.5 * 10 ** -d + 1e-9
        if not any(abs(v - f) <= tol for f in allowed):
            problems.append(f"the number {v:g} is not in the facts")
    return problems


def describe(text: str, *sources, generate=None) -> dict:
    """{"text", "source": "template" | "model", "problems"}. Without `generate`, the template text.
    With it, the model's rewrite if it passes `check` against the template and the sources, else the
    template with the problems found and the rejected rewrite under "rejected"."""
    if generate is None:
        return {"text": text, "source": "template", "problems": []}
    payload = json.dumps(sources[0] if len(sources) == 1 else list(sources), default=str)
    try:
        out = (generate(PROMPT.format(facts=payload, text=text)) or "").strip()
    except Exception as e:                                       # the model is optional; never fail the caller
        return {"text": text, "source": "template", "problems": [f"model error: {type(e).__name__}"]}
    problems = check(out, facts(text, *sources))
    if problems:                                                 # the rejected rewrite is kept for review or training
        return {"text": text, "source": "template", "problems": problems, "rejected": out}
    return {"text": out, "source": "model", "problems": []}
