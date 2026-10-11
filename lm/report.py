"""Turn this run's files into one self-contained page (lm/runs/report.html): the data that was
generated, how it was checked, the base model's scores, the training curve, and the fine-tuned
model's scores next to the base. Re-run it any time; the page refreshes itself while training runs.

Run: lm/.venv/bin/python lm/report.py   (or with --watch to regenerate every 20 s while training runs)
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS, DATA = ROOT / "runs", ROOT / "data"
TOTAL = 500
DPO_TOTAL = 300
V2_TOTAL = 1000


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _training(log_name: str = "train.log", proc: str = "adapter-path lm/adapters/mapping$", adapter: str = "mapping", total: int = TOTAL):
    """(val points, train points, step, running, done, oom) from a training log."""
    log = RUNS / log_name
    text = log.read_text(errors="ignore") if log.exists() else ""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\r", "\n")
    val = [(int(i), float(v)) for i, v in re.findall(r"^\s*(\d+)\s+val\s+([\d.]+)", clean, re.M)]
    # training rows look like "   325    0.134 ▲       12     19.4k" (loss, trend arrow, tok/s, tokens)
    train = [(int(i), float(v)) for i, v in re.findall(r"^\s*(\d+)\s+([\d.]+)\s+\S+\s+[\d.]+\s+[\d.]+k?\s*$", clean, re.M)]
    val += [(int(i), float(v)) for i, v in re.findall(r"[Ii]ter (\d+): [Vv]al loss ([\d.]+)", clean)]
    train += [(int(i), float(v)) for i, v in re.findall(r"[Ii]ter (\d+): [Tt]rain loss ([\d.]+)", clean)]
    steps = [i for i, _ in train + val] + [int(x) for x in re.findall(r"(\d+)/" + str(total), clean)]
    running = subprocess.run(["pgrep", "-f", proc], capture_output=True).returncode == 0
    done = (ROOT / "adapters" / adapter / "adapters.safetensors").exists() and not running
    oom = "Insufficient Memory" in clean
    return val, train, max(steps or [0]), running, done, oom


def _chart(val, train, w=560, h=200, total=TOTAL):
    pts = val + train
    if not pts:
        return "<p class='muted'>Waiting for the first loss value…</p>"
    xmax = max(total, max(i for i, _ in pts))
    ymax = max(v for _, v in pts) * 1.1
    X = lambda i: 40 + (w - 60) * i / xmax      # noqa: E731
    Y = lambda v: h - 30 - (h - 50) * v / ymax  # noqa: E731
    g = [f"<line x1='40' y1='{h-30}' x2='{w-20}' y2='{h-30}' class='ax'/>", f"<line x1='40' y1='20' x2='40' y2='{h-30}' class='ax'/>"]
    for k in range(0, 6):
        v = ymax * k / 5
        g.append(f"<text x='34' y='{Y(v)+4:.0f}' class='tk' text-anchor='end'>{v:.2f}</text>")
    for s in range(0, xmax + 1, max(50, (xmax // 5) // 50 * 50)):
        g.append(f"<text x='{X(s):.0f}' y='{h-12}' class='tk' text-anchor='middle'>{s}</text>")
    if len(train) > 1:
        g.append("<polyline class='tr' points='" + " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in train) + "'/>")
    if val:
        g.append("<polyline class='va' points='" + " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in val) + "'/>")
        g += [f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='4' class='vd'><title>step {i}: {v}</title></circle>"
              f"<text x='{X(i)+6:.0f}' y='{Y(v)-8:.0f}' class='tk'>{v:.3f}</text>" for i, v in val]
    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='Training and validation loss by step'>{''.join(g)}</svg>"
            "<div class='lg'><span><i class='sw va'></i>Validation loss (held-out examples)</span>"
            "<span><i class='sw tr'></i>Training loss</span></div>")


def _bar(v):
    pct = round(100 * (v or 0))
    return f"<div class='bar'><i style='width:{pct}%'></i><span>{pct}%</span></div>"


def _metrics_table(base, tuned, dpo=None):
    rows = [("Valid JSON", "valid_json"), ("Mapping loads the data", "loads_data"), ("Every field exactly right", "exact"),
            ("Roles correct (entity, time, type, value, group)", "roles_correct"), ("Attributes overlap", "attrs_overlap")]
    head = "<tr><th>Measure</th><th>Base model</th><th>Fine-tuned (SFT)</th><th>+ reward / punish (ORPO)</th></tr>"
    body = "".join(f"<tr><td>{n}</td><td>{_bar(base['overall'][k]) if base else '…'}</td>"
                   f"<td>{_bar(tuned['overall'][k]) if tuned else '<span class=muted>after training</span>'}</td>"
                   f"<td>{_bar(dpo['overall'][k]) if dpo else '<span class=muted>after ORPO</span>'}</td></tr>" for n, k in rows)
    return f"<table>{head}{body}</table>"


def _domains(base, tuned, dpo=None):
    if not base:
        return ""
    head = "<tr><th>Domain</th><th>In training?</th><th>Base: loads</th><th>SFT: loads</th><th>ORPO: loads</th><th>Base: exact</th><th>SFT: exact</th><th>ORPO: exact</th></tr>"
    body = ""
    for d, v in base["per_domain"].items():
        t = (tuned or {}).get("per_domain", {}).get(d)
        q = (dpo or {}).get("per_domain", {}).get(d)
        body += (f"<tr><td>{html.escape(d)}</td><td>{'yes' if v['seen'] else '<b>never seen</b>'}</td><td>{_bar(v['loads_data'])}</td>"
                 f"<td>{_bar(t['loads_data']) if t else '…'}</td><td>{_bar(q['loads_data']) if q else '…'}</td>"
                 f"<td>{_bar(v['exact'])}</td><td>{_bar(t['exact']) if t else '…'}</td><td>{_bar(q['exact']) if q else '…'}</td></tr>")
    return f"<table class='dom'>{head}{body}</table>"


def _round3(v1, sft, pref) -> str:
    if not (v1 or sft or pref):
        return ""
    runs = [("Earlier best (ORPO v1)", v1), ("Round 3 SFT", sft), ("Round 3 ORPO", pref)]
    rows = [("Mapping loads the data", "loads_data"), ("Every field exactly right", "exact"),
            ("Loads, after the data repair", "loads_repaired"), ("Exactly right, after the data repair", "exact_repaired")]
    head = "<tr><th>Measure</th>" + "".join(f"<th>{n}</th>" for n, _ in runs) + "</tr>"
    body = "".join(f"<tr><td>{label}</td>" + "".join(f"<td>{_bar(r['overall'].get(k)) if r else '…'}</td>" for _, r in runs) + "</tr>"
                   for label, k in rows)
    unseen = "".join(f"<td>{_bar(r['unseen_domains'].get('exact_repaired')) if r else '…'}</td>" for _, r in runs)
    doms = ""
    ref = next((r for _, r in runs if r), None)
    for d, v in ref["per_domain"].items():
        doms += f"<tr><td>{html.escape(d)}</td><td>{'yes' if v['seen'] else '<b>never seen</b>'}</td>" + "".join(
            f"<td>{_bar(r['per_domain'][d].get('exact_repaired')) if r else '…'}</td>" for _, r in runs) + "</tr>"
    return (f"<div class='card' style='margin-top:12px'><h2>Round 3 results (the same {ref['overall']['n']} new held-out tables)</h2>"
            f"<table>{head}{body}<tr><td>Never-seen domains: exactly right, after repair</td>{unseen}</tr></table>"
            f"<h2 style='margin-top:14px'>Per domain: exactly right, after repair</h2><table class='dom'><tr><th>Domain</th><th>In training?</th>"
            + "".join(f"<th>{n}</th>" for n, _ in runs) + f"</tr>{doms}</table></div>")


def _snap_note() -> str:
    s = _load(RUNS / "snap.json")
    if not s:
        return ""
    return ("<p class='warn'><b>Plus the verifier's repair step</b> (snap misspelled column names to the closest real column, "
            f"never invent one): fine-tuned mappings load the data <b>{100 * s['tuned']['loads']:.0f}%</b> of the time and "
            f"<b>{100 * s['tuned']['exact']:.0f}%</b> are exactly right; the base model with the same repair loads "
            f"{100 * s['base']['loads']:.0f}%.</p>")


def build() -> Path:
    meta = _load(DATA / "meta.json") or {}
    base, tuned, dpo = _load(RUNS / "base.json"), _load(RUNS / "tuned.json"), _load(RUNS / "pref.json")
    fails = _load(RUNS / "failures-v1.json") or _load(RUNS / "failures.json")
    teach = _load(RUNS / "teacher.json")
    teaching = subprocess.run(["pgrep", "-f", "lm/teacher.py"], capture_output=True).returncode == 0
    collecting = subprocess.run(["pgrep", "-f", "collect_failures.py"], capture_output=True).returncode == 0
    dval, dtrain, dstep, drun, ddone, doom = _training("pref.log", "adapter-path lm/adapters/mapping-pref$", "mapping-pref", DPO_TOTAL)
    v2val, v2train, v2step, v2run, v2done, _ = _training("train-v2.log", "adapter-path lm/adapters/mapping-v2$", "mapping-v2", V2_TOTAL)
    p2val, p2train, p2step, p2run, p2done, _ = _training("pref-v2.log", "adapter-path lm/adapters/mapping-pref-v2$", "mapping-pref-v2", DPO_TOTAL)
    v2meta = _load(DATA / "v2" / "meta.json") or {}
    v2sft, v1on2, v2pref, v2on1 = (_load(RUNS / f"{n}.json") for n in ("v2-sft", "pref-on-v2", "v2-pref", "v2-pref-on-v1"))
    round3 = subprocess.run(["pgrep", "-f", "lm/round3.sh"], capture_output=True).returncode == 0
    fails2 = _load(RUNS / "failures-v2.json") or (_load(RUNS / "failures.json") if collecting else None)
    v2ex = None
    if (DATA / "v2" / "train.jsonl").exists():
        v2ex = json.loads((DATA / "v2" / "train.jsonl").read_text().splitlines()[0])["messages"][0]["content"].split("Reply with")[0]
    val, train, step, running, done, oom = _training()
    ex = None
    if (DATA / "train.jsonl").exists():
        first = json.loads((DATA / "train.jsonl").read_text().splitlines()[0])["messages"]
        ex = (first[0]["content"].split("Reply with")[0].split("(CSV):")[1].strip(), first[1]["content"])
    st = lambda ok, live=False: "<span class='st live'>running</span>" if live else ("<span class='st ok'>done</span>" if ok else "<span class='st wait'>next</span>")  # noqa: E731
    steps = [
        ("1", "Generate training data", st(bool(meta)),
         f"<p>{meta.get('counts', {}).get('train', '…')} training, {meta.get('counts', {}).get('valid', '…')} validation and "
         f"{meta.get('counts', {}).get('test', '…')} test tables across 15 domains, equal share each. Column names mix English, Spanish, "
         f"German, French and Hindi in 6 naming styles; every date format appears.</p>"
         f"<p><b>Never seen in training</b> (random draw, seed {meta.get('seed', '')}): {', '.join(meta.get('unseen_domains', []))}.</p>"
         + (f"<p class='muted'>One example table and its correct answer:</p><pre>{html.escape(ex[0])}</pre><pre class='ans'>{html.escape(ex[1])}</pre>" if ex else "")),
        ("2", "Verify the answers", st(bool(meta)),
         "<p>Every test answer was loaded against its own rows with <code>vertiq.mapping.apply</code>: <b>0 failures</b>. "
         "Day-first and month-first dates are always decidable from the rows.</p>"),
        ("3", "Score the base model (before training)", st(bool(base)),
         f"<p>Qwen 3.5 4B (Apache-2.0, 4-bit MLX), no training, on all held-out tables"
         + (f" ({base['seconds']} s)" if base else "") + ".</p>"),
        ("4", "Fine-tune with LoRA on this Mac", st(done, running),
         "<p class='warn'>First attempt (batch 2, 8 layers) ran out of GPU memory at step 3. Retried with batch 1 and 4-step "
         "gradient accumulation, 4 LoRA layers, gradient checkpointing, 640-token sequences.</p>"
         + f"<p>Step <b>{step}</b> of {TOTAL} · 4.06M trainable parameters (0.096% of the model) · only the answers are learned.</p>"
         + f"<div class='prog'><i style='width:{100 * step / TOTAL:.0f}%'></i></div>" + _chart(val, train)),
        ("5", "Score the fine-tuned model", st(bool(tuned)),
         "<p>The same held-out tables, the same checks, with the LoRA adapter loaded.</p>"),
        ("6", "Collect the model's own mistakes", st(bool(fails) and not collecting and "pairs" in (fails or {}), collecting),
         "<p>Fresh tables (new seed) from the 11 trained domains only, 36 each, so no domain is favoured and the 4 test-only "
         "domains stay unseen. Every answer that isn't exactly right becomes a pair: <b>chosen</b> = the correct mapping "
         "(rewarded), <b>rejected</b> = what the model wrote (punished).</p>"
         + (f"<p>{fails['done']} of {fails['total']} tables mapped · <b>{sum(fails['failed'].values())}</b> mistakes kept"
            + (f" → {fails.get('train')} training and {fails.get('valid')} validation pairs" if 'pairs' in fails else "") + ".</p>"
            f"<div class='prog'><i style='width:{100 * fails['done'] / max(1, fails['total']):.0f}%'></i></div>"
            "<p class='muted'>Mistakes per domain: " + " · ".join(f"{d} {fails['failed'].get(d, 0)}/{n}" for d, n in fails['tried'].items()) + "</p>"
            if fails else "")),
        ("7", "Reward / punish: preference training on those mistakes (ORPO)", st(ddone, drun),
         "<p class='warn'>DPO was tried first: it keeps a second copy of the model as its reference, ran at ~4 minutes a step "
         "and then ran out of GPU memory on this 16 GB Mac. Switched to ORPO, which rewards the correct answer and punishes the "
         "model's own wrong one with no reference model.</p>"
         "<p>Starts from the fine-tuned model (fused). Kept only if it beats the fine-tuned model.</p>"
         + (f"<p>Step <b>{dstep}</b> of {DPO_TOTAL}</p><div class='prog'><i style='width:{100 * dstep / DPO_TOTAL:.0f}%'></i></div>"
            + _chart(dval, dtrain, total=DPO_TOTAL) if (drun or ddone or dval) else "")),
        ("8", "Score after preference training (keep it only if better)", st(bool(dpo)), "<p>The same 114 held-out tables and checks.</p>"),
        ("9", "Teacher writes new tables (verified)", st(bool(teach) and not teaching and teach.get("tried") == teach.get("target"), teaching),
         "<p>Teacher chain: optional API teachers you configure first, then the larger local open model "
         "(Qwen 3.5 9B, Apache-2.0) as the fallback. It invents realistic, messy tables from 20 new domains in 7 "
         "languages. What the rows can settle is settled by the rows first (real column names, roles held to their value types, "
         "the date format); a table is accepted only if its mapping then loads the table <b>and</b> the teacher, mapping it "
         "blind, agrees with itself. Ambiguous tables are rejected. Rejections are kept with the reason. The 4 test-only "
         "domains are never used.</p>"
         "<p class='warn'>First run: 1 of 24 accepted. The 9B wrote good tables but mislabelled their date formats; with the "
         "rows deciding the format, a trial accepted 4 of 8.</p>"
         + (f"<p>{teach['tried']} of {teach['target']} tried · <b>{teach['accepted']}</b> accepted · chain: {' → '.join(teach['chain'])}</p>"
            f"<div class='prog'><i style='width:{100 * teach['tried'] / max(1, teach['target']):.0f}%'></i></div>"
            "<p class='muted'>Rejected because: " + " · ".join(f"{k} {v}" for k, v in teach['reasons'].items()) + "</p>" if teach else "")),
        ("10", "Round 3 data: the model sees what each column holds", st(bool(v2meta)),
         "<p>Error analysis of the best model: 86 of the 88 remaining role errors were <b>real columns chosen wrongly</b> "
         "(mostly the event-type column filed as a plain attribute, or a date column taken as the entity). So the prompt now "
         "carries each column's kind, read from its values by code: id, date with its format, number, row counter, text. "
         "Tables are more realistic too: audit columns hold what their names say, and amounts come in real locale styles.</p>"
         + (f"<pre>{html.escape(v2ex.split('(CSV):')[1].strip())}</pre>" if v2ex else "")),
        ("11", "Fine-tune on round 3 data + the teacher's verified tables", st(v2done, v2run),
         f"<p>From the base model, about one pass over the data. Step <b>{v2step}</b> of {V2_TOTAL}.</p>"
         + f"<div class='prog'><i style='width:{100 * v2step / V2_TOTAL:.0f}%'></i></div>" + (_chart(v2val, v2train, total=V2_TOTAL) if (v2val or v2train) else "")),
        ("12", "Score it next to the best earlier model, on the same new tables", st(bool(v2sft and v1on2)),
         "<p>The earlier ORPO model is shown the same tables in the prompt format it was trained on, so the comparison is fair.</p>"),
        ("13", "Collect its mistakes (denser: 60 tables per trained domain)", st(bool(_load(RUNS / "failures-v2.json")), collecting and round3),
         (f"<p>{fails2['done']} of {fails2['total']} mapped · <b>{sum(fails2['failed'].values())}</b> mistakes kept.</p>"
          f"<div class='prog'><i style='width:{100 * fails2['done'] / max(1, fails2['total']):.0f}%'></i></div>" if fails2 else "")),
        ("14", "Reward / punish on those mistakes (ORPO)", st(p2done, p2run),
         f"<p>Step <b>{p2step}</b> of {DPO_TOTAL}</p><div class='prog'><i style='width:{100 * p2step / DPO_TOTAL:.0f}%'></i></div>"
         + (_chart(p2val, p2train, total=DPO_TOTAL) if (p2val or p2train) else "")),
        ("15", "Final score (kept only if it beats the earlier best)", st(bool(v2pref)), "<p>Same tables, same checks, raw and with the data repair.</p>"),
    ]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{'<meta http-equiv="refresh" content="20">' if running or collecting or drun or teaching or round3 else ''}
<title>Vertiq LM training</title><style>
:root{{--bg:#f7f6f2;--p:#fff;--ink:#16161a;--mu:#5d5c58;--li:#dedcd4;--ok:#1baf7a;--okb:#e3f6ee;--run:#2a78d6;--runb:#e6f0fb;--wa:#c98500;--wab:#fbf0d6;--lm:#6250d6;--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;--mono:ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141413;--p:#1d1d1b;--ink:#f0efec;--mu:#c3c2b7;--li:#34332f;--ok:#2fc78f;--okb:#123326;--run:#4a96ea;--runb:#152a40;--wa:#e0a63a;--wab:#3a2d10;--lm:#9085e9}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--font)}}.w{{max-width:1080px;margin:0 auto;padding:22px 24px 40px}}
h1{{font-size:24px;margin:0}}.lede{{color:var(--mu);margin:4px 0 16px}}.muted{{color:var(--mu)}}
.step{{display:grid;grid-template-columns:40px minmax(0,1fr);gap:12px;margin-bottom:12px}}.n{{width:32px;height:32px;border-radius:50%;background:var(--ink);color:var(--bg);display:flex;align-items:center;justify-content:center;font-weight:700}}
.card{{min-width:0;background:var(--p);border:1px solid var(--li);border-radius:14px;padding:12px 16px}}.card h2{{font-size:16px;margin:0 0 6px;display:flex;gap:10px;align-items:center}}
.st{{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px}}.ok{{background:var(--okb);color:var(--ok)}}.live{{background:var(--runb);color:var(--run)}}.wait{{background:var(--li);color:var(--mu)}}
.warn{{background:var(--wab);border-left:3px solid var(--wa);padding:6px 10px;border-radius:0;font-size:13.5px}}
pre{{background:var(--bg);border:1px solid var(--li);border-radius:8px;padding:8px 10px;font:12px/1.5 var(--mono);overflow-x:auto;margin:6px 0}}pre.ans{{border-color:var(--ok)}}
code{{font-family:var(--mono);font-size:12.5px}}.prog{{height:8px;background:var(--li);border-radius:99px;overflow:hidden;margin:6px 0 10px}}.prog i{{display:block;height:100%;background:var(--run)}}
svg{{width:100%;max-width:640px;height:auto}}.ax{{stroke:var(--li)}}.tk{{font-size:10.5px;fill:var(--mu);font-family:var(--font)}}.va{{fill:none;stroke:var(--lm);stroke-width:2.5}}.tr{{fill:none;stroke:var(--mu);stroke-width:1;opacity:.6}}.vd{{fill:var(--lm)}}
.lg{{display:flex;gap:14px;font-size:12.5px;color:var(--mu)}}.lg span{{display:inline-flex;gap:6px;align-items:center}}.sw{{width:14px;height:3px;display:inline-block}}.sw.va{{background:var(--lm)}}.sw.tr{{background:var(--mu)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}}th,td{{text-align:left;padding:6px 8px;border-top:1px solid var(--li)}}th{{font-size:11px;text-transform:uppercase;color:var(--mu);border-top:0}}
.bar{{position:relative;height:16px;background:var(--li);border-radius:4px;min-width:90px}}.bar i{{position:absolute;inset:0 auto 0 0;background:var(--ok);border-radius:4px;opacity:.75}}.bar span{{position:relative;padding-left:6px;font-size:11.5px;font-weight:600}}
@media (max-width:700px){{.w{{padding:16px}}.dom th:nth-child(5),.dom td:nth-child(5),.dom th:nth-child(6),.dom td:nth-child(6){{display:none}}}}
</style></head><body><div class="w">
<h1>Vertiq LM: first job, mapping any table to Vertiq's contract</h1>
<p class="lede">Open-source base model, fine-tuned on this Mac, checked by the data itself. Updated {time.strftime('%H:%M:%S')}{' · refreshing every 20 s' if (running or teaching or round3) else ''}.</p>
{''.join(f"<div class='step'><div class='n'>{n}</div><div class='card'><h2>{t} {s}</h2>{b}</div></div>" for n, t, s, b in steps)}
<div class='card'><h2>Results: base vs fine-tuned (all {base['overall']['n'] if base else '…'} held-out tables)</h2>{_metrics_table(base, tuned, dpo)}
{_snap_note()}
<h2 style='margin-top:14px'>Per domain (none hidden in an average)</h2>{_domains(base, tuned, dpo)}</div>
{_round3(v1on2, v2sft, v2pref)}
</div></body></html>"""
    RUNS.mkdir(exist_ok=True)
    out = RUNS / "report.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    if "--watch" in sys.argv:
        while True:
            build()
            if _load(RUNS / "v2-pref-on-v1.json"):
                break
            time.sleep(20)
    print(build())
