"""The teacher: a stronger model that writes new training tables for Vertiq LM, every one verified
before it is used.

Teacher chain (first that answers wins, the rest are fallbacks):
  1. any API teachers you configure, all optional, OpenAI-compatible (DeepSeek, OpenRouter, Together,
     a vLLM server...):  TEACHER_API_KEY / TEACHER_BASE_URL / TEACHER_MODEL, and more as
     TEACHER2_API_KEY / TEACHER2_BASE_URL / TEACHER2_MODEL, TEACHER3_..., in that order
  2. the larger local open model (Qwen 3.5 9B, Apache-2.0, MLX), always available, offline

Closed providers whose terms forbid training on their output (OpenAI, Anthropic, Google) are refused
unless TEACHER_ALLOW_CLOSED=1. Each example records which teacher wrote it.

Verification, so the teacher is never trusted on its word:
  * what the rows can settle is settled by the rows (vertiq.mapping.repair: real column names, roles
    held to their value types, the date format); a table whose dates or entity stay ambiguous is
    rejected, and the repaired mapping must then load the table (vertiq.mapping.apply)
  * it must then map the same table blind (the student's exact prompt) and agree with itself
  * rejected tables are kept with the reason (the punish record)
No bias: domains and languages rotate evenly; the four test-only domains are never used.

Run: lm/.venv/bin/python lm/teacher.py --n 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from make_data import FMTS, PROMPT  # noqa: E402
from vertiq.mapping import Mapping, apply, repair  # noqa: E402

LOCAL_TEACHER = ROOT / "models" / "qwen3.5-9b-4bit"
CLOSED = ("api.openai.com", "anthropic.com", "googleapis.com", "generativelanguage")
# new domains the template generator never covered; nothing close to the test-only domains
# (energy, insurance, manufacturing, retail sales)
DOMAINS = ["bank account transactions", "real estate leases", "pharmacy dispensing", "public transport ridership",
           "airline loyalty activity", "vehicle fleet trips", "nonprofit donations", "legal case management",
           "IoT temperature sensors", "gym memberships", "library loans", "municipal permits",
           "freelance marketplace gigs", "mobile app sessions", "SaaS subscriptions", "customer support tickets",
           "veterinary clinic visits", "car rentals", "construction project payments", "school fee payments"]
LANGS = ["English", "Spanish", "German", "French", "Hindi (romanised)", "Portuguese", "Indonesian"]
QUIRKS = ["abbreviated headers", "units inside header names", "currency symbols in amounts", "thousands separators",
          "trailing spaces in headers", "an extra audit column", "codes instead of names", "mixed-case headers"]

WRITE = """Invent a realistic export from a {domain} system, as a table with 4 to 8 columns and exactly 3 rows.
Column names in {lang}, with this real-world quirk: {quirk}. Dates in the time column must use the strftime format {fmt}
{dayhint}. Include one column for who or what each row is about, one for when it happened, and optionally a kind-of-event
column, one numeric amount or measure column (digits only, optionally with a currency symbol or thousands separators),
a site/branch/unit column, and other descriptive or audit columns.
Reply with JSON only:
{{"columns": [...], "rows": [[...], [...], [...]], "mapping": {{"entity": col, "time": col, "type": col or null,
"value": col or null, "group": col or null, "attrs": [descriptive columns, not row numbers or audit columns],
"date_format": "{fmt}"}}}}"""


class APITeacher:
    def __init__(self, name: str, base_url: str, key: str, model: str):
        self.name, self.url, self.key, self.model = name, base_url.rstrip("/"), key, model

    def __call__(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}).encode()
        req = urllib.request.Request(f"{self.url}/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class LocalTeacher:
    name = "local:qwen3.5-9b"

    def __init__(self, path: Path = LOCAL_TEACHER):
        self.path, self._m = path, None

    def __call__(self, prompt: str) -> str:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        if self._m is None:
            self._m = load(str(self.path))
        model, tok = self._m
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=text, max_tokens=700, sampler=make_sampler(temp=0.7), verbose=False)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def chain() -> list:
    """API teachers you configured (optional, in order), then the local model as the fallback."""
    out = []
    for prefix in ["TEACHER"] + [f"TEACHER{i}" for i in range(2, 6)]:
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        if not key:
            continue
        url = os.environ.get(f"{prefix}_BASE_URL", "https://api.deepseek.com").strip()
        if any(c in url for c in CLOSED) and os.environ.get("TEACHER_ALLOW_CLOSED") != "1":
            print(f"skipping {prefix}: {url} is a closed provider whose terms may forbid training on its output")
            continue
        out.append(APITeacher(f"api:{prefix.lower()}", url, key, os.environ.get(f"{prefix}_MODEL", "deepseek-chat")))
    if LOCAL_TEACHER.exists():
        out.append(LocalTeacher())
    if not out:
        raise SystemExit("no teacher: add an API key or download the local model")
    return out


def ask(teachers: list, prompt: str) -> tuple[str, str]:
    """(answer, teacher name) from the first teacher that answers."""
    last = None
    for t in teachers:
        try:
            return t(prompt), t.name
        except Exception as e:          # an API down or out of quota: fall back to the next teacher
            last = e
            print(f"teacher {t.name} failed ({e.__class__.__name__}); falling back")
    raise RuntimeError(f"every teacher failed: {last}")


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in the answer")
    return json.loads(m.group(0))


def candidate(teachers, r: random.Random, i: int) -> dict:
    domain, lang, quirk, fmt = DOMAINS[i % len(DOMAINS)], LANGS[i % len(LANGS)], QUIRKS[r.randrange(len(QUIRKS))], r.choice(FMTS)
    ambiguous = fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y")
    hint = "and at least one date must have a day above 12" if ambiguous else ""
    raw, who = ask(teachers, WRITE.format(domain=domain, lang=lang, quirk=quirk, fmt=fmt, dayhint=hint))
    rec = {"domain": domain, "lang": lang, "quirk": quirk, "teacher": who, "raw": raw[:1500]}
    try:
        d = _json(raw)
        cols, rows = [str(c) for c in d["columns"]], d["rows"]
        if len(rows) != 3 or any(len(row) != len(cols) for row in rows) or len(set(cols)) != len(cols):
            return rec | {"ok": False, "reason": "malformed table"}
        table = [{c: str(v) for c, v in zip(cols, row)} for row in rows]
        truth = Mapping.parse(d["mapping"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return rec | {"ok": False, "reason": f"unparseable: {e}"}
    truth, notes = repair(truth, table)                  # the rows settle what they can, e.g. the date format
    unsure = [n for n in notes if n.startswith(("ambiguous", "confirm"))]
    if unsure:
        return rec | {"ok": False, "reason": "not decidable from the rows: " + unsure[0]}
    _ds, problems = apply(truth, table)
    if problems:
        return rec | {"ok": False, "reason": "own mapping doesn't load: " + problems[0]}
    csv = "\n".join([",".join(cols)] + [",".join(str(row[c]) for c in cols) for row in table])
    prompt = PROMPT.format(csv=csv)
    blind, _w = ask(teachers, prompt)
    try:
        b, _n = repair(Mapping.parse(blind), table)
    except (ValueError, json.JSONDecodeError):
        return rec | {"ok": False, "reason": "blind re-map unparseable"}
    same = (b.as_dict() | {"attrs": sorted(b.attrs)}) == (truth.as_dict() | {"attrs": sorted(truth.attrs)})
    if not same:
        diff = [k for k in ("entity", "time", "type", "value", "group", "date_format") if getattr(b, k) != getattr(truth, k)]
        return rec | {"ok": False, "reason": "blind re-map disagrees on " + (", ".join(diff) or "attrs")}
    answer = json.dumps(truth.as_dict(), ensure_ascii=False)
    return rec | {"ok": True, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    teachers = chain()
    print("teacher chain:", " -> ".join(t.name for t in teachers))
    r = random.Random(a.seed)
    out_dir = ROOT / "data" / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    acc, rej = open(out_dir / "accepted.jsonl", "w"), open(out_dir / "rejected.jsonl", "w")
    stats, reasons, by_dom, by_teacher, started = Counter(), Counter(), Counter(), Counter(), time.time()
    for i in range(a.n):
        try:
            rec = candidate(teachers, r, i)
        except RuntimeError as e:
            rec = {"ok": False, "reason": str(e), "domain": DOMAINS[i % len(DOMAINS)], "teacher": "none"}
        stats["tried"] += 1
        if rec["ok"]:
            stats["accepted"] += 1
            by_dom[rec["domain"]] += 1
            by_teacher[rec["teacher"]] += 1
            acc.write(json.dumps({"messages": rec["messages"], "domain": rec["domain"], "teacher": rec["teacher"]}, ensure_ascii=False) + "\n")
            acc.flush()
        else:
            reasons[rec["reason"].split(":")[0]] += 1
            rej.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rej.flush()
        (ROOT / "runs" / "teacher.json").write_text(json.dumps({"tried": stats["tried"], "target": a.n, "accepted": stats["accepted"],
                                                               "reasons": reasons, "by_domain": by_dom, "by_teacher": by_teacher,
                                                               "chain": [t.name for t in teachers], "seconds": round(time.time() - started)}))
    print(json.dumps({"tried": stats["tried"], "accepted": stats["accepted"], "reasons": reasons}, indent=2))


if __name__ == "__main__":
    main()
