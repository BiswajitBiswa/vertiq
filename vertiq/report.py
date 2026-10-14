"""One command from exported tables to a written report: everything Vertiq can say about the data, in its
own words, and what it cannot say yet.

  python -m vertiq.report config.json [--out report.md]

config.json names the tables and how they map to the contract; nothing else is configured:

  {"records": {"file": "orders.csv", "mapping": {"entity": "Customer", "time": "Date", "clock": "Time",
               "value": "Total", "attrs": ["Order No", "Channel"], "date_format": "%d/%m/%Y"}},
   "items":   {"file": "order_lines.csv", "key": "Order No", "event_key": "Order No", "item": "Item",
               "qty": "Qty", "item_group": "Category"},
   "noun": "orders", "currency": "₹"}

CSV is read with the standard library; .xlsx needs openpyxl (pip install openpyxl). Every mapping is checked
against the rows by vertiq.mapping.repair before use, and every section below comes from a Vertiq module:
summary and narrate (what happened), patterns (tested patterns), affinity (items bought together) and the
tasks behind the evaluation gate (predictions, or why there are none yet).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import affinity, narrate, patterns, summary
from .mapping import Mapping, apply, attach_items, repair
from .tasks import predict_event, predict_value


def read_table(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError as e:
            raise SystemExit("reading .xlsx needs openpyxl: pip install openpyxl (or save the sheet as CSV)") from e
        ws = load_workbook(p, read_only=True, data_only=True).worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        head = [str(h).strip() if h is not None else f"column {i + 1}" for i, h in enumerate(rows[0])]
        return [{h: ("" if v is None else str(v)) for h, v in zip(head, r)} for r in rows[1:]]
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return [{(k or "").strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def build(config: dict, base: Path = Path(".")) -> tuple[object, dict, list[str]]:
    """The dataset, item groups and loading notes from a config. Without a mapping, Vertiq proposes one from
    the values (vertiq.tables.propose); without join columns, it finds them (vertiq.tables.find_join)."""
    from dataclasses import replace
    from .tables import find_join, propose
    rec = config["records"]
    rows = read_table(base / rec["file"])
    log = []
    it = dict(config.get("items") or {})
    lines = read_table(base / it["file"]) if it else []
    if it and not it.get("key"):
        found, why = find_join(rows, lines)          # first: the join key names the record, not whom it is about
        log += [f"Line items: {w}" for w in why]
        if found:
            it.update(found)
    if rec.get("mapping"):
        proposal = Mapping.parse(rec["mapping"])
    else:
        proposal, why = propose(rows, record_key=it.get("event_key"))
        log += [f"Mapping proposal: {w}" for w in why]
        if proposal is None:
            return None, {}, log
    m, notes = repair(proposal, rows)
    ds, problems = apply(m, rows)
    if ds.events and not any(e.type for e in ds.events):
        name = config.get("event", "record")
        ds = type(ds)(ds.entities, [replace(e, type=name) for e in ds.events], ds.actions)
    log += [f"Mapping used: {json.dumps(m.as_dict())}"] + [f"Mapping check: {n}" for n in notes]
    log += [f"Loading: {p}" for p in problems[:12]] + ([f"Loading: and {len(problems) - 12} more"] if len(problems) > 12 else [])
    groups: dict = {}
    if it:
        if it.get("key"):
            ds, groups, more = attach_items(ds, lines, key=it["key"], event_key=it["event_key"], item=it["item"],
                                            qty=it.get("qty"), item_group=it.get("item_group"))
            log += [f"Line items: {n}" for n in more]
    return ds, groups, log


def run(config: dict, base: Path = Path(".")) -> str:
    ds, groups, log = build(config, base)
    noun, cur = config.get("noun", "records"), config.get("currency", "")
    if ds is None:
        return "\n".join(["# Vertiq report", "", "## Loading", ""] + [f"- {x}" for x in log] + ["", "No mapping could be proposed."]) + "\n"
    out = ["# Vertiq report", "", f"{len(ds.events)} {noun} from {len(ds.entities)} entities.", "", "## Loading", ""]
    out += [f"- {x}" for x in log]
    if not ds.events:
        return "\n".join(out + ["", "Nothing loaded; see the notes above."]) + "\n"
    span = max(1, (max(e.at for e in ds.events).date() - min(e.at for e in ds.events).date()).days + 1)
    days = config.get("summary_days") or max(1, span // 2)
    out += ["", "## What happened", "", narrate.summary(summary.summarise(ds, days=days), entities=config.get("entities", "entities"))]
    pat = patterns.find(ds, item_groups=groups or None, noun=noun, currency=cur)
    out += ["", "## Patterns", ""]
    if pat["status"] == "learning":
        out.append(f"Not enough {noun} to test patterns: {pat['records']}, {pat['need']} needed.")
    else:
        out.append(f"{pat['features']} features found in the data; {pat['tested']} comparisons tested; "
                   f"{len(pat['found'])} hold after false-discovery control at {pat['q']:.0%} and are large enough to matter.")
        out += [f"- {f['text']}" for f in pat["found"]]
    lines = [(e.entity, f"{e.entity}|{e.at.isoformat()}|{i}", e.at, name) for i, e in enumerate(ds.events) for name, _q in (e.items or ())]
    if lines:
        a = affinity.analyse(lines)
        combos = a.get("combos", [])
        out += ["", "## Items bought together", ""]
        out += ([f"- {' + '.join(c['items'])}: on {c['bills_together']} {noun} together, {c['lift']} times as often as chance" for c in combos[:10]]
                or [f"No pair of items appears together on at least {affinity.MIN_PAIR} {noun}, the minimum for a combination."])
    types = sorted({e.type for e in ds.events if e.type})
    target = config.get("target") or (types[0] if len(types) == 1 else None)
    out += ["", "## Predictions", ""]
    if target:
        window = config.get("window_days", 30)
        out.append("- " + narrate.gate(predict_event(ds, target=target, window_days=window).report()))
        if any(e.value for e in ds.events):
            out.append("- " + narrate.gate(predict_value(ds, target=target, window_days=window).report()))
    else:
        out.append("- No single event type to predict; set \"target\" in the config.")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="A written report from exported tables (see the module docstring).")
    ap.add_argument("config")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = Path(a.config)
    text = run(json.loads(cfg.read_text()), cfg.parent)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
