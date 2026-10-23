"""One command from exported tables to a written report: everything Vertiq can say about the data, in its
own words, and what it cannot say yet.

  python -m vertiq.report config.json [--out report.md]

config.json names the tables and how they map to the contract; nothing else is configured:

  {"records": {"file": "orders.csv", "mapping": {"entity": "Customer", "time": "Date", "clock": "Time",
               "value": "Total", "attrs": ["Order No", "Channel"], "date_format": "%d/%m/%Y"}},
   "items":   {"file": "order_lines.csv", "key": "Order No", "event_key": "Order No", "item": "Item",
               "qty": "Qty", "item_group": "Category"},
   "entity_table": {"file": "customers.csv", "trainable": "trainable"},
   "noun": "orders", "currency": "₹", "entities": "customers"}

Consent: with "trainable" naming a column of the entity table, only entities whose value there is true (true, 1,
yes) are used to train or test models; an entity missing from the table is not. "train": false keeps every entity
out of training. Counts, the summary and patterns still use every record: they describe, they do not learn.

CSV is read with the standard library; .xlsx needs openpyxl (pip install openpyxl). Every mapping is checked
against the rows by vertiq.mapping.repair before use, and every section below comes from a Vertiq module:
summary and narrate (what happened), patterns (tested patterns), affinity (items bought together) and the
tasks behind the evaluation gate (predictions, or why there are none yet).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from . import affinity, narrate, patterns, summary
from .mapping import Mapping, apply, attach_items, repair
from .tasks import predict_event, predict_value


def _grid(path: Path) -> list[list]:
    """All cells of the first sheet with data, as rows of values. The format is read from the file's first bytes,
    not its name, since exports are often misnamed (an .xls that is really .xlsx): a zip is Excel 2007+
    (needs openpyxl), an OLE container is Excel 97-2003 (needs xlrd), anything else is read as CSV."""
    head = path.open("rb").read(8)
    if head.startswith(b"PK"):
        try:
            from openpyxl import load_workbook
        except ImportError as e:
            raise SystemExit("reading Excel files needs openpyxl: pip install openpyxl (or save the sheet as CSV)") from e
        import io
        wb = load_workbook(io.BytesIO(path.read_bytes()), read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            if any(any(v not in (None, "") for v in r) for r in rows):
                return rows
        return []
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        try:
            import xlrd
        except ImportError as e:
            raise SystemExit("reading Excel 97-2003 files needs xlrd: pip install xlrd (or save the sheet as CSV)") from e
        book = xlrd.open_workbook(str(path))
        for sh in book.sheets():
            if sh.nrows:
                return [sh.row_values(i) for i in range(sh.nrows)]
        return []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        return [row for row in csv.reader(f, dialect)]


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S") if (getattr(v, "hour", 0) or getattr(v, "minute", 0)) else v.strftime("%Y-%m-%d")
    return str(v)


def read_table(path: str | Path) -> list[dict]:
    """A table as {column: text} rows. The header is the first row, among the first 20, that fills at least 80% of
    the widest row with text: title lines and blank rows above it are skipped. Repeated or empty header names
    are made unique ("column 7", "Total (2)")."""
    rows = [[_cell(v) for v in r] for r in _grid(Path(path))]
    if not rows:
        return []
    width = max(sum(1 for v in r if v.strip()) for r in rows[:20])
    at = next(i for i, r in enumerate(rows[:20]) if sum(1 for v in r if v.strip()) >= 0.8 * width
              and all(not v.strip() or not v.replace(".", "").replace("-", "").isdigit() for v in r))
    head, seen = [], Counter()
    for i, h in enumerate(rows[at]):
        name = h.strip() or f"column {i + 1}"
        seen[name] += 1
        head.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return [dict(zip(head, r + [""] * (len(head) - len(r)))) for r in rows[at + 1:]]


def build(config: dict, base: Path = Path(".")) -> tuple[object, dict, list[str]]:
    """The dataset, item groups and loading notes from a config. Without a mapping, Vertiq proposes one from
    the values (vertiq.tables.propose); without join columns, it finds them (vertiq.tables.find_join)."""
    from dataclasses import replace
    from .tables import find_join, find_lines, propose, records_from_lines
    rec = config["records"]
    rows = read_table(base / rec["file"])
    log = []
    it = dict(config.get("items") or {})
    lines = read_table(base / it["file"]) if it else []
    if not it and not rec.get("mapping"):
        spec, why = find_lines(rows)                 # a single table of line items: build the records from it
        if spec:
            log += [f"Table shape: {w}" for w in why]
            lines = rows
            rows = records_from_lines(rows, spec)
            it = {"key": spec["key"], "event_key": spec["key"], "item": spec["item"], "qty": spec.get("qty"),
                  "item_group": spec.get("item_group")}
    if it and not it.get("key"):
        found, why = find_join(rows, lines)          # first: the join key names the record, not whom it is about
        log += [f"Line items: {w}" for w in why]
        if found:
            it.update(found)
    from .tables import entity_table
    shape = entity_table(rows) if not it else None
    if shape and shape["counts"]:
        log.append(f"Table shape: one row per entity (key {shape['key']!r}) with totals {shape['counts']}; each row is "
                   "read as the entity's latest activity, and its totals describe, never predict")
    if rec.get("mapping"):
        proposal = Mapping.parse(rec["mapping"])
    else:
        proposal, why = propose(rows, record_key=it.get("event_key"), latest=bool(shape and shape["counts"]))
        log += [f"Mapping proposal: {w}" for w in why]
        if proposal is None:
            return None, {}, log
    m, notes = repair(proposal, rows, exclude={it["event_key"]} if it.get("event_key") else ())
    ds, problems = apply(m, rows)
    if ds.events and not any(e.type for e in ds.events):
        name = config.get("event", "record")
        ds = type(ds)(ds.entities, [replace(e, type=name) for e in ds.events], ds.actions)
    log += [f"Mapping used: {json.dumps(m.as_dict())}"] + [f"Mapping check: {n}" for n in notes]
    log += [f"Loading: {p}" for p in problems[:12]] + ([f"Loading: and {len(problems) - 12} more"] if len(problems) > 12 else [])
    groups: dict = {}
    if config.get("entity_table"):                    # a customer list, asset register or policy master
        from .tables import attach_entities
        et = config["entity_table"]
        ds, more, _labels = attach_entities(ds, read_table(base / et["file"]), et.get("key"))
        log += [f"Entities: {n}" for n in more]
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
    allowed, why = trainable(config, ds)
    if allowed is not None:
        from .contract import Dataset
        if not allowed:
            out.append(f"- No model was trained: none of the {len(ds.entities):,} entities may be used for training ({why}). "
                       "The sections above describe the data; they do not train on it.")
            return "\n".join(out) + "\n"
        out.append(f"- Models use only the {len(allowed):,} of {len(ds.entities):,} entities that may be used for training ({why}).")
        ds = Dataset([e for e in ds.entities if e.id in allowed], [e for e in ds.events if e.entity in allowed], ds.actions)
    if target:
        window = config.get("window_days", 30)
        out.append("- " + narrate.gate(predict_event(ds, target=target, window_days=window).report()))
        if any(e.value for e in ds.events):
            out.append("- " + narrate.gate(predict_value(ds, target=target, window_days=window).report()))
    else:
        out.append("- No single event type to predict; set \"target\" in the config.")
    return "\n".join(out) + "\n"


def trainable(config: dict, ds) -> tuple[set | None, str]:
    """The entities that may be used to train models, and why; (None, "") when the config says nothing."""
    if config.get("train") is False:
        return set(), '"train": false in the config'
    col = (config.get("entity_table") or {}).get("trainable")
    if not col:
        return None, ""
    yes = {e.id for e in ds.entities if str((e.attrs or {}).get(col, "")).strip().lower() in ("true", "1", "yes", "y")}
    return yes, f"column {col!r} of the entity table"


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
