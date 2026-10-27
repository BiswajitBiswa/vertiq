"""Training data for Vertiq LM's first job: map an unfamiliar table to Vertiq's contract.

Every example is a table header plus three sample rows, and the correct mapping as JSON. Because the
tables are generated, every answer is known to be right, and `vertiq.mapping.apply` can check any
model's answer against the rows.

No bias by design:
  * every domain gets the same number of examples
  * column names mix English, Spanish, German, French and Hindi (romanised), plus codes and
    abbreviations, in every naming style (snake_case, camelCase, Title Case, UPPER)
  * every supported date format appears equally, and day-first vs month-first is always decidable
    from the rows
  * a random draw (fixed seed, written to meta.json) picks the domains held out entirely, so the
    test isn't hand-picked; results are reported per domain

Run: lm/.venv/bin/python lm/make_data.py [--out lm/data/v2] -> {train,valid,test}.jsonl, meta.json
(lm/data/*.jsonl at the top level are v1, kept so earlier results stay comparable)
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import random
import re
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from vertiq.mapping import DATE_FORMATS, _id_like, _int_ids, date_formats, profile  # noqa: E402

SEED = 20261009
PER_DOMAIN_TRAIN, PER_DOMAIN_VALID, PER_DOMAIN_TEST = 90, 10, 12
UNSEEN = 4                      # domains held out completely
FMTS = [f for f in DATE_FORMATS if "%H" not in f]

PROMPT = ("Map this table to Vertiq's contract. Columns, then 3 sample rows (CSV):\n{csv}\n\n"
          "Reply with JSON only, no explanation: {{\"entity\": column of who/what each row is about, "
          "\"time\": column of when it happened, \"type\": column naming what kind of event (or null), "
          "\"value\": numeric column of the amount or measure (or null), \"group\": column of the site, "
          "branch or unit the entity belongs to (or null), \"attrs\": [other descriptive columns worth keeping; "
          "leave out row numbers, import or audit columns], \"date_format\": strftime format of the time column}}")

# v2 adds what each column holds, read from its values by code, so the model chooses among columns
# knowing which hold dates, amounts or identifiers instead of inferring it from three rows
PROMPT_V2 = PROMPT.replace("\n\nReply with JSON", "\n\nColumn kinds, read from the values: {kinds}\n\nReply with JSON")

# v3 writes one line per column (its name once, its kind, a few distinct values cut short) instead of the header, the
# rows and the kinds, so wide exports fit the training length: 82% of the messy v3 tables fit 640 tokens, against 54%
# (88% with two values per text column, which lost the model more attributes than the extra tables gained)
PROMPT_V3 = ("Map this table to Vertiq's contract. One line per column: its name, what its values hold (read by code), "
             "then sample values:\n{lines}" + PROMPT.split("{csv}", 1)[1])
CELL = 14                       # characters of a sample value shown
TEXT_VALUES = 3                 # distinct values shown for a text or id column (lm/data/v4 used --text-values 2)
REPEATS = False                 # how often a shown value repeats, as "BLR-01 ×2" (--repeats; did worse, lm/README.md)


def lines(rows: list[dict]) -> str:
    """'Cust No: id | C1043 | C2210' per column; 'remarks: empty' when no row has a value."""
    k = dict(zip(rows[0], (p.split(": ", 1)[1] for p in kinds(rows).split("; ")))) if rows else {}
    out = []
    for c in (rows[0] if rows else {}):
        seen = Counter(str(r[c]).strip() for r in rows if str(r[c]).strip())
        if not seen:
            out.append(f"{c}: empty")
            continue
        n = 1 if k[c].startswith(("number", "date", "row")) else TEXT_VALUES   # the kind already says what a number or date is
        # a repeat is kept as a count ("BLR-01 ×2"): repeating values mark a site or a kind, unique ones an identifier
        out.append(f"{c}: {k[c]} | " + " | ".join((v if len(v) <= CELL else v[:CELL - 1] + "…") + (f" ×{m}" if m > 1 and REPEATS else "")
                                                   for v, m in list(seen.items())[:n]))
    return "\n".join(out)


def kinds(rows: list[dict]) -> str:
    """'Cust No: id; Buy Date: date %d/%m/%Y; Amt: number; Sl: row counter; Mode: text'"""
    k, out = profile(rows), []
    for c in (rows[0] if rows else {}):
        if k[c] == "date":
            fits = date_formats([r[c] for r in rows])
            out.append(f"{c}: date {fits[0]}" if len(fits) == 1 else f"{c}: date ({' or '.join(fits)})")
        elif k[c] == "rownum":
            out.append(f"{c}: row counter")
        elif k[c] == "number":
            # long whole numbers of one length (459201, 9876543210) may be ids; say so rather than guess
            vals = [str(r[c]).strip() for r in rows]
            long_ids = _int_ids(rows, c) and len({len(v) for v in vals}) == 1 and len(vals[0]) >= 5
            out.append(f"{c}: number or id" if long_ids else f"{c}: number")
        elif _id_like(rows, c):
            out.append(f"{c}: id")
        else:
            out.append(f"{c}: text")
    return "; ".join(out)


def render(rows: list[dict], version: int = 2) -> str:
    """The prompt for a table. v1 is the original (plain comma join), kept so old models are scored
    on new tables exactly as they were trained; v2 quotes the CSV properly and adds column kinds; v3 is one
    line per column."""
    cols = list(rows[0])
    if version == 3:
        return PROMPT_V3.format(lines=lines(rows))
    if version == 1:
        return PROMPT.format(csv="\n".join([",".join(cols)] + [",".join(str(rw[c]) for c in cols) for rw in rows]))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows([[rw[c] for c in cols] for rw in rows])
    return PROMPT_V2.format(csv=buf.getvalue().rstrip("\n"), kinds=kinds(rows))


# amounts as real systems write them; one style per table
AMOUNT_STYLES = [lambda x: f"{x}", lambda x: f"{x:,}", lambda x: f"₹{x:,}", lambda x: f"$ {x:,}",
                 lambda x: "R$ " + f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."),
                 lambda x: "€ " + f"{x:,.2f}".replace(",", " ").replace(".", ","), lambda x: f"{x} EUR"]


# role: [synonyms...]; each domain also says how to fake the values
D = {
 "retail_sales": dict(entity=["customer id", "cust no", "cliente id", "kunden nr", "client ref", "grahak id"],
    time=["bill date", "invoice date", "fecha venta", "datum", "date achat", "kharid tarikh"],
    type=["channel", "payment mode", "canal", "zahlart"], value=["bill amount", "total", "importe", "betrag", "montant", "rakam"],
    group=["store", "outlet code", "tienda", "filiale", "magasin", "dukaan"],
    attrs=[["item category", "categoria"], ["cashier", "cajero"], ["coupon code", "gutschein"]],
    ids=lambda r: f"C{r.randint(1000, 99999)}", types=["UPI", "card", "cash", "wallet"], val=lambda r: round(r.uniform(80, 4000), 2),
    groups=["BLR-01", "DEL-04", "PUN-02"]),
 "inventory": dict(entity=["sku", "item code", "material", "articulo", "artikel nr", "vastu kod"],
    time=["movement date", "posting date", "fecha mov", "buchungsdatum", "date mouvement"],
    type=["movement type", "txn type", "tipo mov", "bewegungsart"], value=["quantity", "qty", "cantidad", "menge", "quantite", "matra"],
    group=["warehouse", "plant", "almacen", "lager", "entrepot", "godaam"],
    attrs=[["batch", "lote"], ["unit", "einheit"], ["supplier", "proveedor"]],
    ids=lambda r: f"SKU-{r.randint(100, 9999)}", types=["GR", "GI", "transfer", "return"], val=lambda r: r.randint(1, 500),
    groups=["WH1", "WH2", "PL10"]),
 "manufacturing": dict(entity=["equipment", "asset id", "machine no", "equipo", "anlage", "machine", "yantra id"],
    time=["created on", "order date", "fecha orden", "erstellt am", "date creation"],
    type=["failure code", "order type", "codigo falla", "schadenscode", "type panne"], value=["total cost", "repair cost", "costo total", "kosten", "cout"],
    group=["plant", "site", "planta", "werk", "usine"],
    attrs=[["contractor", "contratista"], ["root cause", "ursache"], ["priority", "prioridad"]],
    ids=lambda r: f"EQ-{r.randint(100, 999)}", types=["SEAL", "BEARING", "MOTOR", "PM"], val=lambda r: round(r.uniform(50, 9000), 2),
    groups=["1000", "2000", "3000"]),
 "healthcare": dict(entity=["patient id", "mrn", "uhid", "paciente", "patienten nr", "rogi id"],
    time=["admit date", "encounter date", "fecha ingreso", "aufnahmedatum", "date admission"],
    type=["ward", "encounter type", "servicio", "station", "service"], value=["length of stay", "los days", "charges", "estancia", "verweildauer"],
    group=["hospital", "facility", "centro", "klinik", "etablissement"],
    attrs=[["diagnosis code", "diagnostico"], ["age band", "edad"], ["gender", "sexo"]],
    ids=lambda r: f"P{r.randint(10000, 99999)}", types=["ICU", "general", "maternity", "OPD"], val=lambda r: r.randint(1, 21),
    groups=["H-North", "H-South"]),
 "portfolio": dict(entity=["ticker", "isin", "security", "valor", "wertpapier", "titre"],
    time=["trade date", "settle date", "fecha operacion", "handelstag", "date operation"],
    type=["side", "trade type", "tipo", "richtung", "sens"], value=["notional", "amount", "monto", "betrag", "montant"],
    group=["portfolio", "fund", "cartera", "depot", "fonds"],
    attrs=[["broker", "corredor"], ["currency", "moneda"], ["strategy", "estrategia"]],
    ids=lambda r: r.choice(["INFY", "TCS", "HDFCBANK", "RELIANCE", "AAPL", "MSFT"]), types=["BUY", "SELL"], val=lambda r: round(r.uniform(1000, 250000), 2),
    groups=["Growth-A", "Income-B"]),
 "tax": dict(entity=["gstin", "taxpayer id", "pan", "nif", "steuernummer", "numero fiscal"],
    time=["filing date", "return period", "fecha presentacion", "abgabedatum", "date depot"],
    type=["return type", "form", "modelo", "formular", "formulaire"], value=["tax paid", "liability", "cuota", "steuerbetrag", "impot"],
    group=["state", "jurisdiction", "estado", "bundesland", "region"],
    attrs=[["status", "estado tramite"], ["filed by", "presentado por"], ["notice flag", "aviso"]],
    ids=lambda r: f"29ABCDE{r.randint(1000, 9999)}F1Z{r.randint(1, 9)}", types=["GSTR-1", "GSTR-3B", "ITR-4"], val=lambda r: round(r.uniform(0, 90000), 2),
    groups=["KA", "MH", "DL"]),
 "b2b_invoices": dict(entity=["account", "customer account", "debtor", "cuenta", "debitor", "compte client"],
    time=["invoice date", "doc date", "fecha factura", "rechnungsdatum", "date facture"],
    type=["doc type", "invoice type", "tipo doc", "belegart"], value=["invoice amount", "net value", "importe neto", "nettobetrag", "montant ht"],
    group=["sales org", "region", "org ventas", "verkaufsorg"],
    attrs=[["payment terms", "condiciones"], ["sales rep", "vendedor"], ["currency", "moneda"]],
    ids=lambda r: f"ACC{r.randint(100, 9999)}", types=["INV", "CRN", "DBN"], val=lambda r: round(r.uniform(500, 500000), 2),
    groups=["NORTH", "WEST"]),
 "education": dict(entity=["student id", "roll no", "alumno", "schueler nr", "eleve", "chhatra id"],
    time=["class date", "attendance date", "fecha clase", "datum unterricht", "date cours"],
    type=["subject", "session type", "asignatura", "fach", "matiere"], value=["minutes present", "score", "minutos", "punkte", "note"],
    group=["school", "campus", "colegio", "schule", "ecole"],
    attrs=[["grade", "curso"], ["teacher", "profesor"], ["section", "seccion"]],
    ids=lambda r: f"S{r.randint(100, 9999)}", types=["math", "science", "language"], val=lambda r: r.randint(0, 60),
    groups=["Campus-1", "Campus-2"]),
 "logistics": dict(entity=["shipment id", "awb", "consignment", "envio", "sendung nr", "expedition"],
    time=["dispatch date", "pickup date", "fecha envio", "versanddatum", "date expedition"],
    type=["status", "service level", "estado", "zustand"], value=["weight kg", "freight", "peso", "gewicht", "poids"],
    group=["hub", "depot", "centro", "lager", "plateforme"],
    attrs=[["carrier", "transportista"], ["destination", "destino"], ["mode", "modo"]],
    ids=lambda r: f"AWB{r.randint(100000, 999999)}", types=["picked", "in transit", "delivered"], val=lambda r: round(r.uniform(0.2, 900), 1),
    groups=["HUB-MUM", "HUB-BLR"]),
 "hospitality": dict(entity=["guest id", "booking ref", "huesped", "gast nr", "client", "atithi id"],
    time=["check in", "stay date", "fecha entrada", "anreise", "date arrivee"],
    type=["room type", "rate plan", "tipo habitacion", "zimmerart"], value=["room revenue", "spend", "ingreso", "umsatz", "revenu"],
    group=["property", "hotel", "propiedad", "haus"],
    attrs=[["source", "origen"], ["nights", "noches"], ["nationality", "nacionalidad"]],
    ids=lambda r: f"G{r.randint(1000, 99999)}", types=["deluxe", "suite", "standard"], val=lambda r: round(r.uniform(1500, 40000), 2),
    groups=["GOA-1", "JAI-2"]),
 "telecom": dict(entity=["msisdn", "subscriber id", "linea", "teilnehmer", "abonne", "upbhokta id"],
    time=["usage date", "bill cycle", "fecha uso", "nutzungsdatum", "date usage"],
    type=["service", "usage type", "servicio", "dienst"], value=["data mb", "usage", "consumo", "verbrauch", "conso"],
    group=["circle", "region", "zona", "gebiet"],
    attrs=[["plan", "tarifa"], ["device", "dispositivo"], ["roaming", "itinerancia"]],
    ids=lambda r: f"9{r.randint(100000000, 999999999)}", types=["data", "voice", "sms"], val=lambda r: round(r.uniform(0, 8000), 1),
    groups=["KA", "TN"]),
 "insurance": dict(entity=["policy no", "policy id", "poliza", "police nr", "contrat"],
    time=["claim date", "loss date", "fecha siniestro", "schadendatum", "date sinistre"],
    type=["claim type", "cover", "tipo siniestro", "sparte"], value=["claim amount", "paid", "importe", "schadenhoehe", "montant"],
    group=["branch", "office", "sucursal", "filiale", "agence"],
    attrs=[["status", "estado"], ["surveyor", "perito"], ["channel", "canal"]],
    ids=lambda r: f"POL{r.randint(100000, 999999)}", types=["motor", "health", "fire"], val=lambda r: round(r.uniform(2000, 900000), 2),
    groups=["BR-01", "BR-07"]),
 "energy": dict(entity=["meter id", "consumer no", "medidor", "zaehler nr", "compteur", "meter sankhya"],
    time=["reading date", "read on", "fecha lectura", "ablesedatum", "date releve"],
    type=["tariff", "read type", "tarifa", "tarif"], value=["kwh", "units", "consumo", "verbrauch", "conso kwh"],
    group=["feeder", "substation", "subestacion", "umspannwerk", "poste"],
    attrs=[["phase", "fase"], ["sanctioned load", "carga"], ["category", "categoria"]],
    ids=lambda r: f"MTR{r.randint(10000, 99999)}", types=["domestic", "commercial", "industrial"], val=lambda r: round(r.uniform(10, 3000), 1),
    groups=["FDR-11", "FDR-22"]),
 "hr": dict(entity=["employee id", "emp code", "empleado", "personal nr", "matricule", "karmchari id"],
    time=["event date", "effective date", "fecha efecto", "gueltig ab", "date effet"],
    type=["event", "action type", "accion", "massnahme", "evenement"], value=["salary", "hours", "salario", "gehalt", "salaire"],
    group=["department", "cost center", "departamento", "abteilung", "service rh"],
    attrs=[["grade", "nivel"], ["manager", "jefe"], ["location", "ubicacion"]],
    ids=lambda r: f"E{r.randint(1000, 99999)}", types=["hire", "promotion", "leave", "exit"], val=lambda r: round(r.uniform(20000, 300000), 2),
    groups=["FIN", "OPS", "ENG"]),
 "agriculture": dict(entity=["farm id", "plot no", "parcela", "feld nr", "parcelle", "khet id"],
    time=["harvest date", "sowing date", "fecha cosecha", "erntedatum", "date recolte"],
    type=["crop", "season", "cultivo", "kultur", "culture"], value=["yield kg", "output", "rendimiento", "ertrag", "rendement"],
    group=["village", "district", "municipio", "gemeinde", "commune"],
    attrs=[["irrigation", "riego"], ["seed variety", "variedad"], ["soil type", "suelo"]],
    ids=lambda r: f"F{r.randint(100, 9999)}", types=["paddy", "wheat", "cotton"], val=lambda r: round(r.uniform(100, 9000), 1),
    groups=["Mandya", "Nashik"]),
}
DISTRACT = [["row no", "sl no", "n fila", "zeile"], ["import batch", "load id", "lote carga"], ["created by", "creado por", "erfasst von"],
            ["last modified", "geaendert am", "modifie le"]]
STYLES = ["snake", "camel", "pascal", "upper", "title", "lower_space"]


def styled(name: str, style: str) -> str:
    w = [x for x in re.split(r"[ _]+", name) if x]
    if style == "snake":
        return "_".join(x.lower() for x in w)
    if style == "camel":
        return w[0].lower() + "".join(x.capitalize() for x in w[1:])
    if style == "pascal":
        return "".join(x.capitalize() for x in w)
    if style == "upper":
        return "_".join(x.upper() for x in w)
    if style == "title":
        return " ".join(x.capitalize() for x in w)
    return " ".join(x.lower() for x in w)


def sample_dates(r: random.Random, fmt: str) -> list[str]:
    """Three dates; at least one with day > 12 so day-first and month-first can be told apart."""
    base = date(2025, 1, 1) + timedelta(days=r.randint(0, 600))
    ds = [base + timedelta(days=r.randint(0, 40)) for _ in range(3)]
    if all(d.day <= 12 for d in ds):
        ds[0] = ds[0].replace(day=r.randint(13, 28))
    return [d.strftime(fmt) for d in ds]


def example(dom: str, r: random.Random) -> tuple[str, dict]:
    spec = D[dom]
    style = r.choice(STYLES)
    used = set()

    def pick(pool):
        for _ in range(10):
            c = styled(r.choice(pool), style)
            if c not in used:
                used.add(c)
                return c
        c = styled(r.choice(pool), style) + "_2"
        used.add(c)
        return c
    truth = {"entity": pick(spec["entity"]), "time": pick(spec["time"])}
    for role in ("type", "value", "group"):
        truth[role] = pick(spec[role]) if r.random() < 0.75 else None
    attrs = r.sample(spec["attrs"], r.randint(0, len(spec["attrs"])))
    truth["attrs"] = [pick(a) for a in attrs]
    # audit columns hold what their names say, as real exports do: a row counter, one batch id per file,
    # who entered the row, when it was last touched
    noise = [(pick(DISTRACT[k]), k) for k in r.sample(range(len(DISTRACT)), r.randint(0, 2))]
    batch = r.choice(["BATCH7", "LD-2031", "IMP_0412", "B-88"])
    stamp = date(2026, 1, 1) + timedelta(days=r.randint(0, 60))
    money = r.choice(AMOUNT_STYLES) if r.random() < 0.4 else AMOUNT_STYLES[0]
    fmt = r.choice(FMTS)
    truth["date_format"] = fmt
    dates = sample_dates(r, fmt)
    rows = []
    for i in range(3):
        row = {truth["entity"]: spec["ids"](r), truth["time"]: dates[i]}
        if truth["type"]:
            row[truth["type"]] = r.choice(spec["types"])
        if truth["value"]:
            row[truth["value"]] = money(spec["val"](r))
        if truth["group"]:
            row[truth["group"]] = r.choice(spec["groups"])
        for c in truth["attrs"]:
            row[c] = r.choice(["A", "B", "C", "north", "yes", "no", "2", "standard", "high", "low"])
        for c, k in noise:
            row[c] = [str(i + 1), batch, r.choice(["admin", "sys", "asha.k", "j.muller", "etl_user"]),
                      f"{stamp + timedelta(days=i)}T{r.randint(7, 19):02d}:{r.randint(0, 59):02d}:00"][k]
        rows.append(row)
    cols = list(rows[0])
    r.shuffle(cols)
    table = [{c: str(rw[c]) for c in cols} for rw in rows]
    return render(table), {**truth, "_rows": table}


# v3: what real exports carry besides the record itself. Wide tables, several amount columns, a second date,
# free text, empty and placeholder columns, ISO timestamps. None of these columns belongs in the mapping.
MESS = {
    "address": [["street_number", "street_name", "city", "state", "zip"], ["address", "locality", "pincode"]],
    "coords": [["latitude", "longitude", "location"]],
    "contact": [["contact_1_name", "contact_1_type", "contact_1_city"], ["captain", "created_by"]],
    "fees": [["fee_paid", "fee_unpaid", "fee_waived", "subtotal_paid"], ["tax", "round_off", "charges"]],
    "text": [["description", "notes"], ["resolution_description"], ["violations", "comments"]],
    "empty": [["landmark", "email", "gstin", "descriptor_2", "anniversary"]],
    "placeholder": [["remarks", "external_id", "pax"]],
    "dates": [["closed_date", "due_date", "last_modified", "resolution_updated"]],
}
WORDS = "the of pothole sink handwashing repaired inspected closed pending water leak front store sign permit noise heat".split()


def add_mess(prompt: str, truth: dict, r: random.Random) -> tuple[str, dict]:
    """Distractor columns added to a table, and sometimes ISO timestamps on the time column; the mapping is
    unchanged, since none of the added columns plays a role."""
    rows = truth["_rows"]
    used = set(rows[0])
    for kind in r.sample(list(MESS), r.randint(3, len(MESS))):
        for name in r.choice(MESS[kind]):
            if name in used:
                continue
            used.add(name)
            for i, row in enumerate(rows):
                if kind == "address":
                    row[name] = r.choice([str(r.randint(10, 9999)), "MAIN ST", "CHICAGO", "IL", str(r.randint(10000, 99999))])
                elif kind == "coords":
                    row[name] = f"{r.uniform(-90, 90):.6f}"
                elif kind == "contact":
                    row[name] = r.choice(["", "OWNER", "ARCHITECT", "R. SHARMA", "J. LEE", "NAPERVILLE"])
                elif kind == "fees":
                    row[name] = r.choice(["0", "0", str(r.randint(10, 900))])
                elif kind == "text":
                    row[name] = " ".join(r.choice(WORDS) for _ in range(r.randint(4, 12))).upper()
                elif kind == "empty":
                    row[name] = ""
                elif kind == "placeholder":
                    row[name] = r.choice(["-", "N/A", "0"])
                else:
                    row[name] = r.choice(["", f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:00:00.000"])
    if r.random() < 0.3:                                       # ISO timestamps on the time column
        try:
            stamps = [datetime.strptime(rw[truth["time"]], truth["date_format"]) for rw in rows]
            for rw, t in zip(rows, stamps):
                rw[truth["time"]] = t.strftime("%Y-%m-%dT%H:%M:%S") + ".000"
            truth["date_format"] = "%Y-%m-%dT%H:%M:%S.%f"
        except (ValueError, KeyError):
            pass
    cols = list(rows[0])
    r.shuffle(cols)
    table = [{c: str(rw.get(c, "")) for c in cols} for rw in rows]
    truth["_rows"] = table
    return render(table), truth


def chat(prompt: str, truth: dict, dom: str) -> dict:
    ans = json.dumps({k: v for k, v in truth.items() if not k.startswith("_")}, ensure_ascii=False)
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}],
            "domain": dom, "rows": truth["_rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "v2"))
    ap.add_argument("--messy", type=float, default=0.0, help="share of tables given real-export clutter (v3)")
    ap.add_argument("--prompt", type=int, default=2, help="prompt version (3: one line per column)")
    ap.add_argument("--text-values", type=int, default=TEXT_VALUES, help="prompt v3: values shown per text column")
    ap.add_argument("--repeats", action="store_true", help="prompt v3: show how often a value repeats")
    a = ap.parse_args()
    globals().update(TEXT_VALUES=a.text_values, REPEATS=a.repeats)

    def make(dom):
        prompt, truth = example(dom, r)
        if a.messy and r.random() < a.messy:
            prompt, truth = add_mess(prompt, truth, r)
        if a.prompt != 2:                                     # same tables, written in another prompt version
            prompt = render(truth["_rows"], a.prompt)
        return chat(prompt, truth, dom)
    r = random.Random(SEED)
    domains = sorted(D)
    unseen = sorted(r.sample(domains, UNSEEN))
    seen = [d for d in domains if d not in unseen]
    out = {"train": [], "valid": [], "test": []}
    for dom in seen:
        for split, n in (("train", PER_DOMAIN_TRAIN), ("valid", PER_DOMAIN_VALID), ("test", PER_DOMAIN_TEST // 2)):
            for _ in range(n):
                out[split].append(make(dom) | {"seen": True})
    for dom in unseen:
        for _ in range(PER_DOMAIN_TEST):
            out["test"].append(make(dom) | {"seen": False})
    data = Path(a.out)
    data.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        r.shuffle(rows)
        with open(data / f"{split}.jsonl", "w") as f:
            for row in rows:
                keep = row if split == "test" else {"messages": row["messages"]}
                f.write(json.dumps(keep, ensure_ascii=False) + "\n")
    meta = {"seed": SEED, "seen_domains": seen, "unseen_domains": unseen, "counts": {k: len(v) for k, v in out.items()},
            "styles": STYLES, "date_formats": FMTS, "messy": a.messy, "prompt": a.prompt,
            **({"text_values": a.text_values, "repeats": a.repeats} if a.prompt == 3 else {})}
    (data / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
