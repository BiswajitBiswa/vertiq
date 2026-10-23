"""End to end on generated exports: an orders table and an order-lines table, no mapping given."""
import csv
import json
import random

from vertiq import report
from vertiq.tables import find_join, propose

MENU = {"Margherita": "Pizza", "Farmhouse": "Pizza", "Cola": "Drinks", "Lemonade": "Drinks", "Fries": "Sides", "Salad": "Sides"}


def _write(tmp_path, n=300, seed=4, repeat=True):
    rnd = random.Random(seed)
    orders, lines = [], []
    for i in range(n):
        online = rnd.random() < 0.4
        items = [rnd.choice(["Margherita", "Farmhouse"])] if rnd.random() < 0.5 else []
        if rnd.random() < (0.1 if online else 0.6):
            items.append(rnd.choice(["Cola", "Lemonade"]))
        items.append(rnd.choice(["Fries", "Salad"]))
        no = f"ORD-{5000 + i}"
        total = 0
        for it in items:
            q = rnd.choice([1, 1, 2])
            price = 250 if MENU[it] == "Pizza" else 60
            lines.append({"Order No": no, "Store Group": rnd.choice(["Services", ""]), "Item": it, "Category": MENU[it], "Qty": q,
                          "Unit Price": price, "Amount": price * q})
            total += price * q
        phones = ["-", f"98{rnd.randint(10000000, 99999999)}"] + ([f"97{rnd.randint(100, 140):08d}"] if repeat else [])
        orders.append({"Order No": no, "Phone": rnd.choice(phones),
                       "Date": f"{rnd.randint(1, 28):02d}/{rnd.randint(1, 9):02d}/2026", "Time": f"{rnd.randint(1, 11)}:{rnd.randint(10, 59)} PM",
                       "Channel": "Online" if online else "Store", "Payment": rnd.choice(["Card", "Cash", "Wallet"]), "Subtotal": total,
                       "Total": total})
    orders.append({"Order No": "Total", "Phone": "", "Date": "", "Time": "", "Channel": "", "Payment": "",
                   "Subtotal": sum(o["Total"] for o in orders), "Total": sum(o["Total"] for o in orders)})
    for name, rows in (("orders.csv", orders), ("lines.csv", lines)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return orders, lines


def test_propose_and_join_from_values(tmp_path):
    orders, lines = _write(tmp_path)
    rows = report.read_table(tmp_path / "orders.csv")
    m, notes = propose(rows)
    assert m.entity == "Phone" and m.time == "Date" and m.clock == "Time" and m.value in ("Total", "Subtotal")
    assert "Channel" in m.attrs and "Order No" in m.attrs
    j, _ = find_join(rows, report.read_table(tmp_path / "lines.csv"))
    assert (j["event_key"], j["key"], j["item"], j["item_group"]) == ("Order No", "Order No", "Item", "Category")


def test_report_runs_from_file_names_only(tmp_path):
    _write(tmp_path)
    cfg = {"records": {"file": "orders.csv"}, "items": {"file": "lines.csv"}, "noun": "orders", "currency": "₹", "event": "order"}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    text = report.run(cfg, tmp_path)
    assert "## Patterns" in text and "include Drinks" in text and "where Channel is Online" in text
    assert "a summary row" in text and "placeholder entity" in text
    assert "## Predictions" in text


def test_the_join_key_names_the_record_not_the_customer(tmp_path):
    _write(tmp_path, n=80, repeat=False)              # nobody orders twice: phone and order number are both unique
    cfg = {"records": {"file": "orders.csv"}, "items": {"file": "lines.csv"}, "noun": "orders"}
    ds, groups, log = report.build(cfg, tmp_path)
    used = next(x for x in log if x.startswith("Mapping used"))
    assert '"entity": "Phone"' in used and "Order No" in used
    assert any("80 of 80 records got line items" in x for x in log) and groups


def test_a_single_line_item_table_becomes_records(tmp_path):
    rnd = random.Random(9)
    rows = []
    for i in range(120):
        inv, cust, when = f"{700000 + i}", f"{12000 + rnd.randint(0, 40)}", f"{rnd.randint(1, 28):02d}/{rnd.randint(1, 9):02d}/2026 {rnd.randint(9, 20)}:15"
        for _ in range(rnd.randint(1, 4)):
            item = rnd.choice(list(MENU))
            rows.append({"Invoice": inv, "Description": item + f" {rnd.randint(1, 4)}", "Quantity": rnd.choice([1, 2, 3, 6]),
                         "InvoiceDate": when, "Price": rnd.choice(["2.55", "4.95", "1.25", "3.75"]), "Customer ID": cust, "Country": rnd.choice(["UK", "France", "Spain"])})
    with open(tmp_path / "lines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ds, groups, log = report.build({"records": {"file": "lines.csv"}, "noun": "invoices"}, tmp_path)
    assert any("one row per item: records keyed by 'Invoice'" in x for x in log)
    assert len(ds.events) == 120 and all(e.items for e in ds.events)
    assert '"entity": "Customer ID"' in next(x for x in log if x.startswith("Mapping used"))


def test_repeating_names_are_the_entity_and_full_totals_rows_are_skipped(tmp_path):
    rnd = random.Random(2)
    names = [f"{rnd.choice(['ASHA', 'RAVI', 'MEERA', 'JOHN', 'LI', 'SARA'])} {rnd.choice(['K', 'M', 'P', 'S', 'T'])}{i}" for i in range(40)]
    rows = [{"Date": f"{rnd.randint(1, 28):02d}-{rnd.randint(1, 9):02d}-2026 12:{rnd.randint(10, 59)}:00", "Doc No": f"POS{9000 + i}",
             "Customer": rnd.choice(names), "Gross": f"{rnd.randint(5, 40) * 100}.00", "Net": f"{rnd.randint(4, 30) * 100}.00"} for i in range(200)]
    rows.append({"Date": "", "Doc No": "", "Customer": "Total", "Gross": "123456.00", "Net": "99999.00"})
    with open(tmp_path / "sales.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ds, _g, log = report.build({"records": {"file": "sales.csv"}}, tmp_path)
    assert '"entity": "Customer"' in next(x for x in log if x.startswith("Mapping used"))
    assert any("summary row" in x for x in log) and len(ds.events) == 200
    assert any(x.startswith("Mapping proposal: confirm: value") for x in log)


def _customers(tmp_path, n=300, seed=8):
    rnd = random.Random(seed)
    custs, orders = [], []
    for i in range(n):
        cid, seg = f"{40000 + i}", rnd.choice(["Gold", "Silver", "Basic"])
        k = rnd.randint(1, 6)
        first = rnd.randint(1, 200)
        spend = 0
        for j in range(k):
            day = first + j * rnd.randint(5, 30)
            amount = (900 if seg == "Gold" else 300) + rnd.randint(0, 100)
            spend += amount
            orders.append({"Order": f"OR{len(orders):05d}", "Customer": cid, "Date": f"2026-{1 + day // 30 % 12:02d}-{1 + day % 28:02d}", "Amount": amount})
        custs.append({"Customer Id": cid, "Segment": seg, "Orders": k, "Spend": spend, "First Order": f"2026-01-{1 + first % 28:02d}",
                      "Last Order": f"2026-09-{1 + (first + k) % 28:02d}"})
    for name, rows in (("customers.csv", custs), ("orders.csv", orders)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_entity_table_labels_join_events_and_show_in_patterns(tmp_path):
    _customers(tmp_path)
    cfg = {"records": {"file": "orders.csv"}, "entity_table": {"file": "customers.csv"}, "noun": "orders", "entities": "customers"}
    text = report.run(cfg, tmp_path)
    assert "entity table joined on 'Customer Id'" in text
    assert "where Segment is Gold average" in text or "where Segment is Basic average" in text


def test_a_customer_list_alone_is_read_as_one_row_per_entity(tmp_path):
    _customers(tmp_path)
    ds, _g, log = report.build({"records": {"file": "customers.csv"}}, tmp_path)
    assert any("one row per entity" in x for x in log)
    assert '"time": "Last Order"' in next(x for x in log if x.startswith("Mapping used"))


def test_a_product_code_beside_the_product_name_is_not_an_item_group():
    """Online-store lines carry a stock code for each product name. The code is the item again, not a group,
    so patterns and combos speak in product names."""
    from vertiq.tables import find_lines
    rnd = random.Random(4)
    products = [(f"{20000 + i}", f"PRODUCT NUMBER {i}") for i in range(60)]
    rows = []
    for inv in range(300):
        at = f"2010-03-{1 + inv % 28:02d} 1{inv % 10}:00:00"
        for code, name in rnd.sample(products, 3):
            rows.append({"Invoice": str(489000 + inv), "StockCode": code, "Description": name,
                         "Quantity": str(rnd.randint(1, 12)), "InvoiceDate": at, "Price": f"{rnd.uniform(1, 9):.2f}",
                         "Customer ID": str(rnd.randint(12000, 12400))})
    spec, _ = find_lines(rows)
    assert spec["key"] == "Invoice" and spec["item"] in ("Description", "StockCode")
    assert spec["item_group"] is None


def _with_consent(tmp_path, allow):
    """The customers table with a trainable column: True for customers `allow` picks."""
    _customers(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "customers.csv")))
    for r in rows:
        r["trainable"] = "True" if allow(r["Customer Id"]) else "False"
    with open(tmp_path / "customers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_no_model_trains_on_entities_that_may_not_be_used(tmp_path):
    _with_consent(tmp_path, lambda cid: False)
    cfg = {"records": {"file": "orders.csv"}, "entity_table": {"file": "customers.csv", "trainable": "trainable"}, "noun": "orders"}
    text = report.run(cfg, tmp_path)
    assert "No model was trained: none of the" in text and "column 'trainable'" in text
    assert "was promoted" not in text and "## Patterns" in text            # description still runs
    assert "No model was trained" in report.run({"records": {"file": "orders.csv"}, "train": False}, tmp_path)


def test_models_use_only_trainable_entities(tmp_path, monkeypatch):
    _with_consent(tmp_path, lambda cid: int(cid[-1]) % 2 == 0 if cid[-1].isdigit() else False)
    seen = []
    real = report.predict_event

    def spy(ds, **kw):
        seen.append({e.entity for e in ds.events})
        return real(ds, **kw)
    monkeypatch.setattr(report, "predict_event", spy)
    cfg = {"records": {"file": "orders.csv"}, "entity_table": {"file": "customers.csv", "trainable": "trainable"}, "noun": "orders"}
    text = report.run(cfg, tmp_path)
    assert "Models use only the" in text
    assert seen and all(int(x[-1]) % 2 == 0 for x in seen[0])
