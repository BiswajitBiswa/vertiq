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
