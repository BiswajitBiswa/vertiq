"""Exports from real systems carry totals rows, placeholder ids, empty profile columns and a second amount
column that is only partly filled. These tables are generated, not taken from any one source."""
import random

from vertiq.mapping import Mapping, apply, placeholder_ids, repair, structural_rows


def _orders(n=80, seed=3):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        qty = rnd.choice([1, 1, 1, 2, 3])
        price = rnd.choice([40, 70, 90, 120, 250])
        rows.append({"Order": f"O-{1000 + i}", "Customer Phone": rnd.choice(["-", f"9{rnd.randint(100000000, 999999999)}"]),
                     "Tax Id": "-", "Order Date": f"{rnd.randint(1, 28):02d}/0{rnd.randint(1, 9)}/2026", "Time": f"{rnd.randint(1, 12)}:{rnd.randint(10, 59)} PM",
                     "Store": "Main Street", "Channel": rnd.choice(["Online", "Counter"]), "Qty": str(qty),
                     "Unit Price": str(price if rnd.random() < 0.6 else 0), "Line Total": str(price * qty)})
    rows.append({"Order": "", "Customer Phone": "", "Tax Id": "", "Order Date": "", "Time": "", "Store": "", "Channel": "",
                 "Qty": "", "Unit Price": "", "Line Total": ""})
    rows.append({"Order": "Grand Total", "Customer Phone": "", "Tax Id": "", "Order Date": "", "Time": "", "Store": "", "Channel": "",
                 "Qty": "", "Unit Price": "999", "Line Total": "9999"})
    return rows


def test_structural_rows_are_named_and_skipped():
    rows = _orders()
    s = structural_rows(rows)
    assert set(s) == {81, 82} and "summary" in s[82]
    m, _ = repair(Mapping.parse({"entity": "Order", "time": "Order Date", "value": "Line Total", "date_format": "%d/%m/%Y"}), rows)
    ds, problems = apply(m, rows)
    assert len(ds.events) == 80 and any("summary row" in p for p in problems)


def test_placeholder_ids_never_merge_into_one_entity():
    rows = _orders()
    assert "-" in placeholder_ids(rows, "Customer Phone")
    m = Mapping.parse({"entity": "Customer Phone", "time": "Order Date", "value": "Line Total", "date_format": "%d/%m/%Y"})
    ds, problems = apply(m, rows)
    assert "-" not in {e.id for e in ds.entities}
    assert any("placeholder entity" in p for p in problems)


def test_unfit_columns_are_replaced_by_the_clearly_best_one():
    rows = _orders()
    m, notes = repair(Mapping.parse({"entity": "Tax Id", "time": "Order Date", "value": "Unit Price", "group": "Store",
                                     "date_format": "%d/%m/%Y"}), rows)
    assert m.entity != "Tax Id" and m.entity in ("Order", "Customer Phone")
    assert m.value == "Line Total"                    # the unit price is 0 where the line total records the sale
    assert m.group in (None, "Channel")               # one store everywhere is no group; a replacement is text, not "Qty"
    assert any("placeholder" in n for n in notes) and any("Line Total" in n for n in notes)


def test_small_tables_are_left_alone():
    rows = [{"security": "INFY", "trade date": "02/06/2025", "qty": "51"}, {"security": "INFY", "trade date": "24/05/2025", "qty": "444"},
            {"security": "INFY", "trade date": "21/05/2025", "qty": "303"}]
    m = Mapping.parse({"entity": "security", "time": "trade date", "value": "qty", "date_format": "%d/%m/%Y"})
    assert repair(m, rows)[0].as_dict() == m.as_dict()
