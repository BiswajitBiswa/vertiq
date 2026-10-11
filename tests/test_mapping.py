"""A mapping (written by a person or by Vertiq LM) is checked against the real rows, never trusted."""
import pytest

from vertiq.mapping import Mapping, apply, score

ROWS = [{"Patient No": "P1", "Admit Dt": "14/02/2026", "Ward": "ICU", "LOS": "3", "Hospital": "H1", "Dx": "J18", "row_no": "1"},
        {"Patient No": "P2", "Admit Dt": "02/03/2026", "Ward": "OPD", "LOS": "1", "Hospital": "H2", "Dx": "E11", "row_no": "2"}]
GOOD = {"entity": "Patient No", "time": "Admit Dt", "type": "Ward", "value": "LOS", "group": "Hospital",
        "attrs": ["Dx"], "date_format": "%d/%m/%Y"}


def test_a_correct_mapping_loads_every_row():
    ds, problems = apply(Mapping.parse(GOOD), ROWS)
    assert problems == [] and len(ds.events) == 2 and ds.groups() == {"P1": "H1", "P2": "H2"}
    assert ds.events[0].value == 3.0 and ds.events[0].attrs == {"Dx": "J18"}


def test_the_data_catches_a_wrong_mapping():
    wrong = Mapping.parse({**GOOD, "date_format": "%m/%d/%Y"})          # 14/02 can't be month-first
    _ds, problems = apply(wrong, ROWS)
    assert any("doesn't match" in p for p in problems)
    _ds, problems = apply(Mapping.parse({**GOOD, "entity": "Patient"}), ROWS)
    assert problems == ["column 'Patient' (for entity) isn't in the table"]


def test_model_output_is_parsed_and_scored():
    text = 'Sure: {"entity": "Patient No", "time": "Admit Dt", "type": null, "value": "LOS", "group": "Hospital", "attrs": [], "date_format": "%d/%m/%Y"}'
    pred = Mapping.parse(text)
    s = score(pred, Mapping.parse(GOOD))
    assert s["roles"]["type"] is False and s["roles_correct"] == 0.8 and not s["exact"]
    with pytest.raises(ValueError):
        Mapping.parse("no json here")


def test_snap_repairs_near_miss_names_and_never_invents_columns():
    from vertiq.mapping import snap
    cols = list(ROWS[0])
    bad = Mapping.parse({**GOOD, "entity": "Patient Nr", "type": "Wardd", "value": "Admit Dt", "attrs": ["Dxx", "Nonsense col"]})
    fixed, changes = snap(bad, cols)
    assert fixed.entity == "Patient No" and fixed.type == "Ward" and fixed.value is None     # value repeated time: dropped
    assert fixed.attrs == ["Dx"] and any("isn't a column" in c for c in changes)
    _ds, problems = apply(fixed, ROWS)
    assert problems == []


def test_the_rows_decide_the_date_format_and_ambiguity_is_flagged():
    from vertiq.mapping import fix_dates
    guessed = Mapping.parse({**GOOD, "date_format": "%m/%d/%Y"})       # 14/02/2026 proves day-first
    fixed, note = fix_dates(guessed, ROWS)
    assert fixed.date_format == "%d/%m/%Y" and "only one that fits" in note
    assert fix_dates(Mapping.parse(GOOD), ROWS) == (Mapping.parse(GOOD), None)
    early = [{**r, "Admit Dt": d} for r, d in zip(ROWS, ("03/04/2026", "05/06/2026"))]   # no day above 12
    kept, note = fix_dates(guessed, early)
    assert kept.date_format == "%m/%d/%Y" and note.startswith("ambiguous") and "%d/%m/%Y" in note


def test_repair_holds_roles_to_what_their_values_can_be():
    from vertiq.mapping import profile, repair
    rows = [{"Cust No": "C18077", "Buy Date": "04/10/2025", "Shop": "PUN-02", "Amt": "1,267.81", "Sl": "1", "Mode": "cash"},
            {"Cust No": "C20011", "Buy Date": "17/10/2025", "Shop": "BLR-01", "Amt": "90", "Sl": "2", "Mode": "upi"},
            {"Cust No": "C31555", "Buy Date": "21/10/2025", "Shop": "PUN-02", "Amt": "455", "Sl": "3", "Mode": "card"}]
    assert profile(rows) == {"Cust No": "text", "Buy Date": "date", "Shop": "text", "Amt": "number", "Sl": "rownum", "Mode": "text"}
    model = Mapping.parse({"entity": "Buy Date", "time": "Mode", "value": "Shop", "attrs": ["Sl", "Mode"], "date_format": "%m/%d/%Y"})
    m, notes = repair(model, rows)
    # entity was a date, time had none, value was not a number, the row counter is not an attribute
    assert (m.entity, m.time, m.value, m.date_format, m.attrs) == ("Cust No", "Buy Date", None, "%d/%m/%Y", [])
    assert len(notes) == 5
    right = Mapping.parse({"entity": "Cust No", "time": "Buy Date", "type": "Mode", "value": "Amt", "group": "Shop",
                           "attrs": [], "date_format": "%d/%m/%Y"})
    assert repair(right, rows) == (right, [])                          # a correct mapping is never touched
    named = [{**r, "Cust No": n} for r, n in zip(rows, ("Asha", "Ravi", "Meera"))]
    named = [{**r, "Bill": b} for r, b in zip(named, ("INV-1", "INV-2", "INV-3"))]
    m, notes = repair(right, named)
    assert m.entity == "Cust No" and notes[0].startswith("confirm")     # a name is a fine entity: flagged, not moved


@pytest.mark.parametrize("values, want", [
    (["R$ 4.350", "R$ 12.400,00"], [4350, 12400]), (["Rp 1.250.000"], [1250000]), (["¥3,200,000"], [3200000]),
    (["€ 4.350,50"], [4350.5]), (["1,267.81", "90"], [1267.81, 90]), (["₹1,23,456"], [123456]), (["12,5"], [12.5]),
    (["-$45.10"], [-45.1]), (["USD 1,200"], [1200]), (["4 350,75 EUR"], [4350.75]), (["1'234.50"], [1234.5])])
def test_amounts_read_the_way_each_locale_writes_them(values, want):
    from vertiq.mapping import _num, decimal_mark
    dec = decimal_mark(values)
    assert [_num(v, dec) for v in values] == pytest.approx(want)


def test_identifiers_are_not_mistaken_for_amounts():
    from vertiq.mapping import _num
    assert [_num(v) for v in ("ACC4399", "SKU-1678", "EQ-602", "C18077", "cash")] == [None] * 5


def test_a_column_that_mixes_decimal_styles_reads_each_value_on_its_own():
    from vertiq.mapping import _num, decimal_mark
    vals = ["1250.50", "450,00", "12000"]
    assert decimal_mark(vals) == "mixed"
    assert [_num(v, "mixed") for v in vals] == [1250.5, 450.0, 12000.0]


def test_dates_with_times_are_known_formats():
    from vertiq.mapping import date_formats
    assert date_formats(["15/01/2023 09:45", "03/02/2023 18:10"]) == ["%d/%m/%Y %H:%M"]
