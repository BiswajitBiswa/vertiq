"""The Online Retail II preparation: cancellations and lines without a customer are dropped, lines are
aggregated to one order per invoice with value = quantity x price."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "public"))
from online_retail import to_dataset  # noqa: E402


def test_lines_become_orders():
    lines = [
        {"Invoice": "489434", "StockCode": "85048", "Quantity": "12", "InvoiceDate": "2009-12-01 07:45:00", "Price": "6.95", "Customer ID": "13085.0", "Country": "United Kingdom"},
        {"Invoice": "489434", "StockCode": "79323P", "Quantity": "12", "InvoiceDate": "2009-12-01 07:45:00", "Price": "6.75", "Customer ID": "13085.0", "Country": "United Kingdom"},
        {"Invoice": "C489449", "StockCode": "22087", "Quantity": "-12", "InvoiceDate": "2009-12-01 10:33:00", "Price": "2.95", "Customer ID": "16321.0", "Country": "Australia"},
        {"Invoice": "489436", "StockCode": "48173C", "Quantity": "10", "InvoiceDate": "2009-12-01 09:06:00", "Price": "5.95", "Customer ID": "", "Country": "United Kingdom"},
    ]
    ds, counts = to_dataset(lines)
    assert counts["orders"] == 1 and counts["dropped"] == {"cancelled": 1, "no_customer": 1, "bad_date": 0}
    e = ds.events[0]
    assert e.entity == "13085" and e.value == 164.4 and len(e.items) == 2 and ds.groups() == {"13085": "United Kingdom"}
