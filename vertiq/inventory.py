"""Smart inventory: what a business sells and has in stock, how fast each item sells
(from line items), and what to do about it.

  reorder       days of stock left < lead time + a 3-day buffer: reorder now, with a quantity
                that covers the lead time plus two weeks of sales
  slow_movers   in stock but more than SLOW_DAYS of cover (or nothing sold in 28 days): offer them
                to the people who'd like them instead of a blanket discount
  new_arrivals  added in the last NEW_DAYS: told to the people whose choices match
  out_of_stock  never promoted anywhere
  low_margin    under LOW_MARGIN: never discounted past what the margin allows

Items are matched to bills by name (case and spacing ignored). Stock left empty means "not
tracked": nothing is ever called out of stock without a number that says so.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import datetime

SALES_DAYS = 28
SLOW_DAYS = 60
NEW_DAYS = 14
LOW_MARGIN = 0.2
MARGIN_FLOOR = 0.1          # an offer must leave at least 10 points of margin on the item
DEFAULT_LEAD = 7
_ALIASES = {
    "name": ("name", "item", "itemname", "product", "productname", "skuname", "description", "menuitem"),
    "category": ("category", "group", "department", "itemgroup", "type"),
    "price": ("price", "mrp", "sellingprice", "rate", "sp", "saleprice"),
    "cost": ("cost", "costprice", "purchaseprice", "cp", "buyingprice"),
    "stock": ("stock", "qty", "quantity", "closingstock", "instock", "onhand", "available", "stockqty"),
    "lead_days": ("leadtime", "leaddays", "leadtimedays", "reorderdays"),
}


def key(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def num(v) -> float | None:
    s = str(v or "").replace(",", "").replace("₹", "").strip()
    try:
        return float(s) if s else None
    except ValueError:
        return None


def field_map(header: list[str]) -> dict[str, str]:
    norm = {"".join(ch for ch in h.lower() if ch.isalnum()): h for h in header}
    out = {}
    for field, aliases in _ALIASES.items():
        hit = next((norm[a] for a in aliases if a in norm), None)
        if hit:
            out[field] = hit
    return out


@dataclass
class Item:
    """One catalogue item. stock None = not tracked; lead_days None = DEFAULT_LEAD."""
    name: str
    category: str = ""
    price: float | None = None
    cost: float | None = None
    stock: float | None = None
    lead_days: int | None = None
    added_at: datetime | None = None

    @property
    def key(self) -> str:
        return key(self.name)

    @property
    def margin(self) -> float | None:
        return (self.price - self.cost) / self.price if self.price and self.cost is not None and self.price > 0 else None


def parse_table(text: str) -> tuple[list[str], list[dict]]:
    """A catalogue or stock export as CSV text -> (fields found, [item dicts]). Header names are
    matched loosely ("Item Name", "MRP", "Closing Stock"...). Rows without a name are skipped."""
    reader = csv.DictReader(io.StringIO(text))
    fields = field_map(reader.fieldnames or [])
    if "name" not in fields:
        raise ValueError("couldn't find an item name column (e.g. Item, Product, Name)")
    rows = []
    for line in reader:
        item = {f: line.get(col) for f, col in fields.items()}
        if str(item.get("name") or "").strip():
            rows.append(item)
    return sorted(fields), rows


def report(rows: list[Item], sold: dict[str, float], today: datetime) -> dict:
    """`rows`: the catalogue; `sold`: units sold per item key over the last SALES_DAYS days."""
    out = {"tracked": sum(1 for r in rows if r.stock is not None), "items": len(rows), "reorder": [], "slow_movers": [],
           "out_of_stock": [], "new_arrivals": [], "low_margin": [], "tied_up": 0.0}
    for r in rows:
        daily = sold.get(r.key, 0.0) / SALES_DAYS
        margin = r.margin
        base = {"name": r.name, "category": r.category, "stock": r.stock, "daily_sales": round(daily, 2),
                "margin": round(margin, 3) if margin is not None else None}
        if (today - (r.added_at or today)).days <= NEW_DAYS and (r.stock is None or r.stock > 0):
            out["new_arrivals"].append(base)
        if margin is not None and margin < LOW_MARGIN:
            out["low_margin"].append(base)
        if r.stock is None:
            continue
        if r.stock <= 0:
            out["out_of_stock"].append(base)
            continue
        cover = r.stock / daily if daily > 0 else None
        lead = r.lead_days or DEFAULT_LEAD
        if cover is not None and cover < lead + 3:
            need = math.ceil(daily * (lead + 14) - r.stock)
            out["reorder"].append({**base, "days_left": round(cover, 1), "lead_days": lead, "order_qty": max(1, need)})
        elif (cover is not None and cover > SLOW_DAYS) or (daily == 0 and (today - (r.added_at or today)).days > NEW_DAYS):
            tied = r.stock * (r.cost or r.price or 0)
            out["tied_up"] += tied
            out["slow_movers"].append({**base, "days_left": round(cover, 1) if cover else None, "tied_up": round(tied)})
    out["reorder"].sort(key=lambda x: x["days_left"])
    out["slow_movers"].sort(key=lambda x: -x["tied_up"])
    out["tied_up"] = round(out["tied_up"])
    return out


def view(rows: list[Item], rep: dict) -> dict | None:
    """What a message writer needs: {stock: {key: units|None}, margin: {key: m}, category: {key: c},
    slow: [names], new: [{name, category}]}. None without a catalogue."""
    if not rows:
        return None
    return {"stock": {r.key: r.stock for r in rows}, "margin": {r.key: r.margin for r in rows},
            "category": {r.key: (r.category or "").lower() for r in rows},
            "slow": [x["name"] for x in rep["slow_movers"]],
            "new": [{"name": x["name"], "category": (x["category"] or "").lower()} for x in rep["new_arrivals"]]}


def blocked(view: dict | None, item: str, offer: str = "") -> str:
    """Why an item must not be promoted ("" = fine): out of stock, or an offer bigger than its margin."""
    if not view or not item:
        return ""
    k = key(item)
    stock = view["stock"].get(k)
    if stock is not None and stock <= 0:
        return "out of stock"
    m = view["margin"].get(k)
    pct = num(offer.split("%")[0].split()[-1]) if "%" in (offer or "") else None
    if m is not None and pct is not None and pct / 100 > m - MARGIN_FLOOR:
        return "the offer is bigger than this item's margin allows"
    return ""
