"""What's trending, made relevant: a trend reaches a customer only when the business sells it AND
this customer buys it or would likely try it. "Coffee is trending" goes to the cold-coffee regular,
never to the person who only buys tea.

Trend terms come from the host: the business's own list, and/or a feed of web terms
({"generic": [...], "<sector>": [...]}) the host fetches and passes in.
"""
from __future__ import annotations


def norm(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def feed_terms(feed: dict | None, sector: str, limit: int = 30) -> list[str]:
    """Terms for one sector (plus generic ones) from a feed object; [] for anything malformed."""
    if not isinstance(feed, dict):
        return []
    return [str(t) for t in (feed.get(norm(sector)) or []) + (feed.get("generic") or []) if str(t).strip()][:limit]


def matching_items(terms: list[str], items: list[str]) -> dict[str, str]:
    """{item the business sells: the trend term it matches}. A term matches an item when one
    contains the other, word for word ("coffee" ↔ "Cold Coffee")."""
    out: dict[str, str] = {}
    for term in terms:
        t = norm(term)
        if not t:
            continue
        tw = set(t.split())
        for item in items:
            i = norm(item)
            if t in i or i in t or (tw and tw <= set(i.split())):
                out.setdefault(item, term)
    return out


def trending_for(choices: dict | None, sources: list[tuple[str, list[str]]]) -> dict[str, dict]:
    """{item: {"term", "source"}} for the business's items that are trending, from
    [(source name, terms), ...] in priority order. `choices` is vertiq.affinity.analyse output."""
    if not choices:
        return {}
    items = sorted({i for bought in (choices.get("_bought") or {}).values() for i in bought})
    found: dict[str, dict] = {}
    for source, terms in sources:
        for item, term in matching_items(terms, items).items():
            found.setdefault(item, {"term": term, "source": source})
    return found
