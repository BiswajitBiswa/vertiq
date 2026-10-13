"""Lagged sequences: patterns of the form "an event of category A is followed, on the entity's next
event, by category B, typically N days later". Each pair is tested: is B more likely to come next after
A than after any other category? The family of tests is filtered for false discoveries before anything
is reported, as vertiq.insights does.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from .contract import Dataset
from .learn import benjamini_hochberg, two_proportion_p


def find(ds: Dataset, *, attr: str, type: str | None = None, min_support: int = 20, q: float = 0.1,
         limit: int = 10) -> dict:
    """Consecutive-event transitions between values of `attr` (for example a product category), per
    entity, in time order. `type` limits to one event type. Returns tested pairs with lift and median lag."""
    pair, first, nxt, lags = Counter(), Counter(), Counter(), defaultdict(list)
    n = 0
    for evs in ds.by_entity().values():
        seq = [(e.at, e.attrs.get(attr)) for e in evs if (type is None or e.type == type) and e.attrs.get(attr) not in (None, "")]
        for (t1, a), (t2, b) in zip(seq, seq[1:]):
            pair[(a, b)] += 1
            first[a] += 1
            nxt[b] += 1
            lags[(a, b)].append((t2 - t1).days)
            n += 1
    if not n:
        return {"status": "learning", "transitions": 0, "patterns": []}
    cands = []
    for (a, b), c in pair.items():
        if a == b or c < min_support:
            continue
        rest_b, rest_n = nxt[b] - c, n - first[a]                    # B next after any other category
        p = two_proportion_p(c, first[a], rest_b, rest_n) if rest_n else 1.0
        base = rest_b / rest_n if rest_n else 0.0
        rate = c / first[a]
        cands.append({"from": a, "to": b, "support": c, "rate": round(rate, 4), "rate_otherwise": round(base, 4),
                      "lift": round(rate / base, 2) if base else None, "median_lag_days": median(lags[(a, b)]), "_p": p})
    keep = benjamini_hochberg([c["_p"] for c in cands], q) if cands else []
    found = sorted((c for c, k in zip(cands, keep) if k and c["rate"] > c["rate_otherwise"]), key=lambda c: -(c["lift"] or 0))
    return {"status": "ok", "transitions": n, "tested": len(cands),
            "patterns": [{k: v for k, v in c.items() if k != "_p"} for c in found[:limit]]}
