"""Comparing two message variants (e.g. "usual": stick to what they buy, vs "explore": suggest
something new), each against the other and against a control group kept out on purpose."""
from __future__ import annotations

from .learn import two_proportion_p

MIN_ARM = 30
ARMS = ("usual", "explore")


def compare(tally: dict[str, list[int]], window_days: int, *, a: str = "usual", b: str = "explore",
            min_arm: int = MIN_ARM, alpha: float = 0.05) -> dict:
    """`tally`: {arm: [returned, sent]} for both arms (and optionally "control")."""
    rate = {k: (round(x / n, 4) if n else None) for k, (x, n) in tally.items()}
    out = {"window_days": window_days, "arms": {k: {"sent": n, "returned": x, "rate": rate[k]} for k, (x, n) in tally.items()}}
    (xa, na), (xb, nb) = tally.get(a, [0, 0]), tally.get(b, [0, 0])
    if na < min_arm or nb < min_arm:
        out.update(status="learning", verdict=f"needs {min_arm} finished sends in each arm")
        return out
    p = two_proportion_p(xb, nb, xa, na)
    winner = b if rate[b] > rate[a] else a
    out.update(status="ok", p_value=round(p, 5), winner=winner if p < alpha else None,
               verdict=(("Suggesting something new brings more people back" if winner == b
                         else "Their usual brings more people back") if p < alpha else "No clear difference yet"))
    return out
