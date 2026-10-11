"""Training data for a business's own language model: rules every line must pass, contact details
masked, and exports in the formats fine-tuning tools read.

  check          why a line fails ([] = it passes): one line, short, no links, no offer nobody gave,
                 no invented dates or times, none of the banned phrases
  mask           phone numbers and emails never enter a dataset
  training_rows  chat-format SFT rows (mlx-lm-lora / TRL / Unsloth) and preference pairs from
                 human edits (edited = chosen, draft = rejected)
  outcome_rows   rows scored by what happened: each line's lift against its control group (for a
                 reward model) and pairs where the same facts got two lines and one did clearly
                 better (for DPO)
  TRAINABLE      providers whose output a model may learn from; others (e.g. a closed API whose
                 terms forbid it) are skipped
"""
from __future__ import annotations

import json
import re

GOALS = {
    "winback": "bring back a customer who hasn't visited in a while",
    "secondVisit": "get a first-time customer to come back a second time",
    "vip": "thank a regular and make them feel valued",
    "birthday": "wish a customer a happy birthday and invite them in",
    "festival": "greet a customer for a festival and invite them in",
}
KEEP = ("verified", "kept", "edited")
TRAINABLE = {None, "", "teacher", "local"}
MAX_LEN = 240
MIN_OUTCOME_SENDS = 5
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_URL = re.compile(r"https?://|www\.", re.I)
# "free" is an offer in "a free dessert", not in "your chair is free this week" or "a free slot"
_OFFERISH = re.compile(r"(\d+\s?%|[₹$€£]\s?\d+|rs\.?\s?\d+|\bfor free\b|\bdiscount\b|\bcashback\b|\b\d+\s?off\b|"
                       r"\bfree\s+(?!slots?\b|time\b|tables?\b|chairs?\b|seats?\b|to\b|this\b|today\b|tomorrow\b|now\b)[a-z]+)",
                       re.I)
_BANNED = ("please", "successfully", "dear valued", "act now", "limited time only")


def mask(text: str) -> str:
    """Contact details never enter the dataset."""
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text or ""))


def check(text: str, *, offer: str = "", max_len: int = MAX_LEN) -> list[str]:
    """Why a message fails the rules ([] = it passes). The same rules for a teacher's examples and
    for any model that will one day write live."""
    t = (text or "").strip()
    problems = []
    if not t:
        return ["empty"]
    if "\n" in t:
        problems.append("more than one line")
    if len(t) > max_len:
        problems.append(f"longer than {max_len} characters")
    if _URL.search(t):
        problems.append("contains a link")
    if not offer and _OFFERISH.search(t):
        problems.append("mentions an offer that wasn't given")
    low = t.lower()
    problems += [f"uses '{w}'" for w in _BANNED if w in low]
    if re.search(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b\d{1,2}(st|nd|rd|th)\b", low):
        problems.append("invents a date or time")
    return problems


def prompt_for(sector: str, goal: str, offer: str = "", channel: str = "whatsapp", noun: str = "customers") -> str:
    """The instruction a student model learns to follow."""
    return (f"You write short {channel} messages a local business sends to its {noun}. "
            f"Business sector: {sector or 'general'}. Goal: {GOALS.get(goal, goal)}. Tone: warm. "
            + (f"Mention this offer: '{offer}'. " if offer else "Do not invent any discount or offer. ")
            + "Use the token {name} for the customer's name. One line, under 160 characters. "
            "Do not fabricate facts, prices, dates, or claims. Return exactly 3 distinct variants, one per line.")


def training_rows(examples: list[dict]) -> dict:
    """`examples`: [{prompt, output, verdict, sector, source, meta}] oldest first.
    Returns {"sft": [...], "preference": [...]} from those that passed and may be learned from."""
    sft, prefs = [], []
    for r in examples:
        meta = r.get("meta") or {}
        if r.get("verdict") not in KEEP or meta.get("provider") not in TRAINABLE:
            continue
        if r["verdict"] == "edited" and meta.get("final_problems"):
            continue
        answer = meta.get("final") if r["verdict"] == "edited" else r["output"]
        sft.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": answer}],
                    "sector": r.get("sector", ""), "source": r.get("source", "")})
        if r["verdict"] == "edited":
            prefs.append({"prompt": r["prompt"], "chosen": answer, "rejected": r["output"], "sector": r.get("sector", "")})
    return {"sft": sft, "preference": prefs}


def outcome_rows(examples: list[dict], min_sends: int = MIN_OUTCOME_SENDS, min_gap: float = 0.1) -> dict:
    """`examples`: verified lines with meta["outcome"] = {sent, rate, lift}. Returns rewards and
    preference pairs (same fact card, clearly different results)."""
    measured = [x for x in examples if (x.get("meta") or {}).get("outcome", {}).get("sent", 0) >= min_sends
                and (x.get("meta") or {}).get("provider") in TRAINABLE]

    def score(x):
        o = x["meta"]["outcome"]
        return o["lift"] if o.get("lift") is not None else o["rate"]
    rewards = [{"prompt": x["prompt"], "output": x["output"], "reward": score(x), "sent": x["meta"]["outcome"]["sent"],
                "sector": x.get("sector", "")} for x in measured]
    by_card: dict[str, list] = {}
    for x in measured:
        by_card.setdefault(x["meta"].get("card", x["prompt"]), []).append(x)
    pairs = []
    for group in by_card.values():
        group.sort(key=score, reverse=True)
        best, worst = group[0], group[-1]
        if best is not worst and best["output"] != worst["output"] and score(best) - score(worst) >= min_gap:
            pairs.append({"prompt": best["prompt"], "chosen": best["output"], "rejected": worst["output"],
                          "sector": best.get("sector", ""), "lift_gap": round(score(best) - score(worst), 4)})
    return {"rewards": rewards, "preference": pairs}


def jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
