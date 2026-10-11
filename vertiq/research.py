"""The web as a teacher: read what works for a sector (more revenue per visit, bringing people back,
more new customers, moving slow stock) and keep each tactic as an idea to test, with its source.

  search   a web search provider (brave | tavily | serpapi | serper), with the host's key
  teacher  a language model the host passes in reads the results and returns tactics as JSON
  check    a tactic is kept only if it cites one of the pages it was given and maps to a mechanism
           that can test it; everything else is dropped
  judge    reality: a kept tactic stays an idea until a control group shows it worked
"""
from __future__ import annotations

import json
import logging
import re

_log = logging.getLogger("vertiq")

TOPICS = ("increase average order value", "bring back lapsed customers", "get more new customers",
          "sell slow-moving stock")


MECHANISMS = {
    "spend_more": "a spend-more offer sized to each customer's usual bill",
    "bundle": "a combo of items often bought together",
    "add_on": "suggesting an add-on to their usual order",
    "winback_offer": "a win-back message with or without an offer",
    "referral": "asking loyal regulars to bring a friend",
    "new_arrival": "telling the right people about something new",
    "slow_mover": "offering slow-moving stock to the people who'd like it",
    "loyalty_tier": "a nudge toward their next loyalty tier",
    "timing": "sending at the moment they usually visit",
}


def search(query: str, provider: str, api_key: str, n: int = 5, *, http=None) -> list[dict]:
    """[{title, url, snippet}] from a search provider (brave | tavily | serpapi | serper); [] without a
    provider and key, or on any failure. `http`: the host's httpx-like client (get/post)."""
    provider = (provider or "").strip().lower()
    if not provider or not api_key:
        return []
    if http is None:                    # the host passes its HTTP client; vertiq never opens a connection itself
        _log.warning("no http client given for web search")
        return []
    try:
        if provider == "brave":
            r = http.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                         headers={"X-Subscription-Token": api_key, "Accept": "application/json"}, timeout=10)
            hits = (r.json().get("web") or {}).get("results") or []
            return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("description", "")} for h in hits][:n]
        if provider == "tavily":
            r = http.post("https://api.tavily.com/search", json={"api_key": api_key, "query": query, "max_results": n},
                          timeout=15)
            return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("content", "")}
                    for h in (r.json().get("results") or [])][:n]
        if provider == "serpapi":
            r = http.get("https://serpapi.com/search.json", params={"engine": "google", "q": query, "num": n,
                                                                   "api_key": api_key}, timeout=15)
            return [{"title": h.get("title", ""), "url": h.get("link", ""), "snippet": h.get("snippet", "")}
                    for h in (r.json().get("organic_results") or [])][:n]
        if provider == "serper":
            r = http.post("https://google.serper.dev/search", json={"q": query, "num": n},
                          headers={"X-API-KEY": api_key}, timeout=10)
            return [{"title": h.get("title", ""), "url": h.get("link", ""), "snippet": h.get("snippet", "")}
                    for h in (r.json().get("organic") or [])][:n]
    except Exception:
        _log.warning("web search failed (%s)", provider, exc_info=True)
        return []
    _log.warning("unknown search provider %r", provider)
    return []


def prompt_for(sector: str, topic: str, results: list[dict]) -> str:
    sources = "\n".join(f"[{i + 1}] {r['title']} | {r['url']}\n{r['snippet'][:500]}" for i, r in enumerate(results))
    mech = "\n".join(f"- {k}: {v}" for k, v in MECHANISMS.items())
    return (f"You help a {sector or 'local'} business {topic}. Read these sources:\n{sources}\n\n"
            f"Extract up to 5 concrete tactics that these sources support. Each must map to one mechanism:\n{mech}\n\n"
            'Reply with JSON only: [{"tactic": "...", "why": "...", "mechanism": "<one key above>", '
            '"source": "<the url it came from>"}]. Use only the urls above. No tactic without a source.')


def parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    return [d for d in data if isinstance(d, dict)]


def check(t: dict, urls: set) -> list[str]:
    """Why a tactic can't be kept ([] = it can): it must cite a page it was given and map to a mechanism."""
    problems = []
    if str(t.get("source") or "") not in urls:
        problems.append("cites a page it wasn't given")
    if t.get("mechanism") not in MECHANISMS:
        problems.append("no mechanism can test it")
    if not 10 <= len(str(t.get("tactic") or "")) <= 300:
        problems.append("not a usable tactic")
    return problems


def read(sector: str, topics, searcher, teacher, known: set | None = None) -> dict:
    """Search, read and check tactics for one sector. `searcher(query) -> results`,
    `teacher(prompt) -> text`. Returns {"tactics": [...kept...], "counts": {...}}; `known` holds
    tactic texts (lower case) already stored, so they come back as duplicates, not new."""
    seen = set(known or ())
    counts = {"status": "ok", "searched": 0, "kept": 0, "dropped": 0, "duplicates": 0}
    kept = []
    for topic in topics:
        results = [r for r in searcher(f"{sector} {topic} tactics") if r.get("url")]
        if not results:
            continue
        counts["searched"] += 1
        try:
            tactics = parse(teacher(prompt_for(sector, topic, results)))
        except Exception:
            _log.warning("teacher failed reading research (%s / %s)", sector, topic, exc_info=True)
            continue
        urls = {r["url"] for r in results}
        for t in tactics:
            if check(t, urls):
                counts["dropped"] += 1
                continue
            text = " ".join(str(t["tactic"]).split())
            if text.lower() in seen:
                counts["duplicates"] += 1
                continue
            seen.add(text.lower())
            kept.append({"topic": topic, "tactic": text, "why": " ".join(str(t.get("why") or "").split())[:500],
                         "mechanism": t["mechanism"], "source": str(t["source"])[:500]})
            counts["kept"] += 1
    return {"tactics": kept, "counts": counts}
