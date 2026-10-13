"""A check for actions proposed by an automated agent or a person: before an action is taken on an entity,
ask what the measured evidence says about it.

  check(proposal, tasks, ...) -> {"decision": "allow" | "hold" | "block", "reasons": [...], "evidence": {...}}

  allow  the action has a measured effect in the wanted direction (its interval excludes zero), the
         estimate for this entity agrees, and no rule or limit objects
  block  the measured average effect goes the wrong way with confidence, or a rule flags the entity
  hold   anything else, to be decided by a person: the action was never measured, the evidence is
         unclear, the estimate for this entity disagrees with the average (individual estimates carry no
         interval, so they never block on their own), or the proposal exceeds a limit

The check never acts and never decides on its own evidence-free judgement: an action Vertiq has no
outcome data for is held, not allowed. It reads only the uplift tasks (tasks.predict_uplift) and rule
flags (vertiq.rules) the host passes in.
"""
from __future__ import annotations

import math
from statistics import NormalDist


def _interval(a: dict, n_t: int, n_c: int, level: float) -> tuple[float, float]:
    """Normal interval for the difference of two rates, treated minus control."""
    pt, pc = a["treated_rate"], a["control_rate"]
    se = math.sqrt(max(pt * (1 - pt), 1e-9) / max(n_t, 1) + max(pc * (1 - pc), 1e-9) / max(n_c, 1))
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    return a["effect"] - z * se, a["effect"] + z * se


def check(proposal: dict, tasks: dict, *, goal: str = "reduce", flags: list[dict] | None = None,
          limits: dict | None = None, level: float = 0.95, min_entity_effect: float = 0.0, at=None) -> dict:
    """`proposal`: {"entity", "action", optional "amount"}. `tasks`: {action name: UpliftTask} for the
    target the host cares about. `goal`: "reduce" when the target event is to be prevented (a failure,
    a cancellation), "increase" when it is wanted (a renewal, a repeat order). `flags`: rule flags for
    entities (dicts with "entity" and "rule"). `limits`: {action name: largest "amount" allowed without
    a person}. `min_entity_effect`: how far the entity's own estimate may go the wrong way before the
    action is held (0: any harmful estimate holds it)."""
    sign = -1.0 if goal == "reduce" else 1.0
    eid, action = proposal["entity"], proposal["action"]
    reasons, evidence = [], {}
    task = tasks.get(action)
    entity_flags = [f for f in (flags or []) if f.get("entity") == eid]
    if entity_flags:
        evidence["flags"] = [f.get("rule") for f in entity_flags]
        return {"decision": "block", "reasons": [f"rule flags on this entity: {', '.join(map(str, evidence['flags']))}"],
                "evidence": evidence}
    limit = (limits or {}).get(action)
    if limit is not None and proposal.get("amount", 0) > limit:
        return {"decision": "hold", "reasons": [f"amount {proposal['amount']} exceeds the limit of {limit} for {action}"],
                "evidence": evidence}
    if task is None or task.status == "learning":
        return {"decision": "hold", "reasons": [f"no measured outcomes for {action}" if task is None
                                                else f"{action} is still collecting data"], "evidence": evidence}
    rep = task.report()
    m = rep["metrics"]
    a = m["average_effect"]
    lo, hi = _interval(a, m["treated"], m["control"], level)
    evidence.update(average_effect=a["effect"], interval=[round(lo, 4), round(hi, 4)], level=level,
                    treated=m["treated"], control=m["control"], p_value=a["p_value"])
    wanted_lo, wanted_hi = sorted((sign * lo, sign * hi))                 # in the wanted direction, positive is good
    if wanted_hi < 0:
        return {"decision": "block", "reasons": [f"{action} works against the goal on average ({a['effect']:+.3f}, "
                                                 f"interval {lo:+.3f} to {hi:+.3f})"], "evidence": evidence}
    entity_effect = None
    if task.status == "trained":
        s = task.score(eid, at)
        if s is not None:
            entity_effect = s["uplift"]
            evidence["entity_effect"] = entity_effect
            if sign * entity_effect < -min_entity_effect:
                # an individual estimate has no interval, so it can hold an action but never block it
                return {"decision": "hold", "reasons": [f"the estimate for {eid} goes against the goal ({entity_effect:+.3f}); "
                                                        "a person should decide"], "evidence": evidence}
    if wanted_lo <= 0:
        reasons.append(f"the average effect of {action} is not clear (interval {lo:+.3f} to {hi:+.3f})")
    if task.status != "trained":
        reasons.append("individual effects could not be separated; the decision rests on the average")
    if entity_effect is None and task.status == "trained":
        reasons.append(f"no estimate for {eid} (no history at this time)")
    if reasons and wanted_lo <= 0:
        return {"decision": "hold", "reasons": reasons, "evidence": evidence}
    if entity_effect is not None and sign * entity_effect <= 0:
        return {"decision": "hold", "reasons": [f"the estimate for {eid} shows no benefit ({entity_effect:+.3f})"],
                "evidence": evidence}
    return {"decision": "allow", "reasons": reasons or [f"{action} has a measured effect in the wanted direction"],
            "evidence": evidence}
