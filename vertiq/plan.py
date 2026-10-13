"""What to do under a budget: given the estimated effect of each action on each entity, its cost and the
value of the outcome, choose at most one action per entity so that the expected benefit is largest
without exceeding the budget.

  candidates  [{"entity", "action", "effect", "cost", "value"}]: `effect` is the estimated change in
              the probability of the target event if the action is taken (from an uplift model);
              `value` is what one target event is worth (positive if the event is wanted, such as a
              renewal; negative if it is to be prevented, such as a failure).
  benefit     effect x value - cost, in money.

The choice is a multiple-choice knapsack, solved greedily: actions are taken in order of benefit per
unit of cost, and an entity's action is upgraded when a costlier one adds more benefit per extra unit
of cost than the next best alternative. Only actions with positive expected benefit are chosen.

`from_uplift` builds candidates from trained uplift tasks (tasks.predict_uplift). An action whose model
could not separate individual effects contributes its average effect for every entity, and an action
still collecting data is left out and reported, so a plan never rests on effects that were not measured.
"""
from __future__ import annotations


def allocate(candidates: list[dict], budget: float) -> dict:
    """{"assignments": [{"entity", "action", "effect", "cost", "benefit"}], "spent", "expected_benefit",
    "budget", "unassigned_with_positive_benefit"}. At most one action per entity; total cost <= budget."""
    best_for: dict[str, list[dict]] = {}
    for c in candidates:
        b = c["effect"] * c["value"] - c["cost"]
        if b <= 0:
            continue
        best_for.setdefault(c["entity"], []).append({**c, "benefit": b})
    # per entity, keep only options that are not dominated (higher cost must bring higher benefit),
    # then walk them as increments: first option, then upgrades
    steps = []
    for eid, opts in best_for.items():
        opts.sort(key=lambda o: (o["cost"], -o["benefit"]))
        frontier = []
        for o in opts:
            if frontier and o["benefit"] <= frontier[-1]["benefit"]:
                continue                               # costs more, adds nothing
            # keep the upper concave hull from (0, 0), so each upgrade adds less per unit of cost than the
            # step before it and the greedy order never needs an upgrade before its predecessor
            while frontier:
                c1, b1 = frontier[-1]["cost"], frontier[-1]["benefit"]
                c0, b0 = (frontier[-2]["cost"], frontier[-2]["benefit"]) if len(frontier) > 1 else (0.0, 0.0)
                if (b1 - b0) * (o["cost"] - c1) <= (o["benefit"] - b1) * (c1 - c0):
                    frontier.pop()
                else:
                    break
            frontier.append(o)
        prev = None
        for o in frontier:
            dc = o["cost"] - (prev["cost"] if prev else 0.0)
            db = o["benefit"] - (prev["benefit"] if prev else 0.0)
            steps.append((db / dc if dc > 0 else float("inf"), eid, o, prev))
            prev = o
    steps.sort(key=lambda s: -s[0])
    chosen: dict[str, dict] = {}
    spent = 0.0
    for _ratio, eid, o, prev in steps:
        cur = chosen.get(eid)
        if (prev is None and cur is not None) or (prev is not None and cur is not prev):
            continue                                   # an upgrade applies only on top of its predecessor
        extra = o["cost"] - (cur["cost"] if cur else 0.0)
        if spent + extra > budget + 1e-9:
            continue
        chosen[eid] = o
        spent += extra
    assignments = sorted(({"entity": o["entity"], "action": o["action"], "effect": round(o["effect"], 4),
                           "cost": o["cost"], "benefit": round(o["benefit"], 2)} for o in chosen.values()),
                         key=lambda a: -a["benefit"])
    return {"assignments": assignments, "spent": round(spent, 2), "budget": budget,
            "expected_benefit": round(sum(a["benefit"] for a in assignments), 2),
            "unassigned_with_positive_benefit": len(set(best_for) - set(chosen))}


def from_uplift(tasks: dict, entity_ids: list[str], *, costs: dict[str, float], value, budget: float, at=None) -> dict:
    """A plan from uplift tasks: {action name: UpliftTask}, the cost of each action, and the value of one
    target event (a number, or {entity id: number}; negative for events to prevent). Returns allocate()'s
    result plus "sources": how each action's effects were obtained ("individual", "average" or "excluded")."""
    candidates, sources = [], {}
    for name, task in tasks.items():
        rep = task.report()
        if task.status == "learning":
            sources[name] = {"source": "excluded", "reason": "still collecting data"}
            continue
        if task.status == "trained":
            sources[name] = {"source": "individual"}
        else:
            sources[name] = {"source": "average", "reason": rep.get("reason", "")}
        avg = rep["metrics"]["average_effect"]["effect"]
        for eid in entity_ids:
            if task.status == "trained":
                s = task.score(eid, at)
                if s is None:
                    continue
                eff = s["uplift"]
            else:
                eff = avg
            v = value.get(eid, 0.0) if isinstance(value, dict) else value
            candidates.append({"entity": eid, "action": name, "effect": eff, "cost": costs[name], "value": v})
    return {**allocate(candidates, budget), "sources": sources}
