"""Evidence-driven investigations; no concealed mechanism or reference access here."""

import copy
from fractions import Fraction as Q

import numpy as np

from experiments.counterfactual_core import EXPANSIONS, POWERS, PROBES, admitted, at, propose
from experiments.gap_inquiry import digest

POLICIES = ("entropy", "spread", "balanced", "random", "learned")
MAX_STEPS = 10
EVALUATION_PROBES = tuple(map(Q, ("3/16", "5/16", "13/16", "19/16", "7/4", "5/2", "5", "10")))


class ProbeTable:
    def __init__(self, event):
        self.event = event
        self.candidates = event["candidates"]
        names = sorted(self.candidates[0]["observables"]) if self.candidates else []
        self.queries = [(z, n) for z in PROBES if z != 1 for n in names if n != event["question"]["axis"]]
        self.values = np.full((len(self.candidates), len(self.queries)), np.nan)
        self.valid = np.zeros_like(self.values, dtype=bool)
        for i, c in enumerate(self.candidates):
            for j, (z, n) in enumerate(self.queries):
                if admitted(c, z):
                    v = at(c["observables"][n], z)
                    if v is not None:
                        self.values[i, j] = float(v)
                        self.valid[i, j] = True

    def features(self, alive, used):
        values, valid = self.values[alive], self.valid[alive]
        result = []
        for j, (z, _) in enumerate(self.queries):
            numbers = values[:, j][valid[:, j]]
            _, counts = np.unique(numbers, return_counts=True)
            proportions = counts / max(len(numbers), 1)
            entropy = float(-(proportions * np.log(proportions)).sum())
            scale = max(1., float(np.max(np.abs(numbers))) if len(numbers) else 1.)
            spread = float(np.ptp(numbers)) / scale if len(numbers) else 0.
            distance = min((abs(float(z - Q(q[0]))) for q in used), default=1.)
            result.append([1., entropy, spread, len(counts) / max(len(numbers), 1),
                           len(numbers) / max(int(sum(alive)), 1), np.log1p(abs(float(z))),
                           min(distance, 16.) / 16., 1. / (1 + len(used))])
        return np.asarray(result)

    def update(self, alive, evidence):
        if evidence["status"] != "SIMULATED_OBSERVATION":
            return alive.copy()
        query = (Q(evidence["z"]), evidence["variable"])
        j = self.queries.index(query)
        bound = evidence["absolute_error_bound"]
        tolerance = 1e-9 * (1 + abs(evidence["value"]))
        return alive & self.valid[:, j] & (np.abs(self.values[:, j] - evidence["value"]) <= bound + tolerance)


def compatible(table, evidence):
    alive = np.ones(len(table.candidates), dtype=bool)
    for item in evidence:
        alive = table.update(alive, item)
    return alive


def select(table, alive, used, policy, weights, seed):
    features = table.features(alive, used)
    eligible = [i for i, q in enumerate(table.queries) if (str(q[0]), q[1]) not in used and features[i, 4] > 0]
    if not eligible:
        return None, features
    # Ask the original question first. Once its curves agree, seek a different
    # observable that separates the surviving explanations of that answer.
    direct = [i for i in eligible if table.queries[i][1] == table.event["question"]["target"] and features[i, 1] > 1e-10]
    eligible = direct or [i for i in eligible if features[i, 1] > 1e-10] or eligible
    if policy == "random":
        rng = np.random.default_rng(seed + len(used))
        return eligible[int(rng.integers(len(eligible)))], features
    if policy == "learned":
        scores = features @ np.asarray(weights)
    elif policy == "entropy":
        scores = features[:, 1] * features[:, 4]
    elif policy == "spread":
        scores = features[:, 2] * features[:, 4]
    elif policy == "balanced":
        scores = features[:, 6] + .01 * features[:, 4]
    else:
        raise ValueError("Undeclared investigation policy")
    return max(eligible, key=lambda i: (float(scores[i]), -i)), features


def commit(event, predictor):
    event = copy.deepcopy(event)
    event["predictor"] = predictor
    event["commitment"] = digest({k: v for k, v in event.items() if k != "commitment"})
    return event


def investigate(event, matrices, evidence_source, policy, weights, seed, sink, saved=None):
    """Pause at durable boundaries; finish or explicitly preserve the original gap.

    The source exposes only observe(z,index,variable). Checks, concealed truth,
    external names and final evaluation are deliberately absent from this API.
    """
    if saved is None:
        state = {"original_goal": event["question"]["id"], "predictor": event["predictor"],
                 "initial_commitment": event["commitment"], "events": [event], "observations": [],
                 "decisions": [], "followups": [], "training": [], "policy": policy,
                 "weights": list(weights), "seed": seed, "status": "INVESTIGATING"}
        sink(copy.deepcopy(state))
    else:
        state = copy.deepcopy(saved)
        if (state["initial_commitment"], state["policy"], state["seed"], state["weights"]) != (event["commitment"], policy, seed, list(weights)):
            raise ValueError("Changed original goal, procedure or resumption state")
        if state["status"] != "INVESTIGATING":
            return state
    event = state["events"][-1]
    table = ProbeTable(event)
    alive = compatible(table, state["observations"])
    for step in range(len(state["observations"]), MAX_STEPS):
        if not alive.any():
            if len(state["events"]) > 1:
                break
            enlarged = propose(event["question"], matrices, (*POWERS, *EXPANSIONS))
            event = commit(enlarged, state["predictor"])
            state["events"].append(event)
            state["followups"].append({"kind": "EXPAND_MECHANISM", "trigger": "Every initial explanation contradicted by collected evidence",
                                        "original_goal": state["original_goal"], "powers_added": list(EXPANSIONS), "external_hints": 0})
            sink(copy.deepcopy(state))  # New proposals precede further evidence.
            table = ProbeTable(event)
            alive = compatible(table, state["observations"])
            if not alive.any():
                break
        used = {(r["z"], r.get("variable", event["question"]["target"])) for r in state["observations"]}
        index, features = select(table, alive, used, policy, weights, seed)
        if index is None:
            break
        # Consensus of all executable predictions, not one confident estimate.
        if state["observations"] and max(features[:, 1], default=0.) < 1e-10:
            break
        z, variable = table.queries[index]
        decision = {"goal": state["original_goal"], "predictor": state["predictor"], "proposal": event["commitment"],
                    "index": step, "z": str(z), "variable": variable, "features": features[index].tolist(),
                    "survivors_before": int(sum(alive)), "policy": policy}
        decision["id"] = digest(decision)
        if len(state["decisions"]) == step:
            state["decisions"].append(decision)
        elif state["decisions"][step] != decision:
            raise ValueError("A pending intervention changed on resumption")
        sink(copy.deepcopy(state))
        observation = evidence_source.observe(z, step, variable)
        observation = {**observation, "variable": variable, "decision": decision["id"]}
        state["observations"].append(observation)
        previous = int(sum(alive))
        alive = table.update(alive, observation)
        progress = float(np.log((previous + 1) / (sum(alive) + 1))) if alive.any() and observation["status"] == "SIMULATED_OBSERVATION" else -1.
        state["training"].append({"features": decision["features"], "verified_later_target": progress,
                                  "decision": decision["id"], "evidence": observation.get("id"), "proposal": event["commitment"]})
        if variable != event["question"]["target"]:
            state["followups"].append({"kind": "DISCRIMINATE_EXPLANATIONS", "original_goal": state["original_goal"],
                                        "variable": variable, "z": str(z), "reason": "Investigate what else changes under competing mechanisms"})
        sink(copy.deepcopy(state))
    state["survivors"] = [c["id"] for c, keep in zip(table.candidates, alive, strict=True) if keep]
    state["status"] = "SCOPED_ANSWER" if state["survivors"] else "OPEN_MECHANISM_GAP"
    state["answer"] = [{"candidate": c["id"], "curve": c["curve"], "conditions": c["conditions"], "analysis": c["analysis"],
                        "mechanism": c["spec"], "aliases": event["aliases"].get(c["id"], [])}
                       for c in table.candidates if c["id"] in state["survivors"]]
    state["next_action"] = ("Use scoped surviving predictions; choose a different unexplored dependency next" if state["survivors"]
                            else "Preserve residuals and original question; a broader mechanism representation or new evidence is required")
    state["id"] = digest(state)
    sink(copy.deepcopy(state))
    return state


def next_question(frontier, finished, followups=()):
    """Spread investigations across retained domains; never replay a closed id."""
    counts = {d: sum(r["domain"] == d for r in finished.values()) for d in {q["domain"] for q in frontier}}
    candidates = [q for q in frontier if q["id"] not in finished]
    priority = {r["question"] for r in followups if "question" in r}
    if not candidates:
        return None
    return max(candidates, key=lambda q: (q["id"] in priority, -counts[q["domain"]], len(q["models"]), q["id"]))
