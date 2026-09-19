"""Fixed information-seeking control over actual owner-held field weights."""

import copy

import numpy as np
import torch

from experiments.gap_inquiry import digest
from experiments.structural_field import design


def choose(session, subject, candidates, used, policy):
    x = torch.tensor(candidates, dtype=torch.float64)
    radial, rich = (session.model(subject, k) for k in ("radial", "directional"))
    with torch.no_grad():
        if policy == "disagreement":
            a = design(x, "directional")
            covariance = torch.linalg.inv(rich.gram) * .01**2
            variance = torch.einsum("nip,pq,niq->n", a, covariance, a)
            scores = ((radial(x) - rich(x))**2).mean(-1) + variance
        elif policy == "balanced":
            known = np.array([r["x"] for r in session.subjects[subject]["observations"]])
            scores = torch.tensor(((np.asarray(candidates)[:, None] - known[None])**2).sum(-1).min(-1))
        else:
            raise ValueError("Unknown finite investigation policy")
        scores[list(used)] = -torch.inf
        return int(scores.argmax())


def investigate(session, subject, initial, candidates, selection, audit, measure, policy, sink):
    """Commit each probe before obtaining evidence; resume without repeating credit."""
    record = session.subjects[subject]
    if record["qualification"] is not None:
        return record["qualification"]
    contract = digest({"subject": subject, "policy": policy, "initial": initial.tolist(), "candidates": candidates.tolist(),
                       "selection": selection.tolist(), "audit": audit.tolist(), "goal": record["goal"], "source": record["source"]})
    if session.pending is None:
        session.pending = {"contract": contract, "subject": subject, "policy": policy, "initial_completed": 0,
                           "used": [], "trace": [], "decision": None, "selection": [], "audit": []}
        sink(session.state())
    pending = session.pending
    if pending["contract"] != contract:
        raise ValueError("Preserve the existing pending investigation before changing its premises")
    while pending["initial_completed"] < 12:
        i = pending["initial_completed"]
        session.observe(subject, measure(initial[i], "initial-" + str(i), "acquisition"))
        pending["initial_completed"] += 1
        sink(session.state())
    if record["original_state"] is not None and "initial_answer" not in record:
        record["initial_answer"] = session.view(subject).imagine(record["original_state"])
        sink(session.state())
    while len(pending["used"]) < 12:
        if pending["decision"] is None:
            index = choose(session, subject, candidates, pending["used"], policy)
            decision = {"index": index, "x": candidates[index].tolist(), "goal": record["goal"],
                        "source": record["source"], "revision": record["revision"],
                        "predictors": {k: session.model(subject, k).identity() for k in ("radial", "directional")}}
            pending["decision"] = decision | {"id": digest(decision)}
            sink(session.state())
        decision = pending["decision"]
        if (decision["revision"] != record["revision"] or decision["predictors"] !=
                {k: session.model(subject, k).identity() for k in ("radial", "directional")}):
            raise ValueError("Stale physical investigation decision")
        row = measure(decision["x"], "probe-" + str(decision["index"]), "acquisition")
        session.observe(subject, row)
        pending["used"].append(decision["index"])
        pending["trace"].append({"decision": copy.deepcopy(decision), "observation": row})
        pending["decision"] = None
        sink(session.state())
    for key, points, purpose in (("selection", selection, "selection"), ("audit", audit, "adequacy")):
        while len(pending[key]) < len(points):
            index = len(pending[key])
            pending[key].append(measure(points[index], key + "-" + str(index), purpose))
            sink(session.state())
    receipt = session.qualify(subject, pending["selection"], pending["audit"])
    record["investigation"] = copy.deepcopy(pending)
    session.pending = None
    sink(session.state())
    return receipt
