"""Independently replay and, when supported, integrate the sustained successor."""

import copy
import json

import numpy as np
import torch

from experiments.continuing_growth.audit import replay_scores
from experiments.continuing_growth.data import load as growth_data
from experiments.self_study.algebra import check, independent
from experiments.sustained_refinement import (
    OUT,
    PREFIXES,
    ROOT,
    RUN,
    STEPS,
    Session,
    contracts,
    load,
    read,
    state,
    write,
)
from sera.session_state import model_identity
from workbench.storage import Store


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve completed sustained audit")
    final, procedure = read(OUT / "final.json"), read(OUT / "procedure.json")
    reports = [replay_scores(final["initial"])]
    reports += [replay_scores(r["result"]) for r in final["results"]]
    reports += [replay_scores(r["after"]) for r in procedure["trials"]]
    parent = read(RUN / "parent.json")
    session = Session(parent)
    protected = {k: v.clone() for k, v in session.owner.state_dict().items()
                 if not k.startswith((*PREFIXES, "refinement_policy."))}
    selected = final["selected"] or "balanced_adam-3901"
    chosen = state(RUN / f"{selected}-{STEPS:04d}.pt")
    session.restore(chosen)
    if not all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items()):
        raise AssertionError("Protected retained knowledge changed")
    cases = []
    rng = np.random.default_rng(39981)
    for domain in ("integral", "sum"):
        for _ in range(32):
            p = rng.integers(-8, 9, 4).tolist()
            proposed = session.owner.study_maps[domain].propose(p)
            good = check(domain, p, proposed)["accepted"] and independent(domain, p, proposed)
            cases.append({"domain": domain, "input": p, "proposal": proposed, "accepted": good})
    if not all(c["accepted"] for c in cases):
        raise AssertionError("Retained certified mathematics changed")
    data, probe = load("train"), growth_data("probe")
    for arm in ("balanced_adam", "autonomous"):
        session.restore(state(RUN / f"{arm}-3901-{STEPS-128:04d}.pt"))
        for _ in range(128):
            session.step(data, probe, arm)
        if not equal(session.state(), state(RUN / f"{arm}-3901-{STEPS:04d}.pt")):
            raise AssertionError("Refinement did not resume exactly: "+arm)
    session.restore(chosen)
    grounded = session.base.base.base.base.base.base.base.grounded
    if grounded.gate["autonomous"]:
        raise AssertionError("Closed source gate changed")
    result = {"contracts": contracts(), "owner": model_identity(session.owner),
              "selected": selected, "admitted": final["admitted"], "mathematics": cases,
              "protected_tensors": len(protected), "protected_equal": True,
              "exact_resume_decisions": 256, "same_actual_owner": session.owner is session.base.owner,
              "prediction_items": sum(r["items"] for r in reports),
              "maximum_loss_difference": max(r["maximum_loss_difference"] for r in reports),
              "novel_definition_gate": grounded.gate["autonomous"]}
    if final["admitted"]:
        store = Store(ROOT / "runs/sera-sustained-growth-live")
        if store.read() is not None:
            raise FileExistsError("Preserve existing refined owner")
        saved = session.snapshot()
        store.commit(saved, None)
        restored = Session(saved["parent"], saved=copy.deepcopy(chosen))
        if model_identity(restored.owner) != saved["owner"]:
            raise AssertionError("Integrated owner did not reload exactly")
        result["store"] = "runs/sera-sustained-growth-live"
    else:
        result["store"] = "runs/sera-growth-live"
    write(OUT / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"mathematics", "contracts"}}, indent=2))


if __name__ == "__main__":
    main()
