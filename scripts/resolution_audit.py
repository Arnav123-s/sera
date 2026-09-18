"""Verify original-gate decisions, exact replay and admission of step resolution."""

import json

import numpy as np
import torch

from experiments.continuing_growth.audit import replay_scores
from experiments.continuing_growth.data import load as growth_data
from experiments.refinement_resolution import OUT, ROOT, RUN, STEPS, ResolutionSession, contracts
from experiments.self_study.algebra import check, independent
from experiments.sustained_refinement import PREFIXES, admissible, load, read, state, write
from experiments.sustained_refinement import RUN as PARENT_RUN
from scripts.refinement_audit import equal
from sera.session_state import model_identity
from workbench.storage import Store


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve completed step-resolution audit")
    final = read(OUT / "final.json")
    reports = [replay_scores(r["result"]) for r in final["results"]]
    parent = read(PARENT_RUN / "parent.json")
    selected = int(final["selected"])
    session = ResolutionSession(parent, saved=state(RUN / f"{selected}-0000.pt"))
    protected = {k: v.clone() for k, v in session.owner.state_dict().items()
                 if not k.startswith((*PREFIXES, "refinement_policy."))}
    counted, recovered = 0, 0
    for seed in (4001, 4002):
        saved = state(RUN / f"{seed}-{STEPS:04d}.pt")
        session.base.restore(saved["state"])
        anchor = session.base.anchor
        for event in session.base.events:
            for attempt in event["step_resolution"]:
                expected = admissible(event["before"], attempt["measurement"], anchor)
                improved = attempt["measurement"]["macro"] > event["before"]["macro"]+1e-10
                if (expected != attempt["original_gate"] or improved != attempt["strict_macro_progress"]
                        or attempt["accepted"] != (expected and improved)):
                    raise AssertionError("Step-resolution bypassed its original independent gate")
                counted += 1
            last = event["step_resolution"][-1]
            if last["accepted"] != event["accepted"]:
                raise AssertionError("Incorrect transactional admission")
            if event["accepted"] and event["after"] != last["measurement"]:
                raise AssertionError("Credit differs from verified returned model")
            recovered += event["accepted"] and len(event["step_resolution"]) > 1
    session = ResolutionSession(parent, saved=state(RUN / f"{selected}-{STEPS-128:04d}.pt"))
    data, probe = load("train"), growth_data("probe")
    for _ in range(128):
        session.step(data, probe)
    expected = state(RUN / f"{selected}-{STEPS:04d}.pt")
    if not equal(session.state(), expected):
        raise AssertionError("Step-resolution continuation differs from retained checkpoint")
    if not all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items()):
        raise AssertionError("Unapproved inherited tensors changed")
    cases = []
    rng = np.random.default_rng(40981)
    for domain in ("integral", "sum"):
        for _ in range(32):
            p = rng.integers(-8, 9, 4).tolist()
            proposed = session.owner.study_maps[domain].propose(p)
            good = check(domain, p, proposed)["accepted"] and independent(domain, p, proposed)
            cases.append({"domain": domain, "input": p, "proposal": proposed, "accepted": good})
    if not all(c["accepted"] for c in cases):
        raise AssertionError("Retained exact mathematics failed")
    grounded = session.base.base.base.base.base.base.base.base.grounded
    if grounded.gate["autonomous"]:
        raise AssertionError("Source gate changed")
    result = {"contracts": contracts(), "owner": model_identity(session.owner), "admitted": final["admitted"],
              "verified_step_checks": counted, "recovered_proposals": recovered, "exact_resume_decisions": 128,
              "prediction_items": sum(r["items"] for r in reports),
              "maximum_loss_difference": max(r["maximum_loss_difference"] for r in reports),
              "protected_tensors": len(protected), "protected_equal": True, "mathematics": cases,
              "novel_definition_gate": grounded.gate["autonomous"], "same_actual_owner": session.owner is session.base.owner}
    if final["admitted"]:
        store = Store(ROOT / "runs/sera-step-resolution-live")
        if store.read() is not None:
            raise FileExistsError("Preserve an existing resolution owner")
        saved = session.snapshot()
        store.commit(saved, None)
        restored = ResolutionSession(parent, saved=expected)
        if model_identity(restored.owner) != saved["owner"]:
            raise AssertionError("Integrated resolution owner failed exact reload")
        result["store"] = "runs/sera-step-resolution-live"
    else:
        result["store"] = "runs/sera-growth-live"
    write(OUT / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contracts", "mathematics"}}, indent=2))


if __name__ == "__main__":
    main()
