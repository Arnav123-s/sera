"""Independent receipt, novelty, composition, restart and owner audit for CF-001."""

import copy
import json
from fractions import Fraction as Q

import torch

from experiments.continuing_growth.data import load as growth_data
from experiments.continuing_growth.model import measure
from experiments.discovery_frontier import (
    DOMAINS,
    OUT,
    RUN,
    RUNS,
    STEPS,
    FrontierSession,
    assess,
    credit,
    digest,
    read,
    sha,
    write,
)
from experiments.self_chosen import equations as eq
from experiments.verified_completion.credit import decode
from scripts.refinement_audit import equal


def validate(saved, name, parent):
    records = copy.deepcopy(decode(parent["state"])["records"])
    count, compositions = 0, 0
    for index, event in enumerate(saved["inner"]["events"]):
        pending = read(RUN / "proposals" / f"{name}-{index:04d}.json")
        if event["sequence"] != index or digest({k: v for k, v in pending.items() if k != "commitment"}) != pending["commitment"]:
            raise ValueError("Changed or reordered pre-check commitment")
        if any(event[k] != v for k, v in pending.items()) or event["parent_owner"] != parent["owner"]:
            raise ValueError("Credit bound to a different predictor or goal")
        if digest({k: v for k, v in event.items() if k != "id"}) != event["id"]:
            raise ValueError("Changed investigation receipt")
        total = 0.
        for p, outcome in zip(event["proposals"], event["receipts"], strict=True):
            receipt = assess(event["question"], p)
            expected = credit(event["question"], p, receipt, records)
            if outcome != {"receipt": receipt, "credit": expected}:
                raise ValueError("Changed independent reward or repeated credit")
            if "dependencies" in p:
                if not set(p["dependencies"]) <= {r["id"] for r in records}:
                    raise ValueError("A composed rule depends on future or unqualified knowledge")
                compositions += 1
            if expected["retain"]:
                identifier = digest([pending["commitment"], p, receipt])
                matches = [r for r in saved["inner"]["records"] if r["id"] == identifier]
                if len(matches) != 1 or matches[0]["proposal"] != p or matches[0]["receipt"] != receipt:
                    raise ValueError("Retained coefficients lost their independent source")
                records.append(matches[0])
            total += expected["total"]
            count += 1
        if total != event["reward"]:
            raise ValueError("Changed total progress reward")
    if records != saved["inner"]["records"]:
        raise ValueError("Uncredited records were inserted")
    return {"proposals": count, "composition_candidates": compositions}


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve the completed frontier audit")
    parent, final = read(RUN / "parent.json"), read(OUT / "final.json")
    counts = {}
    for arm, seed in RUNS:
        name = f"{arm}-{seed}"
        counts[name] = validate(torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True), name, parent)
    session = FrontierSession(parent)
    protected = {k: v.detach().clone() for k, v in session.owner.state_dict().items()
                 if not k.startswith("self_question_policy.")}
    probe = growth_data("probe")
    before = measure(session.owner, probe)
    name = final["selected"]
    saved = torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True)
    session.restore(torch.load(RUN / f"{name}-{STEPS-64:04d}.pt", weights_only=True))
    def commit(pending):
        if read(RUN / "proposals" / f"{name}-{pending['sequence']:04d}.json") != pending:
            raise AssertionError("Restart altered a choice or candidate")
    for _ in range(64):
        session.step(name.rsplit("-", 1)[0], commit)
    if not equal(session.state(), saved):
        raise AssertionError("Frontier continuation was not exact")
    if not all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items()) or measure(session.owner, probe) != before:
        raise AssertionError("Earlier knowledge or twelve-strand predictions changed")
    tasks = []
    for record in session.records[session.parent_count:]:
        domain, target = record["question"]["domain"], record["proposal"]["target"]
        if domain not in DOMAINS[:4]:
            continue
        for row in eq.reference_rows(domain, 42983, 3):
            result = session.solve(domain, target, {k: str(v) for k, v in row.items() if k != target})
            if any(Q(r["result"]) != row[target] for r in result["routes"]):
                raise AssertionError("Composed owned execution failed independent values")
            tasks.append({"rule": record["id"], "expected": str(row[target]), "result": result})
    old_ids = set(read(RUN / "sources.json")["excluded_sd_ids"])
    private = read(RUN / "language-private.json")
    if any(r["id"] in old_ids for groups in private.values() for rows in groups.values() for r in rows):
        raise AssertionError("An earlier sealed human group was reused")
    growth = session.base
    source_gate = growth.base.base.base.base.base.base.grounded.gate["autonomous"]
    if source_gate:
        raise AssertionError("Novel-definition source gate changed")
    bad = copy.deepcopy(saved)
    bad["inner"]["events"][0]["reward"] += 1
    try:
        validate(bad, name, parent)
    except ValueError:
        pass
    else:
        raise AssertionError("Forged credit was accepted")
    result = {"passed": True, "owner": session.identity(), "checks": counts,
              "proposal_checks": sum(r["proposals"] for r in counts.values()), "exact_resume_decisions": 64,
              "protected_tensors": len(protected), "inherited_equal": True, "twelve_strand_predictions_unchanged": True,
              "fresh_owned_route_tasks": len(tasks), "tasks": tasks, "old_human_groups_excluded": True,
              "source_gate": source_gate, "same_actual_owner": session.owner is session.base.owner,
              "selected_checkpoint": sha(RUN / f"{name}-{STEPS:04d}.pt")}
    write(OUT / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
