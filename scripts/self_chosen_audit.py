"""Recheck committed conjectures, independent evidence, rewards, resumes and ownership."""

import copy
import json

import torch

from experiments.continuing_growth.data import load as growth_data
from experiments.continuing_growth.model import measure
from experiments.self_chosen import equations, language
from experiments.self_chosen.common import (
    ARMS,
    DOMAINS,
    OUT,
    RUN,
    SEEDS,
    STEPS,
    digest,
    read,
    sha,
    write,
)
from experiments.self_chosen.runtime import Session, reward_for
from scripts.refinement_audit import equal


def validate(saved, name):
    records, seen, checks = [], set(), 0
    for index, event in enumerate(saved["events"]):
        if event["sequence"] != index or event["id"] in seen or digest({k: v for k, v in event.items() if k != "id"}) != event["id"]:
            raise ValueError("Changed, duplicated or reordered investigation")
        pending = read(RUN / "proposals" / f"{name}-{index:04d}.json")
        if digest({k: v for k, v in pending.items() if k != "commitment"}) != pending["commitment"]:
            raise ValueError("Changed pre-assessment conjecture")
        if any(event[k] != v for k, v in pending.items()) or event["parent_owner"] != saved["parent_owner"]:
            raise ValueError("A different goal, predictor or conjecture received credit")
        total = 0.
        for p, outcome in zip(event["proposals"], event["receipts"], strict=True):
            q = event["question"]
            receipt = equations.certify(q["domain"], p) if q["domain"] in DOMAINS[:4] else language.assess(q, p)
            if receipt != outcome["receipt"]:
                raise ValueError("Independent evidence changed")
            expected = reward_for(q, p, receipt, records)
            if expected != outcome["credit"]:
                raise ValueError("Stale, repeated or unsupported discovery credit")
            if expected["retain"]:
                identifier = digest([pending["commitment"], p, receipt])
                matches = [r for r in saved["records"] if r["id"] == identifier]
                if len(matches) != 1:
                    raise ValueError("Missing or repeated retained executable relationship")
                record = matches[0]
                if record["proposal"] != p or record["receipt"] != receipt or record["question"] != q:
                    raise ValueError("Retained rule changed after qualification")
                records.append(record)
            total += expected["total"]
            checks += 1
        if total != event["reward"]:
            raise ValueError("Reward total changed")
        seen.add(event["id"])
    if records != saved["records"]:
        raise ValueError("Uncredited knowledge was inserted into the retained registry")
    return checks


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve the completed discovery audit")
    final = read(OUT / "final.json")
    counts = {}
    for seed in SEEDS:
        for arm in ARMS:
            name = f"{arm}-{seed}"
            counts[name] = validate(torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True), name)
    parent = read(RUN / "parent.json")
    session = Session(parent, saved=torch.load(RUN / "taught.pt", weights_only=True))
    protected = {k: v.detach().clone() for k, v in session.owner.state_dict().items()
                 if not k.startswith(("self_question_policy.", "self_discoveries.", "discovery_action."))}
    probe = growth_data("probe")
    original = measure(session.owner, probe)
    replayed = 0
    for arm in ARMS:
        name = f"{arm}-4101"
        initial = torch.load(RUN / f"{name}-{STEPS-32:04d}.pt", weights_only=True)
        session.restore(initial)
        def commit(pending):
            if read(RUN / "proposals" / f"{name}-{pending['sequence']:04d}.json") != pending:
                raise AssertionError("Restarted choice/conjecture differs before assessment")
        for _ in range(32):
            session.step(arm, commit)
            replayed += 1
        if not equal(session.state(), torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True)):
            raise AssertionError("Investigator state did not resume exactly")
    chosen = torch.load(RUN / f"{final['selected']}-{STEPS:04d}.pt", weights_only=True)
    session.restore(chosen)
    if not all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items()):
        raise AssertionError("Earlier retained weights changed")
    if measure(session.owner, probe) != original:
        raise AssertionError("The inherited twelve-strand predictions changed")
    executions = []
    for record in session.records:
        domain, target = record["question"]["domain"], record["proposal"]["target"]
        if domain not in DOMAINS[:4]:
            continue
        rows = equations.reference_rows(domain, 41983, 3)
        for row in rows:
            usable = {k: str(v) for k, v in row.items() if k != target}
            result = session.solve(domain, target, usable)
            if result["routes"] and any(equations.Q(r["result"]) != row[target] for r in result["routes"]):
                raise AssertionError("Owned retained coefficients failed actual task execution")
            executions.append({"rule": record["id"], "expected": str(row[target]), "result": result})
    growth = session.base
    grounded = growth.base.base.base.base.base.base.grounded
    if grounded.gate["autonomous"]:
        raise AssertionError("Closed novel-definition source gate changed")
    example = copy.deepcopy(chosen)
    example["events"][0]["reward"] += 1
    try:
        validate(example, final["selected"])
    except ValueError:
        tamper_rejected = True
    else:
        raise AssertionError("Forged points were accepted")
    result = {"passed": True, "owner": session.identity(), "independent_proposals": counts,
              "proposal_checks": sum(counts.values()), "exact_resume_decisions": replayed,
              "protected_tensors": len(protected), "protected_equal": True, "twelve_strand_predictions_unchanged": True,
              "source_gate": grounded.gate["autonomous"], "same_actual_owner": session.owner is session.base.owner,
              "tamper_rejected": tamper_rejected, "fresh_owned_route_tasks": len(executions), "executions": executions,
              "selected_checkpoint": sha(RUN / f"{final['selected']}-{STEPS:04d}.pt")}
    write(OUT / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "executions"}, indent=2))


if __name__ == "__main__":
    main()
