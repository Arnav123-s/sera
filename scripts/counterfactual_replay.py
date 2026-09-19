"""Audit the integrated owner, retained answers and background-generalized rules."""

import gc
import json
from fractions import Fraction as Q

import torch

from experiments.counterfactual_check import check_generalization, structural_world
from experiments.counterfactual_core import Z
from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.self_chosen import equations as eq
from scripts.counterfactual_qualification import QUALIFICATION, RUN, STORE, activate
from scripts.counterfactual_qualification import restore as restore_inquiry
from scripts.counterfactual_qualification import verify as verify_freeze
from scripts.counterfactual_study import persist
from scripts.solution_use import perform as prior_perform
from workbench.storage import Store

OUT = ROOT / "research-continuation/45_counterfactual_inquiry"


def main():
    torch.set_num_threads(1)
    activate()
    verify_freeze()
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve the completed owner audit")
    if not (QUALIFICATION / "final.json").exists():
        raise ValueError("Finish the frozen evaluation before qualification")
    final = read(QUALIFICATION / "final.json")
    selected = read(QUALIFICATION / "selection.json")["policy"]
    if final["comparison"][selected]["wrong_consensus"]:
        raise ValueError("The selected final procedure made wrong consensus claims; preserve the study without qualification")
    session, growth = restore_inquiry()
    inventory = read(OUT / "inventory.json")
    inherited = inventory["parameters"]
    current = session.owner.state_dict()
    changed = [k for k, v in inherited.items() if digest(current[k].detach().tolist()) != v["sha256"]]
    if changed:
        raise ValueError("An inherited learned tensor changed: " + str(changed))
    grounded = growth.base.base.base.base.base.base.grounded
    if grounded.owner is not session.owner or grounded.gate != inventory["novel_definition_gate"]:
        raise ValueError("Shared owner or source applicability gate changed")
    tasks = read(ROOT / "research-continuation/44_solution_portfolios/example-tasks.json")
    expected = read(ROOT / "research-continuation/44_solution_portfolios/example-results.json")
    actual = [prior_perform(session.base, growth, request) for request in tasks]
    def substance(value):
        if isinstance(value, list):
            return [substance(v) for v in value]
        if isinstance(value, dict):
            return {k: substance(v) for k, v in value.items() if k != "owner"}
        return value
    if substance(actual) != substance(expected):
        raise ValueError("Earlier practical task behavior changed")
    transfer, groups = [], {}
    for key, record in session.rules.items():
        claim = record["claim"]
        if not check_generalization(claim)["accepted"]:
            raise ValueError("Background-generalized proof changed")
        spec, target = claim["spec"], claim["target"]
        shape_id = digest(claim["shape"])
        groups.setdefault(shape_id, {"shape": claim["shape"], "members": []})["members"].append(
            {"rule": key, "domain": spec["model"]["domain"], "axis": spec["axis"], "target": target,
             "companion": spec["companion"], "power": spec["power"], "condition": claim["condition"]})
        rows = eq.reference_rows(spec["model"]["domain"], 45901 + int(key[:6], 16), count=8)
        for index, row in enumerate(rows):
            controls = {n: Q(row[n]) for n in spec["model"]["controls"]}
            world = structural_world(spec, controls, Z)
            initial = world[target].subs(Z, 1)
            if initial == 0:
                continue
            for factor in (Q(1, 2), Q(3, 2), Q(3)):
                changed_world = {n: v.subs(Z, factor) for n, v in world.items()}
                if any(v.is_finite is not True for v in changed_world.values()):
                    continue
                if changed_world["t"] <= 0 or "m" in changed_world and changed_world["m"] <= 0:
                    continue
                if spec["model"]["domain"] == "polynomials" and not changed_world["t"].is_integer:
                    continue
                answer = session.apply_rule(key, str(initial), str(factor))
                expected_value = str(changed_world[target])
                if Q(answer["conditional_value"]) != Q(expected_value):
                    raise ValueError("Retained rule failed background transfer")
                transfer.append({"rule": key, "context": index, "factor": str(factor), "value": expected_value, "correct": True})
    write(RUN / "background-transfer.json", transfer)
    connections = [v for v in groups.values() if len({m["domain"] for m in v["members"]}) > 1]
    write(OUT / "connections.json", connections)
    selected_record = next(iter(session.records.values()))
    original = read(ROOT / selected_record["artifact"]["path"])
    before = session.identity()
    duplicate_rejected = False
    try:
        session.retain(original, [], {}, {}, selected_record["artifact"])
    except ValueError as error:
        duplicate_rejected = "Repeated" in str(error)
    if not duplicate_rejected or before != session.identity():
        raise ValueError("Repeated credit changed owned weights")
    previous = Store(STORE).read()
    session.saturation = {"finished_questions": sorted(session.records), "next_action":
                          "Current 228-question dependency frontier completed; retain unresolved model gaps and expand representation or observation sources before repeating discovery credit"}
    persist(session, previous)
    snapshot, identity = session.state(), session.identity()
    history = Store(STORE).verify_history()
    summary = {"passed": True, "owner": identity, "parent_owner": session.parent_owner,
               "retained_parent_tensors": len(inherited), "changed_parent_tensors": changed,
               "shared_language_owner": True, "source_gate": grounded.gate,
               "questions": len(session.records), "rules": len(session.rules), "history_revisions": history,
               "earlier_task_results_retained": len(tasks), "background_transfer_checks": len(transfer),
               "cross_domain_shape_connections": len(connections), "repeated_credit_rejected": True,
               "points": sum(v["points"] for v in session.credits.values()),
               "trained_procedure_updates": len(session.training_receipts), "selected_policy": session.selected_policy,
               "new_tensor_records": len(current) - len(inherited), "study_final": sha(QUALIFICATION / "final.json")}
    del session, growth, current, grounded
    gc.collect()
    restored, growth = restore_inquiry()
    if restored.identity() != identity or restored.state() != snapshot:
        raise ValueError("Integrated state did not restore exactly")
    summary["exact_owner_restore"] = True
    summary["exact_investigation_replays"] = sum(1 for _ in RUN.glob("*/*/replay.json"))
    write(OUT / "audit.json", summary)
    from scripts.counterfactual_live import perform
    examples = [{"id": "smaller-and-larger-mass", "kind": "what_if", "domain": "motion_0", "axis": "m", "target": "a0", "multipliers": ["1/4", "1/2", "1", "2", "4"]},
                {"id": "changing-time", "kind": "what_if", "domain": "motion_1", "axis": "t", "target": "x", "multipliers": ["1/2", "1", "2"]},
                {"id": "discrete-accumulation", "kind": "what_if", "domain": "polynomials", "axis": "c1", "target": "s", "multipliers": ["1/2", "1", "2"]},
                {"id": "choose-next", "kind": "inquiry_next"}, *tasks]
    write(OUT / "example-tasks.json", examples)
    outputs = [perform(restored, growth, t) for t in examples]
    write(RUN / "example-results.json", outputs)
    summary["practical_examples"] = len(outputs)
    write(OUT / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
