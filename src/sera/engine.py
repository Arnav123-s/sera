"""Bounded failure-to-skill integration of the R1 and R2 research paths."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path

import numpy as np
import torch

from sera.data import make_batch, seed_for
from sera.evaluation import assess, evaluate
from sera.programs import BudgetExhausted, SkillLibrary, discover, verify_program
from sera.storage import Journal, digest, write_json
from sera.training import source_hash
from sera.world import FiniteWorld, world_experiment


def model_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def improve(model, output: Path, *, seed=0, samples=1024, max_queries=100):
    """Fixed diagnostic policy in v0.1; it is not a learned general improver.

    The candidate obtains resettable observable-state feedback unavailable in
    supervised pretraining. This intervention tests integration, not superiority
    at the same information budget.
    """
    output.mkdir(parents=True, exist_ok=True)
    journal = Journal(output / "journal.sqlite")
    journal.verify()
    if (output / "improvement.json").exists():
        raise FileExistsError("Improvement run already exists; choose a fresh directory")
    started = time.perf_counter()
    diagnosis, _ = evaluate(model, seed=seed, split="diagnosis", samples=256)
    score = diagnosis["tasks"]["ordered_control"]["accuracy"]
    journal.append(
        "diagnosis",
        {
            "kind": "missing_compositional_procedure" if score < 0.95 else "solved",
            "score": score,
            "policy": "fixed_v1",
            "dataset_id": diagnosis["dataset_id"],
        },
    )
    incumbent_hash = model_digest(model)
    incumbent = {"version": "v0", "neural_sha256": incumbent_hash, "skill_id": None}
    write_json(output / "versions" / "v0.json", incumbent)
    write_json(output / "current.json", incumbent)
    if score >= 0.95:
        result = {
            "status": "no_change",
            "reason": "diagnostic_task_already_solved",
            "diagnosis": diagnosis,
        }
        write_json(output / "improvement.json", result)
        return result
    world = FiniteWorld()
    try:
        discovery = discover(
            world.query,
            environment_id=world.environment_id,
            actions=world.actions,
            max_queries=max_queries,
        )
    except (BudgetExhausted, ValueError) as error:
        result = {"status": "unresolved", "reason": str(error), "diagnosis": diagnosis}
        journal.append("unresolved", result)
        write_json(output / "improvement.json", result)
        return result
    verification = verify_program(
        discovery.program, world.query, seed=seed_for("skill-verification", seed)
    )
    library = SkillLibrary(output / "skills")
    skill_id = library.admit(discovery, verification)
    candidate = {
        "version": "v1",
        "parent": "v0",
        "neural_sha256": incumbent_hash,
        "skill_id": skill_id,
    }
    write_json(output / "versions" / "v1.json", candidate)
    journal.append("candidate_frozen", {**candidate, "source_sha256": source_hash()})
    # The candidate is fixed before this fresh seed is generated. The evaluator
    # logs the seed after scoring for reproducibility, never supplies it to discovery.
    evaluation_seed = secrets.randbits(63)
    split = "promotion-fresh"
    dataset_id = digest(
        [
            make_batch(samples, 24, evaluation_seed, split=split, index=t, task=t).dataset_id
            for t in range(4)
        ]
    )
    round_index = journal.reserve_evaluation(dataset_id)
    baseline, baseline_scores = evaluate(
        model, seed=evaluation_seed, split=split, length=24, samples=samples
    )
    after, candidate_scores = evaluate(
        model,
        seed=evaluation_seed,
        split=split,
        length=24,
        samples=samples,
        program=library.load(skill_id),
    )
    if baseline["dataset_id"] != dataset_id or after["dataset_id"] != dataset_id:
        raise RuntimeError("Paired evaluation identity mismatch")
    invariants_ok = verification["passed"] and model_digest(model) == incumbent_hash
    decision = assess(
        candidate_scores,
        baseline_scores,
        round_index=round_index,
        invariants_ok=invariants_ok,
        candidate_cost=discovery.executed_actions + discovery.oracle_queries,
    )
    np.savez_compressed(
        output / "paired_scores.npz",
        **{
            f"{side}_{task}": values
            for side, scores in (("incumbent", baseline_scores), ("candidate", candidate_scores))
            for task, values in scores.items()
        },
    )
    if decision["admitted"]:
        write_json(output / "current.json", candidate)
    result = {
        "status": "promoted" if decision["admitted"] else "rejected",
        "diagnosis": diagnosis,
        "incumbent": baseline,
        "candidate": after,
        "decision": decision,
        "skill_id": skill_id,
        "verification": verification,
        "oracle_queries": discovery.oracle_queries,
        "executed_actions": discovery.executed_actions,
        "evaluation_seed": evaluation_seed,
        "dataset_id": dataset_id,
        "seconds": time.perf_counter() - started,
        "limitations": [
            "fixed diagnostic and routing policy",
            "observable resettable finite-state world",
            "extra oracle feedback compared with neural pretraining",
            "empirical retention gate",
            "local ledger is not security isolation",
        ],
    }
    journal.append("promotion_decision", result)
    journal.verify()
    write_json(output / "improvement.json", result)
    return result


def rollback(output: Path):
    current = json.loads((output / "current.json").read_text(encoding="utf-8"))
    parent = current.get("parent")
    if parent is None:
        raise ValueError("Current version has no parent to roll back to")
    if not parent.startswith("v") or not parent[1:].isdigit():
        raise ValueError("Invalid parent version")
    previous = json.loads((output / "versions" / f"{parent}.json").read_text(encoding="utf-8"))
    journal = Journal(output / "journal.sqlite")
    journal.verify()
    journal.append("rollback", {"from": current["version"], "to": previous["version"]})
    write_json(output / "current.json", previous)
    return previous


def run_world(output: Path, *, seed=0, steps=300):
    model, result = world_experiment(seed, steps)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "states": model.states,
            "actions": model.actions,
            "state_dict": model.state_dict(),
        },
        output / "world.pt",
    )
    write_json(output / "world.json", result)
    return result
