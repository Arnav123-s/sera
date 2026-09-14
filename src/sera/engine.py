"""Bounded failure-to-skill integration of the R1 and R2 research paths."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from sera.data import make_batch, seed_for
from sera.evaluation import evaluate
from sera.programs import BudgetExhausted, SkillLibrary, discover, verify_program
from sera.storage import Journal, write_json
from sera.world import FiniteWorld, world_experiment


def model_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def improve(model, output: Path, *, seed=0, samples=1024, max_queries=100, task=2):
    """Acquire a verified skill against the current executable incumbent.

    The legacy symbolic diagnostic is an explicit fixed control. The connected
    learner trains its intervention policy separately and persists it in the solver.
    """
    from dataclasses import asdict

    from sera.skills import acquire_rule
    from sera.solver import SolverStore, Work

    store = SolverStore(output)
    store.initialize(model)
    current = store.load()
    started = time.perf_counter()
    work = Work()
    diagnosis, _ = evaluate(current, seed=seed, split="diagnosis", samples=256, tasks=range(5))
    from sera.data import TASKS
    score = diagnosis["tasks"][TASKS[task]]["accuracy"]
    work.add("diagnosis_examples", 256 * 5)
    store.journal.append("diagnosis", {"task": TASKS[task], "score": score,
                                      "version": current.version, "policy": "fixed_symbolic_control"})
    if score >= 0.95:
        result = {"status": "no_change", "reason": "diagnostic_task_already_solved",
                  "diagnosis": diagnosis, "current": current.version, "work": work.record()}
        store.journal.append("no_change", result)
        return result
    candidate = store.load()
    try:
        if task == 2:
            world = FiniteWorld()
            def query(actions):
                work.add("discovery_oracle_queries")
                work.add("discovery_oracle_actions", len(actions))
                return world.query(actions)
            discovery = discover(query, environment_id=world.environment_id,
                                 actions=world.actions, max_queries=max_queries)
            def verify_query(actions):
                work.add("verification_oracle_queries")
                work.add("verification_oracle_actions", len(actions))
                return world.query(actions)
            verification = verify_program(discovery.program, verify_query,
                                          seed=seed_for("skill-verification", seed))
            library = SkillLibrary(output / "skills")
            skill_id = library.admit(discovery, verification)
            candidate.skills[str(task)] = {"kind": "transition", "skill_id": skill_id,
                                          "program": asdict(discovery.program)}
            details = {"skill_id": skill_id, "oracle_queries": discovery.oracle_queries,
                       "executed_actions": discovery.executed_actions, "verification": verification}
        else:
            support = make_batch(128, 12, seed, split="rule-support", task=task)
            verification = make_batch(256, 24, seed, split="rule-verification", task=task)
            work.add("support_examples", 128)
            work.add("verification_examples", 256)
            candidate.skills[str(task)] = acquire_rule(support, verification)
            details = {"rule": candidate.skills[str(task)]["rule"]}
    except (BudgetExhausted, ValueError) as error:
        work.seconds = time.perf_counter() - started
        result = {"status": "unresolved", "reason": str(error), "diagnosis": diagnosis,
                  "work": work.record()}
        store.journal.append("unresolved", result)
        write_json(output / "improvement.json", result)
        return result
    work.seconds = time.perf_counter() - started
    def evaluator(solver, fresh_seed):
        return evaluate(solver, seed=fresh_seed, split="promotion-fresh-v2", length=24,
                        samples=samples, tasks=range(5))
    result = store.consider(candidate, evaluator, work=work,
                            description={"task": TASKS[task], "method": "verified_program"})
    result.update(details)
    result["diagnosis"] = diagnosis
    write_json(output / "improvement.json", result)
    return result


def rollback(output: Path):
    from sera.solver import SolverStore
    record = json.loads((output / "current.json").read_text(encoding="utf-8"))
    if record.get("schema_version") in {2, 3}:
        return SolverStore(output).rollback()
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
