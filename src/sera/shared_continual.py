"""Durable corrective learning through the same shared owner used for inference."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

from sera.accounting import Costs
from sera.binding import binding_cases
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy
from sera.shared import SharedR1, replace_shared_owner
from sera.shared_evaluation import binding_scores, evaluate_shared, shared_capabilities
from sera.shared_learning import SharedEvidence, adapt_shared
from sera.solver import SolverStore, Work
from sera.storage import digest, write_json
from sera.training import environment


def initialize_shared_store(root, solver, evidence, specs):
    root = Path(root)
    if root.exists():
        raise FileExistsError("Initialize a fresh shared solver directory to preserve previous work")
    store = SolverStore(root)
    solver.validate()
    if not isinstance(solver.components["r1"], SharedR1):
        raise ValueError("Shared continuation requires one common learned owner")
    evidence.save(root / "evidence" / "e0")
    write_json(root / "evidence-current.json", {"revision": "e0"})
    write_json(root / "worlds.json", [asdict(spec) for spec in specs])
    # The older world-learning interface can still read the admitted trajectories.
    evidence.world.save(root / "experience.json")
    return store.initialize(solver)


def learn_binding(root, *, seed=0, support_count=128, steps=192, method="replay", samples=1024):
    if method not in {"full", "replay", "adapter"}:
        raise ValueError("Ordinary binding learning needs full, replay or adapter updating")
    root = Path(root)
    store, costs, work = SolverStore(root), Costs(), Work()
    incumbent = store.load()
    if not isinstance(incumbent.components["r1"], SharedR1):
        raise ValueError("This historical solver has separate owners; initialize a shared solver first")
    pointer = json.loads((root / "evidence-current.json").read_text(encoding="utf-8"))
    evidence = SharedEvidence.load(root / "evidence" / pointer["revision"])
    specs = [WorldSpec(row["identifier"], tuple(tuple(r) for r in row["table"]), tuple(row["colors"]), row["resettable"])
             for row in json.loads((root / "worlds.json").read_text(encoding="utf-8"))]
    with store.writing():
        proposals = root / "learning"
        proposals.mkdir(exist_ok=True)
        index = max((int(p.name) for p in proposals.iterdir() if p.name.isdigit()), default=-1)+1
        output = proposals / str(index)
        output.mkdir()
    record = {"environment": environment(), "parent": incumbent.identity(), "seed": seed,
              "method": method, "rule": "earliest", "role": "ordinary retained corrective learning"}
    try:
        with costs.phase("corrective-evidence-and-diagnosis", work):
            support = binding_cases(seed=seed, count=support_count, split="support-continued")
            validation = binding_cases(seed=seed, count=128, split="validation-continued")
            record["diagnosis"], _ = binding_scores(incumbent.components["typed"], support, work)
            record["correction"] = "Use the explicitly requested first binding despite later different writes to the same key"
        with costs.phase("shared-owner-update", work):
            core, record["training"] = adapt_shared(incumbent.components["r1"], evidence, support, validation,
                method=method, seed=seed, steps=steps, work=work)
            candidate = copy.deepcopy(incumbent)
            replace_shared_owner(candidate, core)
        with costs.phase("fresh-independent-admission", work):
            record["admission"] = store.consider(candidate,
                lambda solver, fresh: evaluate_shared(solver, specs, seed=fresh, samples=samples, work=work),
                work=work, policy=AdmissionPolicy(max_candidate_cost=10_000_000),
                description={"method": method, "task": "instructed-earliest-binding", "support": support_count,
                             "steps": steps, "learning_record": str(index)},
                required_capabilities=shared_capabilities(incumbent, specs), expected_parent=incumbent.identity())
        with costs.phase("admit-corrective-replay", work):
            evidence.admit_typed(support)
            revision = "e" + digest([pointer, index, sorted(evidence.typed.identifiers)])[:16]
            with store.writing():
                if json.loads((root / "evidence-current.json").read_text(encoding="utf-8")) != pointer:
                    raise ValueError("Replay changed during learning; preserve the proposal and reconcile explicitly")
                evidence.save(root / "evidence" / revision)
                write_json(root / "evidence-current.json", {"revision": revision})
                store.journal.append("shared_corrective_evidence_admitted", {
                    "revision": revision, "parent_revision": pointer["revision"], "learning_record": index,
                    "records": len(evidence.typed.records)})
            record["evidence_revision"] = revision
        record["current"] = store.current_record()
        record["status"] = record["admission"]["status"]
    except Exception as error:
        record.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(costs=costs.record(), work=work.record())
        write_json(output / "learning.json", record)
    return record
