"""Restore every shared candidate and independently repeat its saved measurements."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs
from sera.binding import binding_cases
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy, CapabilityScores, assess
from sera.shared import IndependentTypedControl, make_shared_solver
from sera.shared_archive import load_shared_checkpoint
from sera.shared_evaluation import evaluate_shared, score_record
from sera.shared_learning import SharedEvidence, restore_typed
from sera.solver import tensor_digest
from sera.storage import canonical, write_json
from sera.training import source_hash
from sera.typed_protocol import audit_partitions


def read_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def arrays(row):
    return CapabilityScores({k: np.asarray(v) for k, v in row["objectives"].items()},
        {k: np.asarray(v) for k, v in row["capabilities"].items()}, row["dataset_ids"])


def compare(saved, restored):
    if saved["dataset_ids"] != restored["dataset_ids"]:
        raise ValueError("Repeated inference used different retained cases")
    maximum = 0.
    for group in ("objectives", "capabilities"):
        if saved[group].keys() != restored[group].keys():
            raise ValueError("Repeated inference omitted outputs")
        for name in saved[group]:
            a, b = np.asarray(saved[group][name]), np.asarray(restored[group][name])
            if a.shape != b.shape:
                raise ValueError("Score-vector lengths changed")
            maximum = max(maximum, float(np.abs(a-b).max()))
    if maximum > 1e-7:
        raise ValueError(f"Checkpoint outputs changed: {maximum}")
    return maximum


def audit_trial(root):
    data = json.loads((root / "trial.json").read_text(encoding="utf-8"))
    if data["status"] != "completed" or data["environment"]["source_sha256"] != source_hash():
        raise ValueError("Trial is incomplete or belongs to a different executable source")
    spec = WorldSpec(data["world"]["identifier"], tuple(tuple(r) for r in data["world"]["table"]),
                     tuple(data["world"]["colors"]), data["world"]["resettable"])
    saved = read_gzip(root / "scores.json.gz")
    evidence = SharedEvidence.load(root / "evidence")
    validation = read_gzip(root / "validation.json.gz")
    adaptation = read_gzip(root / "adaptation-evidence.json.gz")
    queries = [r for rule in ("earliest", "latest") for family in ("ordinary", "long", "composition")
        for r in binding_cases(seed=data["budget"].get("query_seed_base", 1_000_000)+data["seed"], count=data["budget"]["samples"], rule=rule,
                                family=family, split="promotion-shared", unique=False)]
    partitions = audit_partitions({"support": evidence.typed.records+[restore_typed(r) for r in adaptation["support"]],
        "validation": [restore_typed(r) for r in validation["typed"]+validation["latest"]+adaptation["validation"]],
        "query": queries})
    base = load_shared_checkpoint(root / "base.pt")
    assert tensor_digest(base) == data["base_tensor_sha256"]
    assert base.core_state_bytes() == data["core_state_bytes"] == (35840 if data["kind"] == "reference" else 32768)
    checks, artifacts = [], []
    for path in sorted(root.rglob("*.pt")):
        artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for item in [{"label": "none", "method": "none"}, *data["runs"]]:
        label, method = item["label"], item["method"]
        if method == "none":
            core = copy.deepcopy(base)
        else:
            selected = item.get("checkpoint_source", label)
            path = root / f"{selected}.pt"
            core = load_shared_checkpoint(path)
            if "checkpoint" in item and hashlib.sha256(path.read_bytes()).hexdigest() != item["checkpoint"]["sha256"]:
                raise ValueError("Checkpoint artifact changed")
            if "tensor_sha256" in item and tensor_digest(core) != item["tensor_sha256"]:
                raise ValueError("Reconstructed parameter identity changed")
        solver = make_shared_solver(copy.deepcopy(base) if method == "separate" else core)
        if method == "separate":
            solver.components["typed"] = IndependentTypedControl(core)
        solver.validate()
        report, scores = evaluate_shared(solver, [spec], seed=data["budget"].get("query_seed_base", 1_000_000)+data["seed"], samples=data["budget"]["samples"],
            retained_samples=data["budget"]["retained_samples"], typed_samples=data["budget"]["typed_samples"])
        difference = compare(saved[label], score_record(scores))
        original = data["baseline"] if method == "none" else item["evaluation"]
        if canonical(report) != canonical(original):
            raise ValueError(f"Repeated report changed for {label}, including neural/calibration outputs")
        if method != "none":
            expected = item["retention_audit"]
            decision = assess(arrays(saved[label]), arrays(saved["none"]), round_index=0, invariants_ok=True,
                              candidate_cost=0, policy=AdmissionPolicy(**expected["policy"]))
            if canonical(expected) != canonical(decision):
                raise ValueError("Retrospective decision arithmetic changed")
        checks.append({"label": label, "maximum_score_difference": difference, "passed": True})
        print(root.name, data["seed"], label, "replayed", flush=True)
    return {"seed": data["seed"], "kind": data["kind"], "passed": True, "source_sha256": source_hash(),
            "partitions": partitions, "checkpoints": artifacts, "replay_checks": checks,
            "decision_arithmetic_checks": len(data["runs"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/shared-learner-repaired"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kind", choices=("delta", "reference"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier verification results")
    torch.set_num_threads(1)
    costs = Costs()
    checks = [audit_trial(args.root / str(args.seed) / kind) for kind in ((args.kind,) if args.kind else ("delta", "reference"))]
    write_json(args.output, {"passed": True, "source_sha256": source_hash(), "trials": checks, "costs": costs.record()})


if __name__ == "__main__":
    main()
