"""Reload checkpoints and independently check the new evaluation evidence."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from sera.capability_evaluation import evaluate_capabilities
from sera.connected import restore_component
from sera.contracts import EvidenceKind, Observation, Provenance
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy, assess
from sera.session_state import model_identity
from sera.solver import SolverStore
from sera.storage import write_json
from sera.training import source_hash
from sera.typed_learning import TypedExample, score_typed, typed_examples
from sera.typed_protocol import audit_partitions, semantic_id, typed_suite


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def provenance(row):
    return Provenance(row["source"], row["record_id"], EvidenceKind(row["kind"]))


def restore_row(row):
    observations = []
    for obs in row["observations"]:
        values = dict(obs)
        values["values"] = tuple(values["values"])
        values["available"] = tuple(values["available"]) if values["available"] is not None else None
        values["provenance"] = provenance(values["provenance"])
        observations.append(Observation(**values))
    target = tuple(row["target"]) if row["task"] == "motion" else row["target"]
    return TypedExample(tuple(observations), row["task"], target, provenance(row["evidence"]), row["split"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("runs/evaluation-v2-complete"))
    parser.add_argument("--frozen", type=Path, default=Path("runs/stage-three-complete"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old verification records")
    torch.set_num_threads(1)
    summary = read(args.study / "summary.json")
    if summary["status"] != "completed" or summary["environment"]["source_sha256"] != source_hash():
        raise ValueError("Incomplete study or changed source")
    for row in summary["artifacts"]:
        if hashlib.sha256((args.study / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed artifact: {row['path']}")
    checks = []
    reexecuted = []
    for trial in summary["typed"]:
        seed = trial["seed"]
        root = args.study / str(seed)
        suite = {p: [restore_row(r) for r in rows] for p, rows in json.loads(gzip.decompress((root / "typed-data.json.gz").read_bytes())).items()}
        audited = audit_partitions(suite)
        if audited["partitions"] != trial["manifest"]["partitions"]:
            raise ValueError("Archived semantic identities differ")
        old = typed_examples(seed=210000 + seed, count=192) + typed_examples(seed=220000 + seed, count=48, split="validation")
        if {semantic_id(r) for r in old} & {semantic_id(r) for rows in suite.values() for r in rows}:
            raise ValueError("Frozen teaching data entered the v2 suite")
        _, regenerated = typed_suite(seed=810000 + seed, test_count=summary["configuration"]["typed_test_count"], exclude=old)
        if regenerated != trial["manifest"]:
            raise ValueError("Fresh generation did not reproduce the saved partitions")
        store = SolverStore(args.frozen / str(seed) / "solver")
        frozen = store.load().components["typed"]
        payload = torch.load(root / "typed-v2.pt", weights_only=True, map_location="cpu")
        fresh = restore_component(payload["config"])
        fresh.load_state_dict(payload["state"])
        if model_identity(fresh) != trial["new_model_identity"] or model_identity(frozen) != trial["frozen_model_identity"]:
            raise ValueError("Variant model identity differs")
        saved = json.loads(gzip.decompress((root / "typed-scores.json.gz").read_bytes()))
        for name, model, programs in (("typed-v1-neural", frozen, False), ("typed-v1-procedural", frozen, True),
                                       ("typed-v2-neural", fresh, False), ("typed-v2-procedural", fresh, True)):
            for partition in ("test-id", "test-extent", "test-composition"):
                report, vectors = score_typed(model, suite[partition], use_programs=programs, return_scores=True)
                error = max(float(np.max(np.abs(v - np.asarray(saved[name][partition][task])))) for task, v in vectors.items())
                if error > 1e-7 or any(abs(v["score"] - trial["results"][name][partition]["tasks"][task]["score"]) > 1e-7 for task, v in report["tasks"].items()):
                    raise ValueError("Frozen checkpoint behavior did not reproduce")
                checks.append({"seed": seed, "variant": name, "partition": partition, "maximum_score_difference": error})
        chosen = next(row for row in reversed(summary["retention"]) if row["seed"] == seed)
        original = read(store.root / "rounds" / f"{chosen['round']}.json")
        worlds = {w["identifier"]: WorldSpec(w["identifier"], tuple(tuple(r) for r in w["table"]), tuple(w["colors"]), w["resettable"])
                  for w in read(store.root / "worlds.json")}
        specs = [worlds[k] for k in original["incumbent"]["tasks"]]
        n = next(iter(original["incumbent"]["tasks"].values()))["prediction"]["episodes"]
        _, before = evaluate_capabilities(store.load(chosen["parent"]), specs, seed=original["seed"], samples=n)
        _, after = evaluate_capabilities(store.load(chosen["candidate"]), specs, seed=original["seed"], samples=n)
        decision = assess(after, before, round_index=chosen["round"], invariants_ok=True,
                          candidate_cost=chosen["decision_v2"]["candidate_cost"], policy=AdmissionPolicy(**chosen["decision_v2"]["policy"]))
        if decision != chosen["decision_v2"]:
            raise ValueError("Full capability decision did not reproduce")
        reexecuted.append({"seed": seed, "round": chosen["round"], "capabilities": len(before.capabilities)})
        print(f"seed {seed}: semantic data, all four typed routes, and paired capability evaluation reproduced", flush=True)
    for row in summary["retention"]:
        d = row["decision_v2"]
        policy = d["policy"]
        bound = d["mean_gain"] - math.sqrt(-2 * d["log_alpha"] / d["samples"])
        failed = sorted(k for k, v in d["retention_loss_by_capability"].items() if v > policy["max_retention_loss"])
        expected = (bound > policy["minimum_gain"] and not failed
                    and max(d["retention_loss_by_task"].values()) <= policy["max_retention_loss"]
                    and d["candidate_cost"] <= policy["max_candidate_cost"])
        if abs(bound - d["gain_lower_bound"]) > 1e-12 or expected != d["admitted"] or failed != d["failed_capabilities"]:
            raise ValueError("Independent decision arithmetic differs")
    result = {"passed": True, "source_sha256": source_hash(), "artifact_hashes_checked": len(summary["artifacts"]),
              "typed_checkpoint_checks": checks, "complete_paired_decisions_reexecuted": reexecuted,
              "decision_arithmetic_checks": len(summary["retention"]),
              "semantic_partitions_checked": len(summary["typed"]) * 5,
              "old_teaching_overlap": 0, "scope": "Exact checkpoint replay and semantic regeneration; no extra training seeds or new admissions."}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
