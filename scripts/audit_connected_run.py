"""Reproduce exposed paired decisions and audit persistent behavior without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sera.connected import evaluate_worlds, execute_goal
from sera.data import TASKS
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy, assess, evaluate
from sera.experience import EvidenceReplay
from sera.solver import SolverStore
from sera.storage import write_json
from sera.training import source_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root):
    started = time.perf_counter()
    summary = read(root / "summary.json")
    if source_hash() != summary["manifest"]["environment"]["source_sha256"]:
        raise ValueError("Audit source differs from the original study")
    seen_support, seen_query, records = set(), set(), []
    for run in summary["runs"]:
        directory = root / str(run["seed"])
        store = SolverStore(directory / "solver")
        current = store.load()
        assert current.identity() == run["current_identity"]
        worlds = {r["identifier"]: WorldSpec(r["identifier"], tuple(tuple(x) for x in r["table"]),
                                             tuple(r["colors"]), r["resettable"])
                  for r in read(store.root / "worlds.json")}
        for path in directory.glob("*evidence.json"):
            EvidenceReplay.load(path)
        episodes = [read(p) for p in sorted((directory / "policy-episodes").glob("*.json"))]
        choices = {r["episode_id"]: r for r in run["policy_test"]["cases"]}
        for episode in episodes:
            ids = set(episode["support_record_ids"])
            assert not seen_support.intersection(ids), "Reused meta support records"
            assert episode["query_dataset_id"] not in seen_query, "Reused meta query dataset"
            seen_support.update(ids)
            seen_query.add(episode["query_dataset_id"])
            assert episode["query_dataset_id"] != episode["support_dataset_id"]
            if episode["episode_id"] in choices:
                choice, predicted = current.components["controller"].choose(
                    episode["features"], resettable=episode["world"]["resettable"])
                assert choice == choices[episode["episode_id"]]["chosen"]
                for method, value in predicted.items():
                    assert abs(value - choices[episode["episode_id"]]["prediction"][method]) < 1e-6
        rounds = []
        for path in sorted((store.root / "rounds").glob("*.json"), key=lambda p: int(p.stem)):
            original = read(path)
            assert original["status"] in {"promoted", "rejected"}
            incumbent, candidate = store.load(original["parent"]), store.load(original["version"])
            count = original["decision"]["samples"] // len(original["incumbent"]["tasks"])
            symbolic = "macro_accuracy" in original["incumbent"]
            def evaluator(solver):
                if symbolic:
                    return evaluate(solver, seed=original["seed"], split="promotion-fresh-v2",
                                    length=original["incumbent"]["length"], samples=count,
                                    tasks=[TASKS.index(name) for name in original["incumbent"]["tasks"]])
                return evaluate_worlds(solver, [worlds[name] for name in original["incumbent"]["tasks"]],
                                       seed=original["seed"], samples=count)
            before_report, before = evaluator(incumbent)
            after_report, after = evaluator(candidate)
            assert before_report["dataset_id"] == after_report["dataset_id"] == original["dataset_id"]
            decision = assess(after, before, round_index=original["round_index"], invariants_ok=True,
                              candidate_cost=original["decision"]["candidate_cost"],
                              policy=AdmissionPolicy(**original["decision"]["policy"]))
            for field in ("mean_gain", "gain_lower_bound"):
                assert abs(decision[field] - original["decision"][field]) < 1e-9
            assert decision["reasons"] == original["decision"]["reasons"]
            assert decision["admitted"] == (original["status"] == "promoted")
            arrays = {f"{prefix}/{task}": value
                      for prefix, data in (("before", before), ("after", after))
                      for task, value in data.items()}
            output = directory / "verification" / f"paired-{path.stem}.npz"
            output.parent.mkdir(exist_ok=True)
            np.savez_compressed(output, **arrays)
            rounds.append({"round": original["round_index"], "status": original["status"],
                           "reproduced": True, "dataset_id": original["dataset_id"],
                           "paired_scores_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                           "paired_samples": decision["samples"]})
        # Public execution after loading the accepted solver, including acquired world skills.
        examples = []
        for record in current.skills.values():
            if record.get("kind") == "action_program":
                result = execute_goal(current, worlds[record["world_id"]], record["start"], record["goal"])
                assert result["used_skill"] and result["success"]
                examples.append(result)
        final, _ = evaluate(current, seed=run["seed"] + 5, split="final-symbolic", samples=512,
                            tasks=range(5))
        assert final == run["symbolic_final"]
        records.append({"seed": run["seed"], "rounds": rounds,
                        "meta_episodes": len(episodes), "verified_goal_skills": len(examples),
                        "execution_examples": examples[:3], "solver_identity": current.identity(),
                        "component_parameters": {name: sum(p.numel() for p in module.parameters())
                                                 for name, module in current.components.items()},
                        "symbolic_parameters": sum(p.numel() for p in current.neural.parameters()),
                        "r1_core_state_bytes": sum(x.numel() * x.element_size()
                                                   for x in current.components["r1"].initial(1).values()),
                        "symbolic_final_reproduced": True})
        print(f"Audited seed {run['seed']}: {len(rounds)} decisions and {len(examples)} executable skills", flush=True)
    result = {"scope": "Reproduction of exposed results; no new training or generalization claim",
              "source_sha256": source_hash(), "runs": records,
              "unique_meta_support_records": len(seen_support),
              "unique_meta_query_datasets": len(seen_query),
              "additional_audit_wall_seconds": time.perf_counter() - started}
    write_json(root / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.input)
