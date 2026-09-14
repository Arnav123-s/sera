"""Repeat an ordinary shared-learning admission and verify resumed inference."""

import argparse
import json
from pathlib import Path

import torch

from sera.binding import binding_cases
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy, assess
from sera.shared_evaluation import evaluate_shared
from sera.shared_learning import SharedEvidence
from sera.solver import SolverStore
from sera.storage import canonical, write_json
from sera.training import source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("runs/sera-0.5-current"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old live audits")
    torch.set_num_threads(1)
    store = SolverStore(args.directory)
    latest = max((int(p.stem), p) for p in (args.directory / "rounds").glob("*.json"))[1]
    row = json.loads(latest.read_text(encoding="utf-8"))
    worlds = json.loads((args.directory / "worlds.json").read_text(encoding="utf-8"))
    specs = [WorldSpec(w["identifier"], tuple(tuple(r) for r in w["table"]), tuple(w["colors"]), w["resettable"]) for w in worlds]
    samples = row["candidate"]["binding"]["earliest"]["ordinary"]["examples"]
    reports, scores = [], []
    for version, original in ((row["parent"], row["incumbent"]), (row["version"], row["candidate"])):
        solver = store.load(version)
        assert solver.neural.owner is solver.components["typed"].owner is solver.components["r1"]
        report, values = evaluate_shared(solver, specs, seed=row["seed"], samples=samples)
        if canonical(report) != canonical(original):
            raise ValueError("Saved admission report did not reproduce exactly")
        reports.append(report)
        scores.append(values)
    expected = row["decision"]
    decision = assess(scores[1], scores[0], round_index=row["round_index"], invariants_ok=True,
                      candidate_cost=expected["candidate_cost"], policy=AdmissionPolicy(**expected["policy"]))
    if canonical(expected) != canonical(decision):
        raise ValueError("Fresh admission arithmetic changed")
    current = store.load()
    if current.version != (row["version"] if expected["admitted"] else row["parent"]):
        raise ValueError("Current pointer does not match the admission outcome")
    predictions = []
    for rule in ("earliest", "latest"):
        rows = binding_cases(seed=1_450_000, count=16, split="test-live", rule=rule)
        for example in rows:
            prediction = current.components["typed"].predict(example.observations, example.task)
            predictions.append({"rule": rule, "record": example.identifier, "target": example.target,
                                "prediction": prediction, "correct": prediction["class"] == example.target})
    pointer = json.loads((args.directory / "evidence-current.json").read_text(encoding="utf-8"))
    evidence = SharedEvidence.load(args.directory / "evidence" / pointer["revision"])
    result = {"passed": True, "source_sha256": source_hash(), "current": store.current_record(),
              "replayed_round": row["round_index"], "status": row["status"], "exact_report_replays": 2,
              "capabilities": len(scores[0].capabilities), "objective_samples": decision["samples"],
              "decision_reproduced": True, "parameter_owner_shared_after_reload": True,
              "typed_evidence_records": len(evidence.typed.records), "world_evidence_records": len(evidence.world.records),
              "ordinary_predictions": predictions, "journal_verified": store.journal.verify()}
    write_json(args.output, result)
    print(row["status"], current.version, "reproduced", len(scores[0].capabilities), "retained capabilities")


if __name__ == "__main__":
    main()
