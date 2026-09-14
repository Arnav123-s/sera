"""Read frozen solver histories and test the scope claimed for typed holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sera.connected import restore_component
from sera.evaluation import assess
from sera.models import ModelConfig, StatefulModel
from sera.mutations import Mutation, mutate
from sera.solver import Solver
from sera.storage import digest, write_json
from sera.training import source_hash
from sera.typed_learning import TASKS, typed_examples


def tensor_state_identity(state):
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        hasher.update(name.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def read_payload(path):
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError(f"Changed checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=True), manifest


def semantic_example(row):
    body = asdict(row)
    for observation in body["observations"]:
        observation.pop("provenance")
    return digest({key: body[key] for key in ("observations", "task", "target")})


def inspect(root):
    result = {"schema_version": 1, "source_sha256": source_hash(), "frozen_histories": []}
    for seed in range(3):
        directory = root / str(seed) / "solver"
        versions = []
        for path in sorted((directory / "versions").glob("v*.pt")):
            payload, manifest = read_payload(path)
            versions.append({"version": path.stem, "solver_identity": manifest["solver_sha256"],
                             "neural_identity": tensor_state_identity(payload["neural_state"]),
                             "components": {name: tensor_state_identity(value["state"])
                                            for name, value in payload["components"].items()}})
        if not versions:
            raise ValueError("Missing frozen solver history")
        result["frozen_histories"].append({"seed": seed, "versions": versions,
            "unique_neural_states": len({row["neural_identity"] for row in versions}),
            "unique_controller_states": len({row["components"]["controller"] for row in versions}),
            "unique_typed_states": len({row["components"]["typed"] for row in versions}),
            "unique_r1_states": len({row["components"]["r1"] for row in versions})})
    # Seed zero is inspected, not promoted or selected for a new study claim.
    pointer = json.loads((root / "0/solver/current.json").read_text(encoding="utf-8"))
    payload, _ = read_payload(root / f"0/solver/versions/{pointer['version']}.pt")
    components = {name: restore_component(value["config"]) for name, value in payload["components"].items()}
    for name, model in components.items():
        model.load_state_dict(payload["components"][name]["state"])
    neural = StatefulModel(ModelConfig(**payload["model_config"]))
    neural.load_state_dict(payload["neural_state"])
    solver = Solver(neural, components=components, skills=payload["skills"])
    if solver.identity() != pointer["solver_sha256"]:
        raise ValueError("Reconstructed solver identity differs")
    result["saved_seed_zero"] = {"version": pointer["version"], "identity": solver.identity(),
        "neural_config": payload["model_config"],
        "components": {name: {"config": {key: value for key, value in model.export_config().items() if key != "programs"},
                               "real_parameters": sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())}
                       for name, model in components.items()}}
    groups = {"neural": neural, "r1": components["r1"], "typed": components["typed"]}
    addresses = {name: {p.data_ptr() for p in model.parameters()} for name, model in groups.items()}
    result["shared_parameter_storage"] = {f"{a}/{b}": len(addresses[a] & addresses[b])
                                           for a, b in (("neural", "r1"), ("r1", "typed"), ("neural", "typed"))}
    instrument_key = next(name for name in components if name.startswith("r2:"))
    original = components[instrument_key]
    replacement, mutation = mutate(solver, Mutation("expand_instrument", instrument_key, original.dimension), seed=112)
    before = tensor_state_identity(original.state_dict())
    after = tensor_state_identity(replacement.components[instrument_key].state_dict())
    result["instrument_expansion_probe"] = {
        "requested_dimension": original.dimension, "old_dimension": original.dimension,
        "new_dimension": replacement.components[instrument_key].dimension,
        "equal_dimension_request_accepted": True, "same_learned_state": before == after,
        "declared_function_preserving": mutation["function_preserving_at_initialization"],
        "interpretation": "The operation is a fresh replacement, even at equal dimension; it is not a transfer-preserving expansion. No candidate was saved or promoted."}
    result["typed_family_checks"] = []
    for seed in range(3):
        extended = typed_examples(seed=230000 + seed, count=128, split="test-extended", family="extended")
        structure = typed_examples(seed=230000 + seed, count=128, split="test-structure", family="structure")
        for task in TASKS:
            left = [row for row in extended if row.task == task]
            right = [row for row in structure if row.task == task]
            result["typed_family_checks"].append({"seed": seed, "task": task, "examples_per_split": len(left),
                "same_semantic_examples_at_same_index": sum(semantic_example(a) == semantic_example(b) for a, b in zip(left, right)),
                "same_record_identifiers": sum(a.identifier == b.identifier for a, b in zip(left, right))})
    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "0/policy-episodes").glob("*.json")]
    result["policy_training_history_features"] = [
        {"episode_id": row["episode_id"], "previous_attempts": row["diagnosis"]["previous_attempts"],
         "previous_score": row["diagnosis"]["previous_score"], "remaining_budget": row["diagnosis"]["remaining_budget"]}
        for row in episodes if row["split"] == "meta-train"]
    result["promoted_component_retention"] = []
    for seed in range(3):
        for path in sorted((root / str(seed) / "solver/rounds").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") != "promoted":
                continue
            result["promoted_component_retention"].append({"seed": seed, "round": path.stem,
                "losses_by_world": {world: {
                    "prediction_accuracy_loss": before["prediction"]["accuracy"] - row["candidate"]["tasks"][world]["prediction"]["accuracy"],
                    "control_success_loss": before["control_success"] - row["candidate"]["tasks"][world]["control_success"]}
                    for world, before in row["incumbent"]["tasks"].items()}})
    decision = assess({"world": np.full(1024, .75)}, {"world": np.full(1024, .65)},
                      round_index=0, invariants_ok=True, candidate_cost=0)
    result["composite_retention_probe"] = {
        "hypothetical": True, "before_prediction": .9, "after_prediction": .6,
        "before_control": .4, "after_control": .9, "before_world_score": .65,
        "after_world_score": .75, "decision": decision,
        "interpretation": "A 30-point prediction regression is hidden by higher control success in the composite world score. This demonstrates a boundary of the supplied scoring contract, not a regression observed in the three released promotions."}
    result["scope"] = "Read-only reconstruction and in-memory probes. No archived model, journal, training record or result is modified. Equality of saved components demonstrates what persisted; it does not imply that the missing integration is impossible."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous review; use a fresh output filename")
    torch.set_num_threads(1)
    result = inspect(args.study)
    write_json(args.output, result)
    print("Frozen component histories, parameter ownership, typed generators and mutation scope checked.")


if __name__ == "__main__":
    main()
