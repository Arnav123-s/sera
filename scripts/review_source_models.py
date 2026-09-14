"""Compare preserved packet models with SERA without rewriting historical evidence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs
from sera.models import KINDS, ModelConfig, StatefulModel, delta_write
from sera.programs import discover
from sera.quantum import EventInstrument
from sera.r2 import ControlledInstrument
from sera.storage import write_json
from sera.training import source_hash

ARCHIVE_HASH = "7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce"
PREFIX = "Quantum_AGI_Research_Package/"
REVIEWED = {
    "experiment.py": "894fac1ddc244fbeea08a3171c1573c496f7fac2fff7d0202d4710416fa6c57b",
    "verify_and_extend.py": "993f003b03850d3b5978f95309787d465cccff5a2cb486c13b3c3b3de2f0fff2",
    "instrument.py": "7b1f192b75d6e34ea3dc333eab7deaa6897e86db5d49c99a790357bccd3e8bfc",
    "program_discovery.py": "211da4c0689b0cbd0ed25bc39dc66442187c8655448bca69f6d732e1d712fca6",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_reviewed(name, path, root):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.ROOT = root
    return module


def cycle_batch(n, length, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(4, (n, 1), generator=generator) + torch.arange(length)[None]) % 4


@torch.no_grad()
def cycle_metrics(model, seed):
    sequence = model.generate(32, seed)
    operators = model.operators()
    return {
        "ID_conditional_nll": -model.log_likelihood(cycle_batch(512, 8, 800 + seed))[:, 1:].mean().item(),
        "OOD_conditional_nll": -model.log_likelihood(cycle_batch(512, 24, 900 + seed))[:, 1:].mean().item(),
        "completeness_error": float((torch.einsum("vji,vjk->ik", operators, operators) - torch.eye(4)).norm()),
        "generated_sequence": sequence,
        "cycle_consistency": sum(sequence[i] == (sequence[i - 1] + 1) % 4 for i in range(1, 32)) / 31,
    }


def metric_differences(observed, expected):
    return {key: abs(observed[key] - expected[key]) for key in observed
            if isinstance(observed[key], (float, int))}


@torch.no_grad()
def equivalence_checks(instrument_module, archive):
    rows = []
    for seed in range(3):
        raw = archive.read(PREFIX + f"results/checkpoints/instrument_{seed}.pt")
        saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        original = instrument_module.InstrumentModel()
        original.load_state_dict(saved)
        legacy = EventInstrument(complex_valued=False)
        legacy.raw.copy_(original.raw)
        controlled = ControlledInstrument(dimension=4, rank=1, complex_valued=False,
                                           event_kind="kraus", event_rank=1)
        controlled.event_raw.copy_(original.raw)
        controlled.action_raw.copy_(torch.eye(4).expand(4, -1, -1))
        tokens = cycle_batch(24, 24, 64000 + seed)
        original_probability = original.log_likelihood(tokens).exp()
        controlled_probability = controlled.sequence(
            tokens, torch.zeros(24, 23, dtype=torch.long), torch.zeros(24, dtype=torch.long)
        )[0].gather(-1, tokens[:, 1:, None]).squeeze(-1)
        legacy_error = float((legacy(tokens) - original_probability).abs().max())
        controlled_error = float((controlled_probability - original_probability[:, 1:]).abs().max())
        rows.append({"seed": seed, "original_checkpoint_sha256": sha(raw),
                     "legacy_event_probability_max_error": legacy_error,
                     "identity_action_controlled_probability_max_error": controlled_error,
                     "tolerance": 2e-6, "passed": max(legacy_error, controlled_error) < 2e-6})
    torch.manual_seed(171)
    memory = torch.randn(3, 1, 4, 8, dtype=torch.float64)
    key = torch.nn.functional.normalize(torch.randn(3, 1, 8, dtype=torch.float64), dim=-1)
    value = torch.randn(3, 1, 4, dtype=torch.float64)
    alpha, beta = torch.rand(2, 3, 1, dtype=torch.float64)
    decayed = alpha[..., None, None] * memory
    reference = decayed + beta[..., None, None] * (value - (decayed @ key[..., None]).squeeze(-1))[..., None] * key[..., None, :]
    error = float((delta_write(memory, key, value, alpha, beta) - reference).abs().max())
    rows.append({"case": "same corrective delta operator at packet 4-by-8 shape",
                 "max_error": error, "tolerance": 1e-12, "passed": error < 1e-12})
    return rows


def program_comparison(module):
    rows = []
    for seed in range(11):
        rng = np.random.default_rng(3300 + seed)
        if seed == 0:
            transition = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [0, 2, 1, 3], [0, 1, 3, 2]])
        else:
            transition = rng.integers(0, 4, (4, 4))
            order = rng.permutation(4)
            for i, state in enumerate(order):
                transition[0, state] = order[(i + 1) % 4]

        def oracle(sequence):
            state = 0
            for action in sequence:
                state = int(transition[int(action), state])
            return state

        original = module.learn_transition_program(oracle)
        current = discover(oracle, environment_id=f"packet-comparison-{seed}", actions=4,
                           max_states=16, max_queries=100)
        original_errors, sera_errors, count = 0, 0, 0
        for length in (12, 24, 96):
            for sequence in rng.integers(0, 4, (512, length)):
                expected = oracle(sequence)
                original_errors += module.execute(original, sequence) != expected
                sera_errors += current.program.execute(sequence) != expected
                count += 1
        rows.append({"seed": seed, "tested_sequences": count, "original_errors": original_errors,
                     "sera_errors": sera_errors, "original_queries": original["queries"],
                     "sera_queries": current.oracle_queries,
                     "original_executed_actions": sum(len(x["query"]) for x in original["query_trace"]),
                     "sera_executed_actions": current.executed_actions})

    def renamed_oracle(sequence):
        return 10 + sum(sequence) % 4

    renamed = module.learn_transition_program(renamed_oracle)
    rejected = None
    try:
        discover(renamed_oracle, environment_id="renamed", actions=4, max_states=16)
    except ValueError as error:
        rejected = str(error)
    return {"matched_tasks": rows,
            "meaning": "Same finite-state induction family; SERA repeats transition queries to check observed determinism. This doubles transition feedback, not learning efficiency.",
            "renamed_state_probe": {"original_states": renamed["states"],
                                    "sera_rejection": rejected,
                                    "boundary": "SERA narrows observable labels to contiguous zero-based IDs; the source discovery dictionary does not."}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_bytes = args.archive.read_bytes()
    if sha(archive_bytes) != ARCHIVE_HASH:
        raise ValueError("Original packet hash differs")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "archive_sha256": ARCHIVE_HASH,
              "sera_source_sha256": source_hash(), "reviewed_modules": REVIEWED,
              "status": "running", "replay_controls": []}
    costs = Costs()
    torch.set_num_threads(1)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            modules = {}
            for filename, expected in REVIEWED.items():
                raw = archive.read(PREFIX + "code/" + filename)
                if sha(raw) != expected:
                    raise ValueError(f"Unreviewed source bytes: {filename}")
                path = args.output / filename
                path.write_bytes(raw)
                name = "experiment" if filename == "experiment.py" else "packet_" + path.stem
                modules[filename] = load_reviewed(name, path, args.output.resolve())
            experiment = modules["experiment.py"]
            verification = modules["verify_and_extend.py"]
            instrument = modules["instrument.py"]
            programs = modules["program_discovery.py"]
            checkpoint_root = args.output / "results/checkpoints"
            checkpoint_root.mkdir(parents=True)
            result["source_models"] = [{"name": name,
                "real_parameters": sum(p.numel() * (2 if p.is_complex() else 1) for p in experiment.Model(name).parameters())}
                for name in experiment.KINDS]
            result["sera_legacy_models"] = [{"name": name,
                "real_parameters": sum(p.numel() * (2 if p.is_complex() else 1) for p in StatefulModel(ModelConfig(kind=name)).parameters()),
                "retained_state_bytes": StatefulModel(ModelConfig(kind=name)).state_bytes()}
                for name in KINDS]
            source_map = json.loads(archive.read(PREFIX + "results/physics_component_map.json"))
            current_map = json.loads(Path("research/physics_component_map.json").read_text(encoding="utf-8"))
            result["concept_map"] = {"source_entries": len(source_map), "sera_entries": len(current_map),
                                     "all_fields_equal": source_map == current_map}
            result["archive_members"] = [{"path": row.filename.removeprefix(PREFIX), "bytes": row.file_size,
                                           "sha256": sha(archive.read(row.filename))}
                                          for row in archive.infolist() if not row.is_dir()]
            with costs.phase("instrument-checkpoint-and-operator-equivalence"):
                result["operator_equivalence"] = equivalence_checks(instrument, archive)
                expected_instruments = json.loads(archive.read(PREFIX + "results/instrument_results.json"))
                result["instrument_checkpoint_checks"] = []
                for seed, expected in enumerate(expected_instruments):
                    model = instrument.InstrumentModel()
                    model.load_state_dict(torch.load(io.BytesIO(archive.read(
                        PREFIX + f"results/checkpoints/instrument_{seed}.pt")), weights_only=True, map_location="cpu"))
                    observed = cycle_metrics(model, seed)
                    result["instrument_checkpoint_checks"].append({"seed": seed, "observed": observed,
                        "differences": metric_differences(observed, expected),
                        "generated_sequence_equal": observed["generated_sequence"] == expected["generated_sequence"]})
            with costs.phase("original-instrument-fresh-training"):
                # These reviewed functions write only within their redirected ROOT.
                with contextlib.redirect_stdout(io.StringIO()):
                    instrument.run()
                fresh = json.loads((args.output / "results/instrument_results.json").read_text())
                result["fresh_instruments"] = [{"observed": row,
                    "differences": metric_differences({key: row[key] for key in (
                        "ID_conditional_nll", "OOD_conditional_nll", "cycle_consistency")}, expected_instruments[row["seed"]])}
                    for row in fresh]
            print("Original instruments: checkpoints, operator equivalence and three fresh runs complete", flush=True)
            with costs.phase("original-and-sera-program-discovery"):
                with contextlib.redirect_stdout(io.StringIO()):
                    source_programs = programs.run()
                result["original_program_results_equal"] = source_programs == json.loads(
                    archive.read(PREFIX + "results/program_discovery.json"))
                result["program_comparison"] = program_comparison(programs)
            print("Eleven original program environments compared directly with SERA", flush=True)
            expected_replay = {(row["kind"], row["seed"]): row for row in json.loads(
                archive.read(PREFIX + "results/replay_adaptation.json"))}
            for kind in experiment.KINDS:
                for seed in range(3):
                    with costs.phase(f"original-replay/{kind}/{seed}"):
                        raw = archive.read(PREFIX + f"results/checkpoints/{kind}_{seed}.pt")
                        (checkpoint_root / f"{kind}_{seed}.pt").write_bytes(raw)
                        observed = verification.replay_job((kind, seed))
                        expected = expected_replay[kind, seed]
                        accuracy_error = max([abs(observed["new_task"]["accuracy"] - expected["new_task"]["accuracy"])] +
                            [abs(observed["retention"][key]["accuracy"] - expected["retention"][key]["accuracy"]) for key in expected["retention"]])
                        result["replay_controls"].append({"observed": observed,
                            "initial_checkpoint_sha256": sha(raw), "maximum_accuracy_difference": accuracy_error})
                    write_json(args.output / "comparison.json", result)
                print(f"Original replay controls: {kind}/0,1,2 complete", flush=True)
            if not all(row["passed"] for row in result["operator_equivalence"]):
                raise AssertionError("Source-to-SERA operator equivalence failed")
            if not result["concept_map"]["all_fields_equal"]:
                raise AssertionError("Preserved physics concept map differs")
            if not result["original_program_results_equal"]:
                raise AssertionError("Original program results differ")
            result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["costs"] = costs.record()
        result["scope"] = "Supplementary source reproduction and direct restricted-model comparison. Earlier 36-core checkpoint/training evidence remains in its original release. Fresh outputs are separate from delivered source and frozen SERA results. Differences are reported without replacing history."
        write_json(args.output / "comparison.json", result)


if __name__ == "__main__":
    main()
