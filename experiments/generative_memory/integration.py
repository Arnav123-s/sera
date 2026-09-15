"""Explicit trained-parent integration experiment, excluded from default unit tests.

The pytest entry point permits the existing frozen process supervisor to apply its
hard deadline without changing the GG-ACT supervisor source. This is a measured
integration experiment with a real saved parent, not a random-fixture capability claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from sera.contracts import EvidenceKind, Provenance
from sera.environments import WorldSpec
from sera.event_ir import Event, EventRole
from sera.generative import SCHEMA, GenerativeSharedR1, SharedGenerativeSession, source_manifest
from sera.shared import replace_shared_owner
from sera.shared_evaluation import evaluate_shared, score_record
from sera.solver import SolverStore
from sera.storage import digest, write_json
from sera.world_graph import ExecutionBudget, interpreter_fingerprint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research-continuation/04_protocols/SHARED-GG-001.json"


def file_hashes(root):
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def truth(family, phase):
    """Assessor only. The learner receives events, never this family or function."""
    phase = np.asarray(phase)
    c, s = np.cos(math.pi*phase), np.sin(math.pi*phase)
    if family == "circle":
        return np.stack((.25 + .8*c - .3*s, -.4 + .8*s + .3*c), -1)
    if family == "ellipse":
        return np.stack((.25 + 1.2*c - .1*s, -.4 + .2*c + .5*s), -1)
    if family == "line":
        return np.stack((.25 + .8*phase, -.4 - .3*phase), -1)
    if family == "radial_growth":
        return np.stack((.25 + (.8+.4*phase)*c - .3*s,
                         -.4 + (.8+.4*phase)*s + .3*c), -1)
    raise ValueError("Unknown frozen integration fixture")


def run(output):
    started, cpu = time.perf_counter(), time.process_time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for name, expected in protocol["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen integration source differs: {name}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    parent_path = ROOT / protocol["parent_store"]
    preserved = file_hashes(parent_path)
    parent = SolverStore(parent_path).load()
    assert parent.identity() == protocol["parent_solver_sha256"]
    parent_identity = parent.identity()
    specs = [WorldSpec(**row) for row in json.loads((parent_path / "worlds.json").read_text())]
    retention = protocol["retention"]
    before_report, before_scores = evaluate_shared(parent, specs, **retention, world_learning=True)
    write_json(output / "retention-before.json", {"report": before_report, "scores": score_record(before_scores)})
    source = source_manifest()
    for name, content in source.items():
        path = output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    records = []
    for index, family in enumerate(protocol["families"]):
        folder = output / family
        folder.mkdir()
        solver = copy.deepcopy(parent)
        owner = GenerativeSharedR1.from_shared(parent.components["r1"], generator_noise=protocol["noise"])
        replace_shared_owner(solver, owner)
        initial_identity = solver.identity()
        SolverStore(folder / "initial").initialize(solver)
        session = SharedGenerativeSession(solver, situation_id="observed-trajectory",
                                          budget=ExecutionBudget(protocol["operation_budget"]))
        phases = np.linspace(-.6, .6, protocol["observations"])
        rng = np.random.default_rng(protocol["observation_seed"] + index)
        measured = truth(family, phases) + rng.normal(0, protocol["noise"], (len(phases), 2))
        prior_predictions = session.predict(protocol["query_phases"])["mean"].tolist()
        updates, events = [], []
        for position, (phase, pair) in enumerate(zip(phases, measured)):
            reported = pair.copy()
            if position == protocol["corrupted_position"]:
                reported += protocol["corruption"]
            event = Event(SCHEMA, position, tuple(map(float, [phase, *reported])),
                          Provenance("observed-simulator", f"observation-{position}", EvidenceKind.SYNTHETIC),
                          entity_refs=("trajectory",))
            update = session.observe(event)
            updates.append(update.record)
            events.append(event.record())
            write_json(folder / "progress.json", {"observations": position+1, "solver": solver.identity(),
                                                   "budget": session.situation.budget.record()})
        pre_correction = session.predict(protocol["query_phases"])["mean"].tolist()
        position = protocol["corrupted_position"]
        correction = Event(SCHEMA, position, tuple(map(float, [phases[position], *measured[position]])),
                           Provenance("verified-sensor-recheck", "correction-1", EvidenceKind.SYNTHETIC),
                           entity_refs=("trajectory",), role=EventRole.TARGET)
        corrected = session.correct(correction)
        (folder / "pre-correction.pt").write_bytes(corrected.predecessor.checkpoint)
        write_json(folder / "pre-correction-situation.json", json.loads(corrected.predecessor.situation_json))
        previous_solver, previous_session = corrected.predecessor.restore()
        assert previous_solver.identity() == corrected.record["parent_solver_sha256"]
        assert previous_session.predict(protocol["query_phases"])["mean"].tolist() == pre_correction
        state_before_branch = solver.identity()
        direct = session.predict(protocol["query_phases"])
        branch = session.fork()
        imagined = branch.predict(protocol["query_phases"])
        assert all(torch.equal(value, imagined[key]) for key, value in direct.items())
        assert solver.identity() == state_before_branch
        snapshot = session.snapshot()
        write_json(folder / "situation.json", snapshot)
        checkpoint = SolverStore(folder / "corrected")
        checkpoint.initialize(solver)
        reloaded = checkpoint.load()
        restored = SharedGenerativeSession.restore(reloaded, snapshot)
        assert restored.predict(protocol["query_phases"])["mean"].tolist() == direct["mean"].tolist()
        assert reloaded.components["r1"] is reloaded.neural.owner is reloaded.components["typed"].owner
        retained_names = []
        for name, tensor in parent.state_dict().items():
            assert torch.equal(tensor, reloaded.state_dict()[name]), name
            retained_names.append(name)
        assert parent.skills == reloaded.skills
        assert {name: p.numel() for name, p in parent.named_parameters()} == {
            name: p.numel() for name, p in reloaded.named_parameters()}
        # Queries and family labels belong to this assessor, never to the update or acceptance callback.
        clean = truth(family, protocol["query_phases"])
        predictions = direct["mean"].numpy()
        costs = session.situation.budget.record()
        record = {"family": family, "initial_solver_sha256": initial_identity,
                  "corrected_solver_sha256": reloaded.identity(), "events": events,
                  "correction": correction.record(), "updates": updates, "correction_update": corrected.record,
                  "query_truth_assessor_only": clean.tolist(), "prior_predictions": prior_predictions,
                  "pre_correction_predictions": pre_correction, "corrected_predictions": predictions.tolist(),
                  "prior_mse": float(np.mean((np.asarray(prior_predictions)-clean)**2)),
                  "pre_correction_mse": float(np.mean((np.asarray(pre_correction)-clean)**2)),
                  "corrected_mse": float(np.mean((predictions-clean)**2)),
                  "class_probabilities": direct["class_probabilities"].tolist(),
                  "registered_generator_bytes": sum(v.numel()*v.element_size() for n, v in
                      reloaded.components["r1"].named_buffers() if n.startswith("generator_")),
                  "retained_parent_state_tensors": len(retained_names), "new_trainable_parameters": 0,
                  "same_owner_views": True, "exact_reload_predictions": True,
                  "exact_conditional_predictions": True, "predecessor_restored": True,
                  "session_snapshot_bytes": len(json.dumps(snapshot).encode()), "operation_budget": costs}
        write_json(folder / "result.json", record)
        records.append(record)
        if family == "circle":
            after_report, after_scores = evaluate_shared(reloaded, specs, **retention, world_learning=True)
            assert score_record(before_scores) == score_record(after_scores)
            assert before_report == after_report
            write_json(output / "retention-after.json", {"report": after_report,
                                                         "scores": score_record(after_scores)})
    assert parent.identity() == parent_identity
    assert file_hashes(parent_path) == preserved
    result = {"experiment": protocol["experiment_id"], "status": "PASS",
              "protocol_sha256": digest(protocol), "interpreter_sha256": interpreter_fingerprint(source),
              "parent_solver_sha256": parent_identity, "parent_files_unchanged": len(preserved),
              "families": [{k: r[k] for k in ("family", "prior_mse", "pre_correction_mse", "corrected_mse",
                                               "class_probabilities", "registered_generator_bytes",
                                               "retained_parent_state_tensors")} for r in records],
              "retention_capabilities": len(before_scores.capabilities),
              "retention_scores_compared": sum(len(v) for v in before_scores.capabilities.values()),
              "retention_exact": True, "source_hashes": protocol["source_hashes"],
              "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
              "boundary": "Four fixed integration fixtures from the existing trained parent, not four random-replicate "
                          "capability benchmarks. Existing controller/instruments are inherited and separately counted; "
                          "new generator buffers belong to R1. Conditional completion is not causal rollout. "
                          "Old routes are unchanged by construction; this establishes conditional retention, not positive transfer.",
              "parent_parameters_by_component": {name: sum(p.numel() for p in module.parameters())
                                                  for name, module in parent.components.items()},
              "artifact_files": file_hashes(output)}
    write_json(output / "result.json", result)
    return result


def test_trained_parent_integration():
    """Run explicitly with pytest under scripts/run_bounded.py's frozen allowlist."""
    torch.set_num_threads(1)
    result = run(Path(os.environ.get("SERA_INTEGRATION_OUTPUT", str(ROOT / "runs/SHARED-GG-001"))))
    assert result["status"] == "PASS"
    print(json.dumps({key: result[key] for key in ("status", "families", "retention_capabilities",
                                                  "retention_scores_compared", "wall_seconds")}))
