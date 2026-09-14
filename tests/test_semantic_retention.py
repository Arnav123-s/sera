from dataclasses import replace

import numpy as np
import pytest

from sera.evaluation import CapabilityScores, assess
from sera.models import ModelConfig, StatefulModel
from sera.solver import SolverStore
from sera.typed_learning import typed_examples
from sera.typed_programs import numeric_input
from sera.typed_protocol import BYTE_HOLDOUT, audit_partitions, semantic_id, typed_suite


def test_semantic_identity_ignores_relabeling_spelling_and_units():
    old = typed_examples(seed=4, count=1)
    byte = next(row for row in old if row.task == "byte_sum")
    obs = byte.observations[0]
    renamed = replace(byte, observations=(replace(obs, values=(32.0, *obs.values, 32.0)),), split="test-new-name")
    assert byte.identifier != renamed.identifier
    assert semantic_id(byte) == semantic_id(renamed)
    spatial = next(row for row in old if row.task == "spatial_relation")
    obs = spatial.observations[0]
    factor, unit = (100, "cm") if obs.units == "m" else (.01, "m")
    converted = replace(spatial, observations=(replace(obs, values=tuple(v * factor for v in obs.values), units=unit), spatial.observations[1]))
    assert semantic_id(spatial) == semantic_id(converted)
    with pytest.raises(ValueError, match="Semantic cases reused"):
        audit_partitions({"support": [byte], "test": [renamed]})
    with pytest.raises(ValueError, match="Semantic cases reused"):
        audit_partitions({"support": [byte], "test": [replace(byte, target=(byte.target + 1) % 4)]})


def test_protocol_has_disjoint_cases_and_deliberate_composition_holdouts():
    suite, manifest = typed_suite(seed=41, support_count=48, validation_count=16, test_count=32)
    assert manifest["cross_partition_overlap"] == 0
    for partition, rows in suite.items():
        for row in rows:
            if row.task == "byte_sum":
                a, b = numeric_input(row.observations)
                assert ((a % 4, b % 4) in BYTE_HOLDOUT) == (partition == "test-composition")
            elif row.task == "modular_sum":
                values = numeric_input(row.observations)
                assert (values == values[::-1]) == (partition == "test-composition")
            elif row.task == "binding":
                query = np.argmax(row.observations[-1].values[:4])
                writes = sum(np.argmax(obs.values[:4]) == query for obs in row.observations[:-1])
                assert writes == (2 if partition == "test-composition" else 1)
    other, _ = typed_suite(seed=42, support_count=48, validation_count=16, test_count=32)
    # Development bucket ownership is independent of random seed.
    assert not {semantic_id(r) for r in suite["support"]} & {semantic_id(r) for r in other["test-id"]}
    repeated, _ = typed_suite(seed=41, support_count=48, validation_count=16, test_count=32)
    assert suite == repeated


def bundle(prediction, control, *, name="world/prediction", identity="paired-cases"):
    return CapabilityScores({"world": (prediction + control) / 2},
                             {name: prediction, "world/control": control},
                             {name: identity, "world/control": "paired-controls"})


def test_control_improvement_cannot_hide_prediction_regression_or_inflate_gain_count():
    before = bundle(np.full(1024, .9), np.full(1024, .4))
    after = bundle(np.full(1024, .6), np.full(1024, .9))
    args = dict(round_index=0, invariants_ok=True, candidate_cost=0)
    assert assess(dict(after), dict(before), **args)["admitted"]  # Historical contract.
    result = assess(after, before, **args)
    assert not result["admitted"]
    assert result["failed_capabilities"] == ["world/prediction"]
    assert result["retention_loss_by_capability"]["world/prediction"] == pytest.approx(.3)
    safe = bundle(np.full(1024, .9), np.full(1024, .9))
    result = assess(safe, before, **args)
    assert result["admitted"] and result["samples"] == 1024
    for wrong in (bundle(np.ones(1024), np.ones(1024), name="renamed"),
                  bundle(np.ones(1024), np.ones(1024), identity="different-cases")):
        with pytest.raises(ValueError, match="capabilit"):
            assess(wrong, before, **args)


def test_persistent_admission_fails_closed_when_required_capability_is_omitted(tmp_path):
    store = SolverStore(tmp_path)
    store.initialize(StatefulModel(ModelConfig(width=8, memory_dim=4)))

    def incomplete(solver, seed):
        return {"dataset_id": str(seed)}, {"objective": np.ones(64)}

    with pytest.raises(ValueError, match="omitted a required"):
        store.consider(store.load(), incomplete, required_capabilities={"critical"})
    assert store.current_record()["version"] == "v0"
    assert store.record("v1")["parent"] == "v0"  # Rejected work remains recoverable.
