import copy
import json

import pytest
import torch

from experiments.cross_route_transfer.study import ROOT, parent
from experiments.transfer_integration.migration import changed_readout_only, migrate


def test_migration_refuses_a_changed_representation():
    solver = parent()
    successor = copy.deepcopy(solver.components["r1"])
    with torch.no_grad():
        successor.typed_categorical.bias[0].add_(.01)
    with pytest.raises(ValueError, match="exceeds"):
        changed_readout_only(solver.components["r1"], successor)


def test_factual_replay_and_finite_reproof_under_new_owner():
    torch.set_num_threads(1)
    solver = parent()
    successor = copy.deepcopy(solver.components["r1"])
    with torch.no_grad():
        successor.typed_numeric.bias[0].add_(.001)
    factual = json.loads((ROOT/"runs/SHARED-GG-001/circle/situation.json").read_text())
    finite = json.loads((ROOT/"research-continuation/16_v3/GC-002/repair/corrected-library.json").read_text())
    _, session, library, report = migrate(solver, successor, factual, finite)
    assert report["generator_outputs_bitwise_equal"]
    assert report["finite_graph_unchanged"]
    assert report["parent_solver_unchanged"]
    assert len(report["stale_records_rejected"]) == 2
    assert len(session.situation.observations) == 16
    assert len(session.situation.labels) == 1
    assert library.owner_id == report["new_owner"]
    assert library.owner_id != finite["owner_sha256"]
    counts = report["replay_and_generator_check_work"]["counts"]
    assert counts.get("factual_observations", 0) == factual["budget"]["counts"].get("factual_observations", 0)
    assert counts["replayed_observations"] >= 16
    assert report["successor_replay_added_operations"].get("factual_observations", 0) == 0
