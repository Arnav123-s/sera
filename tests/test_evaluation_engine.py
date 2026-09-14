import json

import numpy as np
import pytest
import torch

from sera.cli import main
from sera.engine import improve, rollback
from sera.evaluation import AdmissionPolicy, assess
from sera.models import ModelConfig, StatefulModel
from sera.storage import Journal


def test_admission_gain_retention_budget_and_invalid_scores():
    zeros, ones = np.zeros(2048), np.ones(2048)
    arguments = dict(round_index=0, invariants_ok=True, candidate_cost=5)
    assert assess({"a": ones}, {"a": zeros}, **arguments)["admitted"]
    assert not assess({"a": ones}, {"a": ones}, **arguments)["admitted"]
    assert not assess({"a": ones, "b": zeros}, {"a": zeros, "b": ones}, **arguments)["admitted"]
    assert not assess(
        {"a": ones}, {"a": zeros}, policy=AdmissionPolicy(max_candidate_cost=1), **arguments
    )["admitted"]
    for invalid in ([np.nan], [np.inf], [-1], [2]):
        with pytest.raises(ValueError):
            assess({"a": invalid}, {"a": [0]}, **arguments)


def test_evaluation_reuse_and_journal_tampering(tmp_path):
    journal = Journal(tmp_path / "ledger.sqlite")
    assert journal.reserve_evaluation("first") == 0
    assert journal.reserve_evaluation("second") == 1
    with pytest.raises(ValueError, match="consumed"):
        journal.reserve_evaluation("first")
    journal.append("diagnosis", {"score": 0.2})
    journal.append("candidate", {"score": 0.9})
    assert journal.verify()
    with journal.connect() as db:
        db.execute("UPDATE events SET body='{}' WHERE id=1")
    with pytest.raises(ValueError, match="integrity"):
        journal.verify()


def test_actual_failure_discovery_independent_promotion_and_rollback(tmp_path):
    torch.manual_seed(5)
    model = StatefulModel(ModelConfig(width=8, memory_dim=4))
    result = improve(model, tmp_path, samples=512)
    assert result["status"] == "promoted"
    assert result["candidate"]["tasks"]["ordered_control"]["accuracy"] == 1
    assert all(v <= 0 for v in result["decision"]["retention_loss_by_task"].values())
    assert result["diagnosis"]["dataset_id"] != result["dataset_id"]
    assert json.loads((tmp_path / "current.json").read_text())["version"] == "v1"
    assert rollback(tmp_path)["version"] == "v0"
    with pytest.raises(ValueError, match="parent"):
        rollback(tmp_path)


def test_exhausted_search_keeps_incumbent_and_reports_unresolved(tmp_path):
    model = StatefulModel(ModelConfig(width=8, memory_dim=4))
    result = improve(model, tmp_path, samples=16, max_queries=2)
    assert result["status"] == "unresolved"
    assert json.loads((tmp_path / "current.json").read_text())["version"] == "v0"


def test_cli_reports_failures_with_nonzero_status(tmp_path):
    assert main(["evaluate", str(tmp_path / "missing.pt"), "--output", str(tmp_path)]) == 1
