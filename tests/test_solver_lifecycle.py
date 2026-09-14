import json
import os
import subprocess
import sys

import pytest
import torch

from sera.data import make_batch
from sera.engine import improve
from sera.evaluation import evaluate
from sera.models import ModelConfig, StatefulModel
from sera.solver import SolverStore


def test_two_generations_reload_rejection_and_behavioral_rollback(tmp_path):
    torch.manual_seed(21)
    neural = StatefulModel(ModelConfig(width=8, memory_dim=4))
    # Remove test flakiness from a lucky untrained ordered-control classifier.
    first = improve(neural, tmp_path, samples=2048, task=2)
    assert first["status"] == "promoted"
    store = SolverStore(tmp_path)
    x = make_batch(12, 40, 99, split="fresh-process", task=2)
    before = store.load().predict_probabilities(x.inputs)
    assert torch.equal(before.argmax(-1), x.targets)
    command = [sys.executable, "-m", "sera", "evaluate", str(tmp_path),
               "--output", str(tmp_path / "cli-evaluation"), "--samples", "128"]
    subprocess.run(command, check=True, capture_output=True,
                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    cli = json.loads((tmp_path / "cli-evaluation/evaluation.json").read_text())
    assert cli["tasks"]["ordered_control"]["accuracy"] == 1
    second = improve(neural, tmp_path, samples=2048, task=4)
    assert second["status"] == "promoted"
    assert second["parent"] == first["version"]
    assert second["round_index"] == first["round_index"] + 1
    current = store.load()
    novel = make_batch(12, 40, 101, split="novel", task=4)
    assert torch.equal(current.predict_probabilities(novel.inputs).argmax(-1), novel.targets)
    torch.testing.assert_close(current.predict_probabilities(x.inputs), before, atol=0, rtol=0)
    def evaluator(solver, seed):
        return evaluate(solver, seed=seed, split="rejection", samples=128, tasks=range(5))
    rejected = store.consider(current, evaluator)
    assert rejected["status"] == "rejected"
    assert store.load().version == second["version"]
    assert store.rollback()["version"] == first["version"]
    assert "4" not in store.load().skills
    torch.testing.assert_close(store.load().predict_probabilities(x.inputs), before, atol=0, rtol=0)
    with pytest.raises(ValueError, match="consumed"):
        store.journal.reserve_evaluation(rejected["reservation"])
    # Rollback does not rewind the evaluation ledger.
    third = store.consider(store.load(), evaluator)
    assert third["round_index"] == rejected["round_index"] + 1


def test_checkpoint_integrity_and_failed_evaluation_preserve_current(tmp_path):
    store = SolverStore(tmp_path)
    store.initialize(StatefulModel(ModelConfig(width=8, memory_dim=4)))
    def failure(solver, seed):
        raise RuntimeError("Evaluator deliberately interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        store.consider(store.load(), failure)
    assert store.load().version == "v0"
    assert json.loads((tmp_path / "rounds/0.json").read_text())["status"] == "failed"
    invalid = store.load()
    with torch.no_grad():
        next(invalid.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        store.consider(invalid, failure)
    assert store.load().version == "v0"
    with (tmp_path / "versions/v0.pt").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        store.load()
