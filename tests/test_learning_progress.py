import copy

import numpy as np
import pytest
import torch

from experiments.learning_progress.audit import independent
from experiments.learning_progress.data import group_indices, method_rows
from experiments.learning_progress.model import candidates
from experiments.learning_progress.runtime import credit


def test_discovery_credit_cannot_be_farmed_by_forget_relearn():
    low = {"scores": [.2]*6, "accuracy": [.3]*6}
    high = {"scores": [.4]*6, "accuracy": [.5]*6}
    first = credit(low, high, low["scores"])
    lost = credit(high, low, first["highwater"])
    regained = credit(low, high, lost["highwater"])
    assert first["discovery"] == pytest.approx(.1)
    assert regained["discovery"] == 0
    assert lost["reward"]+regained["reward"] == pytest.approx(0)
    assert credit(high, high, first["highwater"])["reward"] == 0


def test_credit_rejects_nonfinite_scores():
    row = {"scores": [.2]*6, "accuracy": [.3]*6}
    bad = copy.deepcopy(row)
    bad["scores"][2] = float("nan")
    with pytest.raises(ValueError):
        credit(row, bad, row["scores"])


def test_group_partitions_preserve_whole_source_groups():
    rows = [{"id": f"{g}/{i}", "group": str(g)} for g in range(15) for i in range(4)]
    partitions = [group_indices(rows, "group", 3, i, 100) for i in range(3)]
    groups = [{rows[i]["group"] for i in part} for part in partitions]
    assert len(set.union(*map(set, partitions))) == 60
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])


def test_methods_avoid_failed_and_duplicate_programs():
    rows = method_rows(34091, 128)
    assert not set(rows["y"].tolist()) & {4, 5}
    assert set(rows["y"].tolist()) == {0, 1, 2, 3, 6}
    assert rows["x"].shape == (128, 7, 12)


def test_procedure_inputs_do_not_contain_future_labels():
    x = candidates([.2]*6, [.5]*6, [0.]*6, [8]*6)
    assert x.shape == (18, 12)
    assert torch.equal(x[:3, :6], x[0, :6].expand(3, 6))
    assert len(set(x[:3, 10].tolist())) == 3


def test_independent_multigold_checker_credits_either_valid_sentence():
    row = {"logits": [[0., 2., 1.]], "gold": [[True, True, False]], "labels": [0]}
    loss, accuracy = independent(row)
    assert accuracy == 1
    assert loss == pytest.approx(-np.log((1+np.exp(2))/(1+np.exp(2)+np.exp(1))))


def test_actual_owner_learning_persistence_and_credit_tampering(tmp_path, monkeypatch):
    from experiments.learning_progress import runtime
    from experiments.learning_progress.common import ROOT, read
    from experiments.learning_progress.model import knowledge
    from experiments.quest_portfolio.runtime import QuestSession

    parent_path = ROOT / "runs/QP-study-001/parent.json"
    assert parent_path.exists(), "Restore the published quest runtime artifacts"
    parent = QuestSession(parent=read(parent_path), authority=tmp_path / "assessor").snapshot()
    (tmp_path / "data-manifest.json").write_text('{"purpose":"unit-test fixture"}', encoding="utf-8")
    monkeypatch.setattr(runtime, "RUN", tmp_path)
    session = runtime.LearningSession(parent=parent)
    owner = session.owner
    assert owner is session.base.base.grounded.base.study.owner
    initial = knowledge(owner)
    generator = torch.Generator().manual_seed(1)
    width = owner.book_mean.shape[1]
    rows = {s: {"x": torch.randn(8, 2, width, generator=generator), "y": torch.ones(8, dtype=torch.long)}
            for s in ("dictionary", "conversation", "mathematics", "science")}
    mask = torch.ones(8, 16, dtype=torch.bool)
    gold = torch.zeros_like(mask)
    gold[:, 0] = True
    rows["reading"] = {"x": torch.zeros(8, 16, len(owner.reading_mean)), "mask": mask,
                       "gold": gold, "y": torch.zeros(8, dtype=torch.long)}
    rows["methods"] = method_rows(34080, 8)
    session.start("unit-test", quota=1)
    session.step(rows, rows, "balanced")
    assert any(not torch.equal(v, knowledge(owner)[k]) for k, v in initial.items())
    with pytest.raises(ValueError, match="unfinished"):
        session.start("renamed")
    snap = session.snapshot()
    resumed = runtime.LearningSession(saved=snap)
    assert resumed.snapshot() == snap
    tampered = copy.deepcopy(snap)
    tampered["highwater"][0] += .1
    with pytest.raises(ValueError, match="high-water"):
        runtime.LearningSession(saved=tampered)
    tampered = copy.deepcopy(snap)
    tampered["events"][0]["credit"]["reward"] += 100
    with pytest.raises(ValueError, match="credit history"):
        runtime.LearningSession(saved=tampered)
