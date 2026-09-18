import copy

import torch

from experiments import refinement_resolution as rs
from experiments.continuing_growth.common import SKILLS


def result(score, correct):
    return {"macro": score, "scores": [score]*len(SKILLS),
            "skills": {s: {"correct": correct} for s in SKILLS}}


def test_smaller_step_must_pass_unchanged_independent_gate(monkeypatch):
    owner = {"w": torch.tensor([1.])}
    monkeypatch.setattr(rs, "knowledge", lambda value: {"w": value["w"].clone()})
    monkeypatch.setattr(rs, "apply", lambda value, weights: value.update(weights))
    monkeypatch.setattr(rs, "measure", lambda value, data: result(.401, 20))
    before = result(.4, 20)
    trial = result(.42, 18)
    accepted, attempts = rs.resolve(owner, {"w": torch.tensor([0.])}, before, trial, copy.deepcopy(before), {})
    assert accepted and len(attempts) == 2
    assert attempts[0]["original_gate"] is False
    assert attempts[-1]["original_gate"] is True
    assert torch.equal(owner["w"], torch.tensor([.5]))
    assert trial == result(.401, 20)


def test_flat_or_negative_progress_cannot_earn_resolution_acceptance(monkeypatch):
    owner = {"w": torch.tensor([1.])}
    monkeypatch.setattr(rs, "knowledge", lambda value: {"w": value["w"].clone()})
    monkeypatch.setattr(rs, "apply", lambda value, weights: value.update(weights))
    monkeypatch.setattr(rs, "measure", lambda value, data: result(.4, 20))
    before = result(.4, 20)
    accepted, attempts = rs.resolve(owner, {"w": torch.tensor([0.])}, before, copy.deepcopy(before), before, {})
    assert not accepted and len(attempts) == 9
    assert not any(a["accepted"] for a in attempts)
