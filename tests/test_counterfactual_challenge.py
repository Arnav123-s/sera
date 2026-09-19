"""Regression for narrow-family agreement hiding an omitted mechanism."""

import copy

import pytest

from experiments import counterfactual_loop
from experiments.counterfactual_challenge import investigate
from experiments.counterfactual_check import (
    Simulation,
    evaluate_investigation,
    replay_investigation,
)
from experiments.gap_inquiry import ROOT, read


@pytest.mark.parametrize("goal,policy", [
    ("406b779c74b2ddc25b984c43bec8e016a52797b732fc4e697d4f9726a67dac0c", "entropy"),
    ("57c4a82c7bb7fb0ebf589b845acb024e557bb68c2cbbe1fcf28e3dbdbf165ee1", "learned"),
])
def test_challenge_repairs_observed_validation_failure_without_new_hints(goal, policy, monkeypatch):
    folder = ROOT / "runs/CI-study-001/validation" / goal
    if not (folder / (policy + ".json")).exists():
        pytest.skip("Optional archived rejected-validation fixture")
    old = read(folder / (policy + ".json"))
    event = read(folder / "initial.json")
    matrices = read(ROOT / "runs/CI-study-001/frontier.json")["matrices"]
    source = Simulation(event, 45100 + int(goal[:8], 16), omitted=True)
    old_copy = copy.deepcopy(old)
    assert evaluate_investigation(old, source)["wrong_consensus"] > 0
    repaired = investigate(event, matrices, source, policy, old["weights"], old["seed"], lambda _: None, old)
    assert old == old_copy
    assessment = evaluate_investigation(repaired, source)
    assert assessment["wrong_consensus"] == 0
    assert assessment["covered"] == assessment["cases"]
    assert repaired["known_basis_challenged"]
    assert repaired["observations"][:len(old["observations"])] == old["observations"]
    assert len(repaired["observations"]) <= 10
    monkeypatch.setattr(counterfactual_loop, "investigate", investigate)
    assert replay_investigation(repaired, source, matrices)["accepted"]
