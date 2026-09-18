"""Synthetic fixtures test isolation/credit plumbing; they do not teach the learner."""

import copy

import numpy as np
import pytest

from experiments.gap_assessor import assess_development, credit, evaluate, validate_receipt
from experiments.gap_inquiry import fit, normalize_program, portfolio, predict


def fixture():
    t = np.linspace(-2, 2, 60)
    return {"time": t.tolist(), "value": (1 + t - .5 * t ** 2).tolist(),
            "ids": [f"development-{i}" for i in range(len(t))], "source": "fixture", "goal": "test"}


def basis():
    return [[1.], [0., 1.], [0., 0., .5], [0., 0., 0., 1 / 6], [0., 0., 0., 0., 1 / 24]]


@pytest.mark.parametrize("kind", ["segments", "continuous", "smooth"])
def test_independent_evaluator_and_boundary_agree(kind):
    candidate = fit({"kind": kind, "degree": 2, "cuts": [0.]}, fixture(), basis())
    assert candidate["status"] == "PROPOSED"
    probes = [-2., -1e-9, 0., 1e-9, 2., 3.]
    assert np.allclose(predict(candidate, probes), evaluate(candidate, probes), atol=1e-12)
    assert assess_development(candidate, fixture())["mse"] < 1e-20


def test_no_boundary_aliases_have_one_program_identity():
    programs = [normalize_program({"kind": kind, "degree": 2, "cuts": []}) for kind in ("segments", "continuous", "smooth")]
    assert programs[0] == programs[1] == programs[2]


def test_sparse_branch_is_rejected_without_imputing_evidence():
    candidate = fit({"kind": "segments", "degree": 4, "cuts": [-1.99]}, fixture(), basis())
    assert candidate["status"] == "REJECTED"


def test_reserved_values_do_not_change_fitted_weights_and_credit_cannot_repeat():
    public = fixture()
    candidate = fit({"kind": "segments", "degree": 2, "cuts": []}, public, basis())
    independent = {**fixture(), "ids": [f"credit-{i}" for i in range(60)]}
    baseline = [99.] * 60
    first = credit(candidate, independent, baseline, .1, set())
    validate_receipt(candidate, first)
    assert first["points"] > 0
    second = credit(candidate, independent, baseline, .1, set(first["credited_ids"]))
    assert second["points"] == 0
    changed = copy.deepcopy(independent)
    changed["value"] = [v + 50 for v in changed["value"]]
    assert credit(candidate, changed, baseline, .1, set())["points"] == 0
    assert candidate == fit({"kind": "segments", "degree": 2, "cuts": []}, public, basis())


def test_forged_stale_and_self_credit_are_rejected():
    candidate = fit({"kind": "segments", "degree": 2, "cuts": []}, fixture(), basis())
    receipt = credit(candidate, {**fixture(), "ids": [f"credit-{i}" for i in range(60)]}, [99.] * 60, .1, set())
    bad = copy.deepcopy(receipt)
    bad["points"] += 1
    with pytest.raises(ValueError, match="Forged"):
        validate_receipt(candidate, bad)
    stale = copy.deepcopy(candidate)
    stale["weights"][0] += 1
    with pytest.raises(ValueError, match="Stale"):
        validate_receipt(stale, receipt)
    self_credit = credit(candidate, fixture(), [99.] * 60, .1, set())
    with pytest.raises(ValueError, match="itself"):
        validate_receipt(candidate, self_credit)


def test_portfolio_deduplicates_same_assumption_predictions():
    candidate = fit({"kind": "segments", "degree": 2, "cuts": []}, fixture(), basis())
    candidate["assessment"] = assess_development(candidate, fixture())
    assert len(portfolio([candidate, copy.deepcopy(candidate)], fixture())) == 1


def test_investigation_resumes_same_rng_policy_optimizer_and_candidates(tmp_path, monkeypatch):
    import torch

    import experiments.gap_inquiry as inquiry

    class Session:
        improve = inquiry.GapSession.improve

        def __init__(self):
            self.owner = torch.nn.Module()
            self.owner.gap_policy = torch.nn.Linear(8, 1, dtype=torch.float64)
            torch.nn.init.zeros_(self.owner.gap_policy.weight)
            torch.nn.init.zeros_(self.owner.gap_policy.bias)
            self.optimizer = torch.optim.Adam(self.owner.gap_policy.parameters(), lr=.003)

    monkeypatch.setattr(inquiry, "inherited_basis", lambda owner: basis())
    whole = Session()
    a = inquiry.search(whole, fixture(), "learned", tmp_path / "whole", limit=48)
    split = Session()
    inquiry.search(split, fixture(), "learned", tmp_path / "split", limit=24)
    restored = Session()
    b = inquiry.search(restored, fixture(), "learned", tmp_path / "split", limit=48)
    assert a == b


@pytest.mark.parametrize("kind", ["segments", "continuous", "smooth"])
def test_consequences_differentiate_generated_coefficients_without_teaching_targets(kind):
    from experiments.gap_consequences import derivative

    candidate = fit({"kind": kind, "degree": 2, "cuts": [0.]}, fixture(), basis())
    for t in [-1.5, -.5, .5, 1.5]:
        assert abs(derivative(candidate, t, 0) - (1 + t - .5 * t * t)) < 1e-10
        assert abs(derivative(candidate, t, 1) - (1 - t)) < 1e-10
        assert abs(derivative(candidate, t, 2) + 1) < 1e-10
    with pytest.raises(ValueError, match="boundary"):
        derivative(candidate, 0., 2)
