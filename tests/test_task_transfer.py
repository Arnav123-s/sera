import numpy as np
import pytest

from experiments.task_transfer.core import (
    acquire,
    calibrate,
    candidate_models,
    choose,
    disjoint,
    evidence,
    features,
    learned_basis,
)


def test_features_have_explicit_domain_and_known_constant():
    assert features(np.zeros((2, 3))).shape == (2, 20)
    assert np.all(features([[0, 0, 0]])[:, 0] == 1)
    with pytest.raises(ValueError):
        features([[0, 0, 1.01]])
    with pytest.raises(ValueError):
        features([[0, float("nan"), 0]])


def test_no_reusing_evidence_roles_or_duplicates():
    first = evidence([[0, 0, 0]], [1], "fit")
    with pytest.raises(ValueError):
        disjoint(first, evidence([[0, 0, 0]], [2], "calibration"))
    with pytest.raises(ValueError):
        evidence([[0, 0, 0]]*64, [1]*64, "calibration")


def test_binomial_bound_and_calibration_failure_withhold():
    rng = np.random.default_rng(4389)
    x = rng.uniform(-1, 1, (64, 3))
    w = np.zeros(20)
    gate = calibrate(w, evidence(x, np.zeros(64), "calibration"), .05)
    assert gate["accepted"] and gate["risk_upper_95"] == pytest.approx(1-.05**(1/64))
    y = np.zeros(64)
    y[0] = 1
    assert not calibrate(w, evidence(x, y, "calibration"), .05)["accepted"]


def test_acquired_span_learns_new_coefficients_and_does_not_use_calibration():
    rng = np.random.default_rng(8751)
    hidden = np.linalg.qr(rng.normal(size=(20, 3)))[0]
    prior = [hidden@rng.normal(size=3) for _ in range(4)]
    basis, _ = learned_basis(prior)
    truth = hidden@np.array([.3, -.8, 1.2])
    x = rng.uniform(-1, 1, (92, 3))
    y = features(x)@truth
    fit = evidence(x[:12], y[:12], "fit")
    selection = evidence(x[12:28], y[12:28], "selection")
    calibration = evidence(x[28:], y[28:], "calibration")
    result = acquire(fit, selection, calibration, basis=basis)
    assert result["gate"]["accepted"]
    assert np.linalg.norm(np.array(result["selected"]["coefficients"])-truth) < 1e-8
    bad = evidence(x[28:], y[28:]+3, "calibration")
    rejected = acquire(fit, selection, bad, basis=basis)
    assert result["selected"]["coefficients"] == rejected["selected"]["coefficients"]
    assert not rejected["gate"]["accepted"]
    candidates = candidate_models(fit, selection, basis)
    assert choose(candidates, "scratch")["method"] not in ("subspace", "innovation")
