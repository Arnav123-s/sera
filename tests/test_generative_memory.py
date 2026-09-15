"""Reference mathematics, scientific separation and decoded-artifact invariants."""

import json

import numpy as np
import pytest
import torch

from experiments.generative_memory.core import (
    CIRCLE,
    ELLIPSE,
    METHODS,
    Observations,
    canonical,
    design,
    fit,
    fit_linear,
    linear_predict,
    metrics,
    predict,
)
from experiments.generative_memory.data import bank


def evidence():
    result = []
    for role, count in (("support", 24), ("selection", 16), ("calibration", 32)):
        rows = bank(100, "circle", .02, role, count)
        result.append(Observations(rows["t"], rows["y"], role))
    return result


def test_posterior_and_evidence_against_observation_space_gaussian():
    rows = evidence()[0]
    model = fit_linear(CIRCLE, rows, .02)
    a, y = design(CIRCLE, rows.t), rows.y.ravel()
    marginal = .02 ** 2 * np.eye(len(y)) + 4 * a @ a.T
    mean = 4 * a.T @ np.linalg.solve(marginal, y)
    covariance = 4 * np.eye(a.shape[1]) - 16 * a.T @ np.linalg.solve(marginal, a)
    np.testing.assert_allclose(model["mean"], mean, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(model["covariance"], covariance, rtol=2e-6, atol=1e-9)
    logp = -.5 * (len(y)*np.log(2*np.pi) + np.linalg.slogdet(marginal)[1] + y@np.linalg.solve(marginal, y))
    assert abs(model["log_evidence"] - logp) < 1e-7


def test_circle_design_obeys_constant_radius():
    t = np.linspace(-2, 2, 111)
    theta = np.array([.3, -.2, .7, .4])
    points = (design(CIRCLE, t) @ theta).reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(points - theta[:2], axis=1), np.hypot(*theta[2:]))
    assert design(ELLIPSE, t).shape == (222, 6)


@pytest.mark.parametrize("origin", ["imagined", "prediction"])
def test_hypothetical_rows_cannot_train(origin):
    rows = evidence()
    bad = Observations(rows[0].t, rows[0].y, "support", origin)
    with pytest.raises(ValueError):
        fit("C_circle", bad, rows[1], rows[2], .02)


def test_query_role_and_overlap_rejected():
    support, selection, calibration = evidence()
    with pytest.raises(ValueError):
        fit("C_circle", Observations(support.t, support.y, "query"), selection, calibration, .02)
    with pytest.raises(ValueError):
        fit("C_circle", support, Observations(support.t, support.y, "selection"), calibration, .02)


def test_evidence_copies_and_freezes_caller_arrays():
    x, y = np.arange(3.), np.ones((3, 2))
    rows = Observations(x, y, "support")
    x[0], y[0, 0] = 99, 99
    assert rows.t[0] == 0 and rows.y[0, 0] == 1
    with pytest.raises(ValueError):
        rows.y[0, 0] = 5


@pytest.mark.parametrize("method", METHODS)
def test_all_controls_decode_without_training_state(method):
    torch.set_num_threads(1)
    artifact, work = fit(method, *evidence(), .02, steps=20)
    frozen = canonical(artifact)
    restored = json.loads(frozen)
    t = np.linspace(-1.5, 1.5, 50)
    original, fresh = predict(artifact, t), predict(restored, t)
    for key in original:
        np.testing.assert_array_equal(original[key], fresh[key])
    assert canonical(artifact) == frozen
    assert np.isfinite(fresh["mean"]).all() and (fresh["variance"] > 0).all()
    assert work["support_observations"] == 24


def test_exact_lookup_has_no_unseen_label_side_channel():
    rows = evidence()
    artifact, _ = fit("A_episodic", *rows, .02)
    recall = predict(artifact, rows[0].t)
    np.testing.assert_allclose(recall["mean"], rows[0].y, atol=1e-7)
    unknown = predict(artifact, np.array([3.14]))
    assert not unknown["available"][0]
    np.testing.assert_equal(unknown["mean"], 0)


def test_mixture_includes_between_model_variance():
    artifact, _ = fit("D_family", *evidence(), .02)
    artifact["class_weights"] = [.5, .25, .25]
    prediction = predict(artifact, np.array([1.5]))
    within = np.einsum("k,kij->ij", prediction["class_weights"], prediction["component_variances"])
    assert np.all(prediction["variance"] >= within - 1e-12)
    assert np.any(prediction["variance"] > within + 1e-4)


def test_random_values_do_not_reuse_seed_as_learner_input():
    a, b = bank(101, "random", .02, "support", 64), bank(101, "random", .02, "interpolation", 128)
    assert set(a["t"]).isdisjoint(b["t"])
    assert a["truth"].shape == (64, 2)
    # Fit API admits only observed coordinate/value banks and an initialization seed.
    assert not {"seed", "family", "environment"} & set(a)


def test_single_model_marginal_nll_matches_normal_formula():
    model = fit_linear(CIRCLE, evidence()[0], .02)
    mu, variance = linear_predict(model, np.array([.1, .2]))
    prediction = {"mean": mu, "variance": variance, "component_means": mu[None],
                  "component_variances": variance[None], "class_weights": np.array([1.]),
                  "available": np.ones(2, dtype=bool)}
    actual = mu + .03
    score = metrics(prediction, mu, actual)
    expected = .5 * np.mean(np.log(2*np.pi*variance) + .03**2/variance)
    assert abs(score["marginal_nll"] - expected) < 1e-12
