import numpy as np
import pytest
import torch

from experiments.parameter_cloud.core import (
    COMPLEX,
    REAL,
    Evidence,
    evolve,
    fit,
    grid_values,
    operators,
    propagator,
)


def test_operators_are_hermitian_and_do_not_wrap_the_parameter_domain():
    grid = grid_values(9)
    laplacian, dirac = operators(grid)
    torch.testing.assert_close(laplacian, laplacian.mH)
    torch.testing.assert_close(dirac, dirac.mH)
    assert laplacian[0, -1] == 0
    assert dirac[:2, -2:].abs().sum() == 0
    assert torch.linalg.eigvalsh(laplacian).min() > -1e-12


@pytest.mark.parametrize("kind", ["schrodinger", "dirac"])
def test_propagators_conserve_norm_and_reverse(kind):
    u = propagator(9, 2.0, 0.1, 8, kind)
    torch.testing.assert_close(u.mH @ u, torch.eye(len(u), dtype=COMPLEX), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_diagonal_loss_phase_alone_cannot_change_the_born_distribution(dimensions):
    grid = grid_values(9)
    potential = grid.square() if dimensions == 1 else (grid[:, None] - grid[None, :]).square()
    prior = torch.softmax(-grid.square() / 2, 0)
    if dimensions == 2:
        prior = prior[:, None] * prior[None, :]
    result = evolve(potential, grid, "unitary", mixing=0, phase=2, steps=7)
    torch.testing.assert_close(result["probability"], prior, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("method", ["schrodinger", "dephased", "imaginary", "diffusion", "dirac"])
def test_zero_kinetic_filter_equals_classical_bayes(method):
    grid = grid_values(9)
    potential = (grid[:, None] + grid[None, :] - 0.4).square() * 3
    expected = evolve(potential, grid, "filter")["probability"]
    actual = evolve(potential, grid, method, mixing=0, phase=0.3, steps=7)
    torch.testing.assert_close(actual["probability"], expected, atol=3e-14, rtol=1e-12)


def test_cloud_correlations_change_predictions_despite_identical_weight_marginals():
    x = torch.tensor([[1., 1.], [1., 1.], [-1., -1.]], dtype=REAL)
    y = torch.tensor([0.1, 0.1, -0.1], dtype=REAL)
    full = fit(Evidence(x, y), "filter", noise=.05, grid_size=33)
    independent = fit(Evidence(x, y), "marginal_product", noise=.05, grid_size=33)
    torch.testing.assert_close(full["mean"], independent["mean"])
    torch.testing.assert_close(full["covariance"].diag(), independent["covariance"].diag())
    direction = torch.ones(2, dtype=REAL)
    assert direction @ full["covariance"] @ direction < 0.01
    assert direction @ independent["covariance"] @ direction > 0.5


def test_training_rejects_query_labels_and_imagined_evidence():
    x, y = torch.ones(4, 2, dtype=REAL), torch.ones(4, dtype=REAL)
    for evidence in (Evidence(x, y, role="test"), Evidence(x, y, origin="imagined")):
        with pytest.raises(ValueError, match="Only observed support"):
            fit(evidence, "filter")


def test_support_curve_is_nested_and_query_streams_do_not_overlap():
    from experiments.parameter_cloud.study import problem
    small = problem("factor_six", "correlated", 9292, 8)
    large = problem("factor_six", "correlated", 9292, 32)
    for first, second in zip(small["support"], large["support"]):
        torch.testing.assert_close(first, second[:8], atol=0, rtol=0)
    seen = set()
    for partition in ("support", "validation", "familiar", "independent", "extrapolation"):
        design = large[partition][0]
        rows = {tuple(row) for row in design.tolist()}
        assert not seen.intersection(rows)
        seen.update(rows)


def test_independent_cloud_updates_all_six_weights_and_roundtrips(tmp_path):
    rng = torch.Generator().manual_seed(97)
    x = torch.randn(128, 6, generator=rng, dtype=REAL)
    truth = torch.tensor([.5, -.5, .25, -.25, 1., -1.], dtype=REAL)
    model = fit(Evidence(x, x @ truth), "filter", noise=.2, joint=False, grid_size=33, sweeps=24)
    torch.testing.assert_close(model["mean"], truth, atol=.02, rtol=0)
    assert model["work"]["coordinate_updates"] == 144
    target = tmp_path / "weights.npz"
    np.savez_compressed(target, state=model["state"].numpy(), mean=model["mean"].numpy())
    with np.load(target, allow_pickle=False) as checkpoint:
        np.testing.assert_array_equal(checkpoint["state"], model["state"].numpy())


@pytest.mark.parametrize("method", ["schrodinger", "dephased", "imaginary", "diffusion", "dirac", "unitary"])
def test_finite_mixed_cloud_is_normalized(method):
    grid = grid_values(9)
    potential = (grid[:, None] + .3 * grid[None, :] - .4).square()
    result = evolve(potential, grid, method, mixing=.03, steps=12)
    assert torch.isfinite(result["state"]).all()
    assert result["probability"].min() >= 0
    assert abs(float(result["probability"].sum()) - 1) < 1e-12
    assert result["max_unitary_norm_error"] < 1e-12
