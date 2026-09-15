"""Loss-informed amplitude dynamics, with classical probability controls.

The independent-weight approximation is explicit. No quantum hardware, physical
electron model, or calibrated-posterior claim is implied by complex amplitudes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

REAL = torch.float64
COMPLEX = torch.complex128
CLOUD_METHODS = ("filter", "schrodinger", "dephased", "imaginary", "diffusion", "dirac", "unitary")


@dataclass(frozen=True)
class Evidence:
    design: torch.Tensor
    targets: torch.Tensor
    role: str = "support"
    origin: str = "observed"

    def admitted(self):
        x, y = self.design, self.targets
        if self.role != "support" or self.origin != "observed":
            raise ValueError("Only observed support can train a weight cloud")
        if x.ndim != 2 or y.shape != (len(x),) or not len(x):
            raise ValueError("Invalid support shapes")
        if x.dtype != REAL or y.dtype != REAL or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Expected finite float64 observations")
        return x, y


def grid_values(size=65, limit=2.0):
    if size < 3 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid finite parameter domain")
    return torch.linspace(-limit, limit, size, dtype=REAL)


def operators(grid):
    """Open-chain Laplacian and a declared finite Dirac-type analogue.

    P is the Hermitian centered difference -i d/dq, with no wraparound.
    H_D = P tensor sigma_x + I tensor sigma_z, in q-major spin ordering.
    It is a finite artificial Hamiltonian, not a relativistic boundary solver.
    """
    n = len(grid)
    h = float(grid[1] - grid[0])
    edge = torch.diag(torch.ones(n - 1, dtype=REAL), 1)
    adjacency = edge + edge.T
    laplacian = (torch.diag(adjacency.sum(1)) - adjacency) / h**2
    momentum = -1j * (edge - edge.T).to(COMPLEX) / (2 * h)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX)
    dirac = torch.kron(momentum, sigma_x) + torch.kron(torch.eye(n, dtype=COMPLEX), sigma_z)
    return laplacian, dirac


@lru_cache(maxsize=128)
def propagator(size, limit, mixing, steps, kind):
    laplacian, dirac = operators(grid_values(size, limit))
    operator = dirac if kind == "dirac" else laplacian
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    scale = mixing / steps
    if kind in {"imaginary", "diffusion"}:
        factors = torch.exp(-scale * eigenvalues)
        return (eigenvectors * factors) @ eigenvectors.mH
    factors = torch.exp(-1j * scale * eigenvalues)
    return (eigenvectors.to(COMPLEX) * factors) @ eigenvectors.to(COMPLEX).mH


def normed(state, probability=False):
    total = state.sum() if probability else state.abs().square().sum()
    if not torch.isfinite(total) or total <= 0:
        raise FloatingPointError("Cloud normalization failed")
    return state / (total if probability else total.sqrt())


def apply_axes(state, operator):
    result = operator @ state
    return result @ operator.T if state.ndim == 2 else result


def evolve(potential, grid, method, *, mixing=0.003, phase=1.0, steps=16):
    """Temper a likelihood once; the kinetic branches alter that posterior.

    Potential contains support negative log likelihood, not an answer oracle.
    A Gaussian prior is shared by every method. Filtering has total beta=1.
    """
    if method not in CLOUD_METHODS or potential.ndim not in (1, 2):
        raise ValueError("Unknown method or unsupported joint dimension")
    if potential.shape != (len(grid),) * potential.ndim or not torch.isfinite(potential).all():
        raise ValueError("Invalid potential")
    if steps < 1 or mixing < 0 or phase < 0:
        raise ValueError("Invalid dynamics budget")
    dimensions, n = potential.ndim, len(grid)
    prior = torch.softmax(-grid.square() / 2, dim=0)
    if dimensions == 2:
        prior = prior[:, None] * prior[None, :]
    energy = potential - potential.min()
    if method == "filter":
        probability = torch.softmax((prior.log() - energy).flatten(), dim=0).reshape(prior.shape)
        return {"probability": probability, "state": probability.clone(),
                "max_unitary_norm_error": 0.0, "operator_bytes": 0, "propagation_steps": 0}
    probability_mode = method == "diffusion"
    state = prior.clone() if probability_mode else prior.sqrt()
    if method != "imaginary" and not probability_mode:
        state = state.to(COMPLEX)
    if method == "dirac":
        if dimensions == 1:
            expanded = torch.zeros(n, 2, dtype=COMPLEX)
            expanded[:, 0] = state
            state = expanded.flatten()
            energy = energy.repeat_interleave(2)
        else:
            expanded = torch.zeros(n, 2, n, 2, dtype=COMPLEX)
            expanded[:, 0, :, 0] = state
            state = expanded.reshape(2 * n, 2 * n)
            energy = energy.repeat_interleave(2, 0).repeat_interleave(2, 1)
    kind = method if method in {"dirac", "imaginary", "diffusion"} else "schrodinger"
    operator = propagator(n, float(grid[-1]), float(mixing), int(steps), kind)
    if state.is_complex():
        operator = operator.to(COMPLEX)
    phase_step = torch.exp(-0.5j * phase * energy / steps)
    filter_step = torch.exp(-energy / (steps * (1 if probability_mode else 2)))
    maximum_error = 0.0
    for _ in range(steps):
        if method in {"schrodinger", "dephased", "dirac", "unitary"}:
            before = state.abs().square().sum()
            state = phase_step * apply_axes(phase_step * state, operator)
            maximum_error = max(maximum_error, float((state.abs().square().sum() - before).abs()))
        else:
            state = apply_axes(state, operator)
        if method != "unitary":
            state = state * filter_step
        if probability_mode:
            if state.min() < -1e-10:
                raise FloatingPointError("Heat propagation produced negative mass")
            state = state.clamp_min(0)
        if method == "dephased":
            # Deliberate phase-erasure ablation; this is not unitary evolution.
            state = state.abs().to(COMPLEX)
        state = normed(state, probability=probability_mode)
    probability = state if probability_mode else state.abs().square()
    if method == "dirac":
        probability = (probability.reshape(n, 2).sum(1) if dimensions == 1
                       else probability.reshape(n, 2, n, 2).sum((1, 3)))
    return {"probability": probability, "state": state,
            "max_unitary_norm_error": maximum_error,
            "operator_bytes": operator.numel() * operator.element_size(),
            "propagation_steps": steps * dimensions}


def sufficient_statistics(evidence, noise):
    x, y = evidence.admitted()
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("Invalid supplied observation noise")
    return x.T @ x / noise**2, x.T @ y / noise**2, y.square().sum() / noise**2


def fit(evidence, method, *, noise=0.2, grid_size=65, limit=2.0,
        mixing=0.003, phase=1.0, steps=16, sweeps=24, joint=True):
    gram, rhs, yy = sufficient_statistics(evidence, noise)
    dimensions = len(rhs)
    work = {"support_rows": len(evidence.targets), "design_scalars": evidence.design.numel(),
            "gram_scalar_products": len(evidence.targets) * dimensions**2,
            "conditional_potential_values": 0, "joint_potential_values": 0,
            "gradient_steps": 0, "propagation_steps": 0, "coordinate_updates": 0}
    precision = gram + torch.eye(dimensions, dtype=REAL)
    if method in {"gaussian", "gaussian_diagonal", "gradient_map"}:
        covariance = torch.linalg.solve(precision, torch.eye(dimensions, dtype=REAL))
        mean = covariance @ rhs
        if method == "gaussian_diagonal":
            covariance = torch.diag(covariance.diag())
        if method == "gradient_map":
            mean = torch.zeros_like(rhs)
            learning_rate = 1 / torch.linalg.eigvalsh(precision).max()
            for _ in range(256):
                mean = mean - learning_rate * (precision @ mean - rhs)
            work["gradient_steps"] = 256
            covariance = torch.zeros_like(covariance)
        return {"mean": mean, "covariance": covariance, "state": mean.clone(), "work": work,
                "max_unitary_norm_error": 0.0, "operator_bytes": 0, "boundary_mass": 0.0}
    grid = grid_values(grid_size, limit)
    maximum_error, operator_bytes = 0.0, 0
    if joint:
        if dimensions != 2:
            raise ValueError("Exact joint cloud is deliberately restricted to two weights")
        points = torch.cartesian_prod(grid, grid)
        potential = (0.5 * torch.einsum("bi,ij,bj->b", points, gram, points) - points @ rhs + yy / 2).reshape(grid_size, grid_size)
        work["joint_potential_values"] = points.shape[0]
        result = evolve(potential, grid, "filter" if method == "marginal_product" else method,
                        mixing=mixing, phase=phase, steps=steps)
        probability = result["probability"]
        if method == "marginal_product":
            probability = probability.sum(1)[:, None] * probability.sum(0)[None, :]
            result["state"] = torch.stack([probability.sum(1), probability.sum(0)])
        weights = probability.flatten()
        mean = weights @ points
        centered = points - mean
        covariance = centered.T @ (weights[:, None] * centered)
        boundary = 1 - probability[1:-1, 1:-1].sum()
        work["propagation_steps"] = result["propagation_steps"]
        return {"mean": mean, "covariance": covariance, "grid": grid,
                "probability": probability, "state": result["state"], "points": points,
                "work": work, "max_unitary_norm_error": result["max_unitary_norm_error"],
                "operator_bytes": result["operator_bytes"], "boundary_mass": float(boundary)}
    if method == "marginal_product":
        raise ValueError("Joint marginal ablation is not an independent learning algorithm")
    mean = torch.zeros(dimensions, dtype=REAL)
    marginals = torch.softmax(-grid.square() / 2, 0).repeat(dimensions, 1)
    states = []
    for _ in range(sweeps):
        states = []
        for coordinate in range(dimensions):
            # Other-factor variance contributes only a constant to this quadratic.
            conditional_rhs = rhs[coordinate] - gram[coordinate] @ mean + gram[coordinate, coordinate] * mean[coordinate]
            potential = gram[coordinate, coordinate] * grid.square() / 2 - conditional_rhs * grid
            result = evolve(potential, grid, method, mixing=mixing, phase=phase, steps=steps)
            marginals[coordinate] = result["probability"]
            mean[coordinate] = marginals[coordinate] @ grid
            states.append(result["state"])
            maximum_error = max(maximum_error, result["max_unitary_norm_error"])
            operator_bytes = max(operator_bytes, result["operator_bytes"])
            work["coordinate_updates"] += 1
            work["conditional_potential_values"] += grid_size
            work["propagation_steps"] += result["propagation_steps"]
    variance = (marginals * (grid[None, :] - mean[:, None]).square()).sum(1)
    return {"mean": mean, "covariance": torch.diag(variance), "grid": grid,
            "marginals": marginals, "state": torch.stack(states), "work": work,
            "max_unitary_norm_error": maximum_error, "operator_bytes": operator_bytes,
            "boundary_mass": float(marginals[:, [0, -1]].sum(1).max())}


def score(model, design, truth, observed, noise):
    """Score fresh observations; intervals are explicitly Gaussian moment bands."""
    prediction = design @ model["mean"]
    variance = torch.einsum("bi,ij,bj->b", design, model["covariance"], design).clamp_min(0) + noise**2
    if "probability" in model:
        points, log_weight = model["points"], model["probability"].flatten().clamp_min(1e-300).log()
        values = []
        for start in range(0, len(design), 64):
            candidate_predictions = design[start:start + 64] @ points.T
            log_likelihood = -(observed[start:start + 64, None] - candidate_predictions).square() / (2 * noise**2) - math.log(noise * math.sqrt(2 * math.pi))
            values.append(-torch.logsumexp(log_likelihood + log_weight[None, :], dim=1))
        nll = torch.cat(values)
        nll_kind = "exact finite Gaussian mixture"
    else:
        nll = 0.5 * ((observed - prediction).square() / variance + variance.log() + math.log(2 * math.pi))
        nll_kind = "Gaussian moment approximation" if "marginals" in model else "Gaussian"
    errors = (prediction - truth).square()
    covered = (observed - prediction).abs() <= 1.959963984540054 * variance.sqrt()
    return {"mse": float(errors.mean()), "nll": float(nll.mean()), "nll_kind": nll_kind,
            "moment_95_coverage": float(covered.to(REAL).mean()),
            "mean_predictive_std": float(variance.sqrt().mean()),
            "squared_errors": errors, "nll_values": nll, "covered": covered,
            "predictions": prediction, "predictive_variances": variance}
