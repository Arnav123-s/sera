"""Exact finite reference operations. No quantum hardware or speedup is implied."""

from __future__ import annotations

import torch
from torch import nn


def pauli(*, dtype=torch.complex128, device=None):
    return torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=dtype, device=device
    )


def bloch_density(vector: torch.Tensor) -> torch.Tensor:
    sigma = pauli(
        dtype=torch.complex128 if vector.dtype == torch.float64 else torch.complex64,
        device=vector.device,
    )
    identity = torch.eye(2, dtype=sigma.dtype, device=vector.device)
    return (identity + torch.einsum("...k,kij->...ij", vector.to(sigma.dtype), sigma)) / 2


def bounded_bloch(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.sqrt(1 + vector.square().sum(-1, keepdim=True))


def partial_swap(
    retained: torch.Tensor, incoming: torch.Tensor, angle: torch.Tensor, *, cross_term: bool = True
) -> torch.Tensor:
    c, s = angle.cos(), angle.sin()
    result = c.square() * retained + s.square() * incoming
    if cross_term:
        result = result - c * s * torch.linalg.cross(retained, incoming, dim=-1)
    return result


def density_from_factor(factor: torch.Tensor) -> torch.Tensor:
    norm = factor.abs().square().sum(dim=(-2, -1), keepdim=True)
    if torch.any(norm <= 0) or not torch.isfinite(norm).all():
        raise ValueError("A density factor must have finite nonzero norm")
    return factor @ factor.mH / norm


def reset_channel(
    rho: torch.Tensor, unitary: torch.Tensor, replacement: torch.Tensor, retention: torch.Tensor
) -> torch.Tensor:
    """Linear CPTP extension includes trace(rho), even for unnormalized operators."""
    if torch.any((retention < 0) | (retention > 1)):
        raise ValueError("Channel retention must be in [0,1]")
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    return retention * (unitary @ rho @ unitary.mH) + (1 - retention) * trace * replacement


def density_residuals(rho: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(rho).all():
        raise ValueError("Nonfinite density matrix")
    return {
        "hermiticity": float((rho - rho.mH).abs().max().detach()),
        "trace_error": float((rho.diagonal(dim1=-2, dim2=-1).sum(-1) - 1).abs().max().detach()),
        "minimum_eigenvalue": float(torch.linalg.eigvalsh((rho + rho.mH) / 2).min().detach()),
    }


class EventInstrument(nn.Module):
    """A QR-normalized Kraus stack shares observation and generation semantics.

    Full-column-rank QR is required; rank collapse is rejected. QR is computed
    once per sequence so that a sequence uses one consistent instrument.
    """

    def __init__(self, dimension: int = 4, vocabulary: int = 4, *, complex_valued=True):
        super().__init__()
        if dimension < 2 or vocabulary < 2:
            raise ValueError("Instrument dimensions must be at least two")
        self.dimension, self.vocabulary = dimension, vocabulary
        dtype = torch.complex64 if complex_valued else torch.float32
        self.raw = nn.Parameter(torch.randn(vocabulary * dimension, dimension, dtype=dtype))

    def kraus(self):
        q, r = torch.linalg.qr(self.raw, mode="reduced")
        if r.diagonal().abs().min().detach() < 1e-7:
            raise ValueError("Rank-deficient instrument parametrization")
        return q.reshape(self.vocabulary, self.dimension, self.dimension)

    def initial_state(self, batch: int):
        return (
            torch.eye(self.dimension, dtype=self.raw.dtype, device=self.raw.device)
            .expand(batch, -1, -1)
            .clone()
            / self.dimension
        )

    def branches(self, rho, kraus=None):
        k = self.kraus() if kraus is None else kraus
        return k[None] @ rho[:, None] @ k.mH[None]

    def probabilities(self, rho, kraus=None):
        branches = self.branches(rho, kraus)
        return branches.diagonal(dim1=-2, dim2=-1).sum(-1).real

    def observe(self, rho, events, kraus=None):
        branches = self.branches(rho, kraus)
        selected = branches[torch.arange(len(events), device=rho.device), events]
        p = selected.diagonal(dim1=-2, dim2=-1).sum(-1).real
        if torch.any(p <= 0) or not torch.isfinite(p).all():
            raise ValueError("Cannot condition on an impossible or nonfinite event")
        return selected / p[:, None, None], p

    def forward(self, events):
        k = self.kraus()
        rho = self.initial_state(events.shape[0])
        probabilities = []
        for event in events.unbind(1):
            rho, p = self.observe(rho, event, k)
            probabilities.append(p)
        return torch.stack(probabilities, dim=1)

    @torch.no_grad()
    def generate(self, length: int, *, seed: int = 0):
        if length < 1:
            raise ValueError("Generation length must be positive")
        generator = torch.Generator(device=self.raw.device).manual_seed(seed)
        k = self.kraus()
        rho = self.initial_state(1)
        output = []
        for _ in range(length):
            p = self.probabilities(rho, k).clamp_min(0)
            event = torch.multinomial(p, 1, generator=generator).squeeze(-1)
            rho, _ = self.observe(rho, event, k)
            output.append(int(event.item()))
        return output
