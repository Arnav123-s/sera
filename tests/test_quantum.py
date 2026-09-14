import pytest
import torch

from sera.models import delta_write
from sera.quantum import (
    EventInstrument,
    bloch_density,
    bounded_bloch,
    density_from_factor,
    density_residuals,
    partial_swap,
    reset_channel,
)


def test_partial_swap_agrees_with_independent_joint_matrix_calculation():
    torch.manual_seed(7)
    swap = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.complex128
    )
    for _ in range(30):
        r, u = bounded_bloch(torch.randn(2, 3, dtype=torch.float64))
        angle = torch.randn((), dtype=torch.float64)
        unitary = angle.cos() * torch.eye(4, dtype=torch.complex128) - 1j * angle.sin() * swap
        joint = torch.kron(bloch_density(r), bloch_density(u))
        evolved = (unitary @ joint @ unitary.mH).reshape(2, 2, 2, 2)
        reduced = torch.einsum("aibi->ab", evolved)
        efficient = bloch_density(partial_swap(r, u, angle))
        torch.testing.assert_close(efficient, reduced, atol=1e-12, rtol=1e-12)


def test_partial_swap_directional_gradients_and_bloch_ball():
    inputs = tuple(
        x.requires_grad_()
        for x in (
            bounded_bloch(torch.randn(5, 3, dtype=torch.float64)),
            bounded_bloch(torch.randn(5, 3, dtype=torch.float64)),
            torch.randn(5, 1, dtype=torch.float64),
        )
    )
    assert torch.autograd.gradcheck(partial_swap, inputs)
    assert partial_swap(*inputs).norm(dim=-1).max() <= 1 + 1e-12


def test_delta_exact_write_and_orthogonal_preservation():
    memory = torch.randn(2, 1, 3, 3, dtype=torch.float64)
    key = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64).expand(2, -1, -1)
    value = torch.randn(2, 1, 3, dtype=torch.float64)
    updated = delta_write(memory, key, value, torch.ones(2, 1), torch.ones(2, 1))
    torch.testing.assert_close(updated[..., 0], value)
    torch.testing.assert_close(updated[..., 1:], memory[..., 1:])


def test_channel_linearity_and_density_invariants():
    torch.manual_seed(8)
    rho = density_from_factor(torch.randn(3, 4, 2, dtype=torch.complex128))
    replacement = density_from_factor(torch.randn(3, 4, 1, dtype=torch.complex128))
    unitary, _ = torch.linalg.qr(torch.randn(3, 4, 4, dtype=torch.complex128))
    retain = torch.rand(3, 1, 1, dtype=torch.float64)
    updated = reset_channel(rho, unitary, replacement, retain)
    residual = density_residuals(updated)
    assert residual["trace_error"] < 1e-12
    assert residual["hermiticity"] < 1e-12
    assert residual["minimum_eigenvalue"] > -1e-12
    torch.testing.assert_close(reset_channel(3 * rho, unitary, replacement, retain), 3 * updated)
    with pytest.raises(ValueError):
        density_from_factor(torch.zeros(2, 2))


@pytest.mark.parametrize("complex_valued", [False, True])
def test_instrument_completeness_likelihood_and_gradients(complex_valued):
    torch.manual_seed(3)
    model = EventInstrument(complex_valued=complex_valued)
    events = torch.tensor([[0, 2, 1, 3]])
    k = model.kraus()
    torch.testing.assert_close((k.mH @ k).sum(0), torch.eye(4, dtype=k.dtype), atol=1e-6, rtol=1e-6)
    state = model.initial_state(1)
    for event in events[0]:
        state = k[event][None] @ state @ k[event].mH[None]
    joint_p = state.diagonal(dim1=-2, dim2=-1).sum(-1).real
    conditional_p = model(events)
    torch.testing.assert_close(conditional_p.prod(-1), joint_p, atol=1e-7, rtol=1e-5)
    (-conditional_p.log().sum()).backward()
    assert model.raw.grad is not None and torch.isfinite(model.raw.grad).all()
    assert model.raw.grad.abs().max() > 0


def test_instrument_impossible_observation_rejected():
    model = EventInstrument(dimension=2, vocabulary=2, complex_valued=False)
    k = torch.stack((torch.eye(2), torch.zeros(2, 2)))
    with pytest.raises(ValueError, match="impossible"):
        model.observe(model.initial_state(1), torch.tensor([1]), k)
