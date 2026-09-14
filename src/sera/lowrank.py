"""The handbook-sized R1 state preset and explicitly measured density truncation."""

from __future__ import annotations

import torch
from torch import nn

from sera.models import DeltaMemory, Rotor


def hadamard(dimension):
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("The fixed mixing dimension must be a power of two")
    matrix = torch.ones(1, 1)
    while len(matrix) < dimension:
        matrix = torch.kron(matrix, torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
    return matrix / dimension**0.5


class LowRankWorkspaces(nn.Module):
    def __init__(self, width, *, count=4, dimension=16, rank=4):
        super().__init__()
        if not 1 <= rank < dimension or count < 1:
            raise ValueError("Invalid workspace rank or count")
        self.count, self.dimension, self.rank = count, dimension, rank
        self.phase = nn.Linear(width, count * dimension)
        self.prepare = nn.Linear(width, count * 2 * dimension)
        self.gate = nn.Linear(width, count)
        self.output = nn.Linear(count * dimension * 2, width)
        self.register_buffer("mix", hadamard(dimension).to(torch.complex64))
        self.last_discarded_mass = 0.0

    def initial(self, reference):
        diagonal = torch.linspace(1.0, 0.5, self.rank, device=reference.device)
        factor = torch.zeros(self.dimension, self.rank, dtype=torch.complex64, device=reference.device)
        factor[:self.rank] = torch.diag(diagonal).to(factor.dtype)
        factor = factor / factor.abs().square().sum().sqrt()
        return factor.expand(len(reference), self.count, -1, -1).clone()

    def step(self, encoded, factor, write):
        b = len(encoded)
        phase = self.phase(encoded).reshape(b, self.count, self.dimension)
        diagonal = torch.complex(phase.cos(), phase.sin())
        rotated = self.mix @ (diagonal[..., :, None] * (self.mix @ factor))
        raw = self.prepare(encoded).reshape(b, self.count, self.dimension, 2)
        vector = torch.complex(raw[..., 0], raw[..., 1])
        vector = torch.cat((vector[..., :-1], torch.ones_like(vector[..., -1:])), -1)
        vector = vector / vector.abs().square().sum(-1, keepdim=True).sqrt()
        alpha = self.gate(encoded).sigmoid()
        expanded = torch.cat((alpha[..., None, None].sqrt() * rotated,
                              (1 - alpha)[..., None, None].sqrt() * vector[..., None]), -1)
        u, singular, _ = torch.linalg.svd(expanded, full_matrices=False)
        discarded = singular[..., self.rank:].square().sum(-1)
        self.last_discarded_mass = float(discarded.max().detach())
        retained = u[..., :self.rank] * singular[..., None, :self.rank]
        retained = retained / retained.abs().square().sum((-2, -1), keepdim=True).sqrt()
        # Query bypass preserves the factor itself; no rank approximation is applied to a query.
        factor = write[..., None, None] * retained + (1 - write[..., None, None]) * factor
        primary = factor.abs().square().sum(-1)
        mixed = (self.mix @ factor).abs().square().sum(-1)
        return self.output(torch.cat((primary, mixed), -1).flatten(1)), factor


class ReferenceMemory(nn.Module):
    def __init__(self, width=256, heads=8, memory_dim=32, *, compact=False):
        super().__init__()
        self.width = width
        rotor_width = width // 2
        self.delta = DeltaMemory(width, heads, memory_dim)
        self.rotor_input = nn.Linear(width, rotor_width)
        self.rotor = Rotor(rotor_width)
        self.rotor_output = nn.Linear(rotor_width, width)
        self.density = LowRankWorkspaces(width, count=2 if compact else 4,
                                         dimension=8 if compact else 16, rank=2 if compact else 4)
        self.router = nn.Linear(width, 3)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.encoder = nn.Sequential(nn.Linear(22, width), nn.Tanh())

    def initial_state(self, batch):
        ref = next(self.parameters()).new_zeros(batch, self.width)
        return {"delta": self.delta.initial(ref), "rotor": self.rotor.initial(ref),
                "density": self.density.initial(ref)}

    def step(self, state, encoded, *, write):
        d, ds = self.delta.step(encoded, state["delta"], write)
        r, rs = self.rotor.step(self.rotor_input(encoded), state["rotor"], write)
        q, qs = self.density.step(encoded, state["density"], write)
        outputs = torch.stack([norm(value) for norm, value in
                               zip(self.norms, (d, self.rotor_output(r), q))], 1)
        hidden = (outputs * self.router(encoded).softmax(-1)[..., None]).sum(1)
        return hidden, {"delta": ds, "rotor": rs, "density": qs}
