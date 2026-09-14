"""The handbook-sized R1 state preset and explicitly measured density truncation."""

from __future__ import annotations

import copy

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
        self.last_diagnostics = None
        self.truncation_gradient = "spectral"

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
        if self.truncation_gradient == "frozen-projector":
            alpha = alpha.clamp(1e-6, 1-1e-6)
        expanded = torch.cat((alpha[..., None, None].sqrt() * rotated,
                              (1 - alpha)[..., None, None].sqrt() * vector[..., None]), -1)
        if self.truncation_gradient == "frozen-projector":
            # Preserve spectral truncation in the forward pass, while treating
            # its selected subspace as constant during each backward pass.
            with torch.no_grad():
                _, singular, vh = torch.linalg.svd(expanded, full_matrices=False)
            retained = expanded @ vh.conj().transpose(-1, -2)[..., :self.rank]
        else:
            u, singular, _ = torch.linalg.svd(expanded, full_matrices=False)
            retained = u[..., :self.rank] * singular[..., None, :self.rank]
        discarded = singular[..., self.rank:].square().sum(-1)
        self.last_discarded_mass = float(discarded.max().detach())
        applied_discard = discarded * write
        self.last_diagnostics = {"rank": self.rank, "dimension": self.dimension,
                                 "discarded_mass": applied_discard.detach().cpu().tolist(),
                                 "meaning": "Single normalized spectral truncation; no cumulative conditional-error guarantee"}
        retained = retained / retained.abs().square().sum((-2, -1), keepdim=True).sqrt()
        # Query bypass preserves the factor itself; no rank approximation is applied to a query.
        factor = write[..., None, None] * retained + (1 - write[..., None, None]) * factor
        primary = factor.abs().square().sum(-1)
        mixed = (self.mix @ factor).abs().square().sum(-1)
        return self.output(torch.cat((primary, mixed), -1).flatten(1)), factor


class ReferenceMemory(nn.Module):
    def __init__(self, width=256, heads=8, memory_dim=32, *, compact=False, routing="all",
                 density_rank=None, density_dimension=None, density_count=None):
        super().__init__()
        if routing not in {"all", "top1"}:
            raise ValueError("Unknown event routing policy")
        self.width = width
        self.routing = routing
        rotor_width = width // 2
        self.delta = DeltaMemory(width, heads, memory_dim)
        self.rotor_input = nn.Linear(width, rotor_width)
        self.rotor = Rotor(rotor_width)
        self.rotor_output = nn.Linear(rotor_width, width)
        self.density = LowRankWorkspaces(width, count=density_count or (2 if compact else 4),
                                         dimension=density_dimension or (8 if compact else 16),
                                         rank=density_rank or (2 if compact else 4))
        self.router = nn.Linear(width, 3)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.encoder = nn.Sequential(nn.Linear(22, width), nn.Tanh())
        self.last_step_diagnostics = None
        self.diagnostic_history = None

    def begin_diagnostics(self):
        self.diagnostic_history = []

    def end_diagnostics(self):
        result, self.diagnostic_history = self.diagnostic_history or [], None
        return result

    def initial_state(self, batch):
        ref = next(self.parameters()).new_zeros(batch, self.width)
        return {"delta": self.delta.initial(ref), "rotor": self.rotor.initial(ref),
                "density": self.density.initial(ref)}

    def step(self, state, encoded, *, write, address=None, address_mask=None):
        weights = self.router(encoded).softmax(-1)
        selected = weights.argmax(-1)
        hidden, next_state, evaluated = torch.zeros_like(encoded), dict(state), {}
        density_diagnostics = None
        for index, (name, core) in enumerate((("delta", self.delta), ("rotor", self.rotor),
                                             ("density", self.density))):
            rows = (torch.arange(len(encoded), device=encoded.device) if self.routing == "all"
                    else (selected == index).nonzero().flatten())
            evaluated[name] = len(rows)
            if not len(rows):
                continue
            inputs = self.rotor_input(encoded[rows]) if name == "rotor" else encoded[rows]
            options = ({"address": address[rows] if address is not None else None,
                        "address_mask": address_mask[rows] if address_mask is not None else None}
                       if name == "delta" else {})
            output, updated = core.step(inputs, state[name][rows], write[rows], **options)
            if name == "rotor":
                output = self.rotor_output(output)
            next_state[name] = state[name].index_copy(0, rows, updated)
            gate = weights[rows, index]
            if self.routing == "top1":
                # Hard event delivery uses a declared straight-through routing gradient.
                gate = gate / gate.detach().clamp_min(1e-12)
            hidden = hidden.index_add(0, rows, self.norms[index](output) * gate[:, None])
            if name == "density":
                density_diagnostics = {"batch_rows": rows.tolist(), **copy.deepcopy(core.last_diagnostics)}
        self.last_step_diagnostics = {"routing": self.routing, "branch_rows_executed": evaluated,
                                      "density": density_diagnostics}
        if self.diagnostic_history is not None:
            if len(self.diagnostic_history) >= 4096:
                raise ValueError("Flush the bounded diagnostic history before continuing")
            self.diagnostic_history.append(copy.deepcopy(self.last_step_diagnostics))
        return hidden, next_state
