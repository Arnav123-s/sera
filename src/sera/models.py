"""Interchangeable stateful cores with explicit session state.

The default is targeted associative memory. Complex/density branches are
experimental additions. Batch forward resets state; streaming owns it explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from sera.contracts import StateOwner
from sera.data import INPUT_DIM
from sera.quantum import bounded_bloch, partial_swap, reset_channel

KINDS = ("delta", "gru", "rotor", "real", "collision", "collision_no_cross", "density", "hybrid")


def delta_write(memory, key, value, retention, strength):
    """Rank-one corrective write; key must be unit norm for exact replacement."""
    decayed = retention[..., None, None] * memory
    error = value - torch.einsum("bhvk,bhk->bhv", decayed, key)
    return decayed + strength[..., None, None] * error[..., :, None] * key[..., None, :]


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "delta"
    width: int = 32
    heads: int = 2
    memory_dim: int = 8

    def __post_init__(self):
        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:
            raise ValueError("Invalid model configuration")


class DeltaMemory(nn.Module):
    def __init__(self, width, heads, dimension):
        super().__init__()
        self.heads, self.dimension = heads, dimension
        self.key = nn.Linear(width, heads * dimension)
        self.query = nn.Linear(width, heads * dimension)
        self.value = nn.Linear(width, heads * dimension)
        self.gates = nn.Linear(width, heads * 2)
        self.output = nn.Linear(heads * dimension, width)
        with torch.no_grad():
            self.gates.bias[:heads].fill_(3.0)

    def initial(self, e):
        return e.new_zeros(len(e), self.heads, self.dimension, self.dimension)

    def step(self, e, memory, write, *, address=None, address_mask=None):
        shape = (len(e), self.heads, self.dimension)
        key_source, query_projection = e, self.query(e)
        if address is not None:
            if address.shape != e.shape or address_mask is None or address_mask.shape != (len(e), 1):
                raise ValueError("Explicit addresses must align with event rows")
            key_source = torch.where(address_mask, address, e)
            query_projection = torch.where(address_mask, self.key(address), query_projection)
        key = F.normalize(self.key(key_source).reshape(shape), dim=-1, eps=1e-8)
        query = F.normalize(query_projection.reshape(shape), dim=-1, eps=1e-8)
        value = self.value(e).tanh().reshape(shape)
        alpha, beta = self.gates(e).sigmoid().chunk(2, -1)
        alpha = 1 - write * (1 - alpha)
        beta = write * beta
        memory = delta_write(memory, key, value, alpha, beta)
        output = torch.einsum("bhvk,bhk->bhv", memory, query).flatten(1)
        return self.output(output), memory


class Rotor(nn.Module):
    def __init__(self, width, *, phase=True):
        super().__init__()
        self.width, self.phase = width, phase
        # Identical parameter count for the no-phase control.
        self.controls = nn.Linear(width, 4 * width)
        self.output = nn.Linear(2 * width, width)

    def initial(self, e):
        return e.new_zeros(len(e), self.width, 2)

    def step(self, e, state, write):
        gate, angle, real, imaginary = self.controls(e).chunk(4, -1)
        gate = gate.sigmoid()
        angle = torch.pi * angle.tanh() if self.phase else angle * 0
        c, s = angle.cos(), angle.sin()
        rotated = torch.stack(
            (c * state[..., 0] - s * state[..., 1], s * state[..., 0] + c * state[..., 1]), -1
        )
        proposal = torch.stack((real.tanh(), imaginary.tanh()), -1)
        updated = gate[..., None] * rotated + (1 - gate[..., None]) * proposal
        state = write[..., None] * updated + (1 - write[..., None]) * state
        return self.output(state.flatten(1)), state


class CollisionMemory(nn.Module):
    def __init__(self, width, *, cross=True):
        super().__init__()
        self.width, self.cross = width, cross
        self.input = nn.Linear(width, width * 3)
        self.angles = nn.Linear(width, width)
        self.output = nn.Linear(width * 3, width)

    def initial(self, e):
        return e.new_zeros(len(e), self.width, 3)

    def step(self, e, state, write):
        incoming = bounded_bloch(self.input(e).reshape(len(e), self.width, 3))
        angle = self.angles(e).tanh()[..., None]
        updated = partial_swap(state, incoming, angle, cross_term=self.cross)
        state = write[..., None] * updated + (1 - write[..., None]) * state
        return self.output(state.flatten(1)), state


class DensityMemory(nn.Module):
    """Exact 4x4 density workspace, not a claim of compressed arbitrary joint state."""

    def __init__(self, width):
        super().__init__()
        self.phase = nn.Linear(width, 4)
        self.prepare = nn.Linear(width, 8)
        self.gate = nn.Linear(width, 1)
        self.output = nn.Linear(32, width)
        h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / 2**0.5
        self.register_buffer("mix", torch.kron(h, h).to(torch.complex64))

    def initial(self, e):
        dtype = torch.complex128 if e.dtype == torch.float64 else torch.complex64
        return torch.eye(4, dtype=dtype, device=e.device).expand(len(e), -1, -1).clone() / 4

    def step(self, e, state, write):
        angle = self.phase(e)
        phases = torch.complex(angle.cos(), angle.sin())
        unitary = self.mix.to(state.dtype) @ torch.diag_embed(phases) @ self.mix.to(state.dtype)
        raw = self.prepare(e).reshape(len(e), 4, 2)
        vector = torch.complex(raw[..., 0], raw[..., 1])
        # A fixed extra coordinate guarantees nonzero preparation even at zero input.
        vector = torch.cat((vector[:, :3], torch.ones_like(vector[:, 3:])), -1)
        vector = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        replacement = vector[..., :, None] * vector.conj()[..., None, :]
        updated = reset_channel(state, unitary, replacement, self.gate(e).sigmoid()[..., None])
        state = write[..., None] * updated + (1 - write[..., None]) * state
        features = torch.cat((state.real.flatten(1), state.imag.flatten(1)), -1)
        return self.output(features), state


class StatefulModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        w = config.width
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, w), nn.Tanh())
        self.cores = nn.ModuleDict()
        if config.kind in {"delta", "hybrid"}:
            self.cores["delta"] = DeltaMemory(w, config.heads, config.memory_dim)
        if config.kind in {"rotor", "real", "hybrid"}:
            self.cores["rotor"] = Rotor(w, phase=config.kind != "real")
        if config.kind in {"density", "hybrid"}:
            self.cores["density"] = DensityMemory(w)
        if config.kind in {"collision", "collision_no_cross"}:
            self.cores["collision"] = CollisionMemory(w, cross=config.kind == "collision")
        if config.kind == "gru":
            self.gru = nn.GRUCell(w, w)
        else:
            self.router = nn.Linear(w, len(self.cores))
            self.norms = nn.ModuleDict({name: nn.LayerNorm(w) for name in self.cores})
        self.decoder = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 4))

    def encode(self, observation):
        if observation.shape[-1] != INPUT_DIM or not torch.isfinite(observation).all():
            raise ValueError("Invalid symbolic observation features")
        return self.encoder(observation)

    def initial_state(self, batch: int):
        ref = next(self.parameters()).new_zeros(batch, self.config.width)
        if self.config.kind == "gru":
            return {"gru": ref}
        return {name: core.initial(ref) for name, core in self.cores.items()}

    def step(self, state, encoded, *, write, address=None, address_mask=None):
        if self.config.kind == "gru":
            hidden = self.gru(encoded, state["gru"])
            return hidden, {"gru": hidden}
        outputs, next_state = [], {}
        for name, core in self.cores.items():
            options = {"address": address, "address_mask": address_mask} if name == "delta" else {}
            output, next_state[name] = core.step(encoded, state[name], write, **options)
            outputs.append(self.norms[name](output))
        weights = self.router(encoded).softmax(-1)
        hidden = (torch.stack(outputs, 1) * weights[..., None]).sum(1)
        return hidden, next_state

    def read(self, hidden):
        return self.decoder(hidden)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, time, feature] input")
        encoded = self.encode(inputs)
        state = self.initial_state(inputs.shape[0])
        for t, event in enumerate(encoded.unbind(1)):
            hidden, state = self.step(state, event, write=1 - inputs[:, t, 14:15])
        return self.read(hidden)

    def state_bytes(self, batch=1):
        return sum(v.numel() * v.element_size() for v in self.initial_state(batch).values())


class Session:
    """Fast-state updates never optimize model parameters."""

    def __init__(self, model: StatefulModel, owner: StateOwner):
        owner.validate()
        self.model, self.owner = model, owner
        self.state = model.initial_state(1)
        self.position = 0

    @torch.no_grad()
    def observe(self, features):
        x = torch.as_tensor(
            features,
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        ).reshape(1, -1)
        hidden, self.state = self.model.step(
            self.state, self.model.encode(x), write=1 - x[:, 14:15]
        )
        self.position += 1
        return self.model.read(hidden).softmax(-1)[0]

    def save(self, path: Path):
        torch.save(
            {
                "owner": self.owner.validate(),
                "config": asdict(self.model.config),
                "position": self.position,
                "state": self.state,
            },
            path,
        )

    def restore(self, path: Path):
        payload = torch.load(
            path, map_location=next(self.model.parameters()).device, weights_only=True
        )
        if payload["owner"] != self.owner.validate() or payload["config"] != asdict(
            self.model.config
        ):
            raise ValueError("Session belongs to a different owner or model configuration")
        reference = self.model.initial_state(1)
        state = payload["state"]
        if set(state) != set(reference) or payload["position"] < 0:
            raise ValueError("Invalid session state schema")
        for name, tensor in state.items():
            if (
                tensor.shape != reference[name].shape
                or tensor.dtype != reference[name].dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Session tensor contract violated")
        self.state, self.position = state, payload["position"]
