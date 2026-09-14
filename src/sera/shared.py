"""One R1 parameter/state core behind world, typed and sequence interfaces."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from sera.contracts import EvidenceKind
from sera.r1 import RecurrentWorldModel
from sera.typed_learning import TASKS, TypedEncoder


class SharedTypedEncoder(TypedEncoder):
    def __init__(self, width, encoding):
        super().__init__(width)
        if encoding not in {"legacy-v1", "symbolic-v2"}:
            raise ValueError("Unknown shared event normalization")
        self.encoding = encoding

    def features(self, observation):
        values = TypedEncoder.features(observation)
        if self.encoding == "symbolic-v2" and observation.modality == "symbolic":
            # Categorical one-hot channels and rule flags use unit magnitude.
            # Position has a fixed scale of eight events, with no future-length access.
            values[:16] = [8 * value for value in values[:16]]
            values[32] *= 8
        return values


class SharedR1(RecurrentWorldModel):
    def __init__(self, width=256, heads=8, memory_dim=32, kind="delta", programs=None,
                 encoding="symbolic-v2", structured_addresses=True, **options):
        super().__init__(width, heads, memory_dim, kind, **options)
        self.encoding = encoding
        self.structured_addresses = structured_addresses
        self.address_encoder = nn.Linear(4, width, bias=False) if structured_addresses else None
        self.typed_encoder = SharedTypedEncoder(width, encoding)
        self.typed_instruction = nn.Embedding(len(TASKS), width)
        if encoding == "symbolic-v2":
            nn.init.normal_(self.typed_instruction.weight, std=.05)
        self.typed_categorical = nn.Linear(width, 4)
        self.typed_numeric = nn.Linear(width, 2)
        self.sequence_encoder = nn.Sequential(nn.Linear(20, width), nn.Tanh())
        self.sequence_decoder = nn.Linear(width, 4)
        self.programs = copy.deepcopy(programs or {})
        self.validity()

    def export_config(self):
        return {**super().export_config(), "type": "shared_r1", "programs": copy.deepcopy(self.programs),
                "encoding": self.encoding, "structured_addresses": self.structured_addresses}

    def validity(self):
        from sera.typed_programs import validate_program
        for task, program in self.programs.items():
            validate_program(program)
            if task != program["task"] or task not in TASKS:
                raise ValueError("Shared typed procedure task mismatch")
        return {"program_integrity_error": 0.0}

    def shared_step(self, state, encoded, write, *, address=None, address_mask=None):
        hidden, state = self.memory.step(state, encoded, write=write, address=address, address_mask=address_mask)
        fused = self.fusion(torch.cat((encoded, hidden), -1))
        if self.adapter is not None:
            fused = fused + self.adapter(fused)
        return fused, state

    def forward_typed(self, observations, tasks):
        if not observations or len(observations) != len(tasks) or any(not row for row in observations):
            raise ValueError("Typed observations and task instructions must align")
        state = self.initial(len(tasks))
        hidden = next(self.parameters()).new_zeros(len(tasks), self.settings["width"])
        for position in range(max(map(len, observations))):
            rows = [i for i, events in enumerate(observations) if position < len(events)]
            indices = torch.tensor(rows)
            events = [observations[i][position] for i in rows]
            encoded = self.typed_encoder(events) + self.typed_instruction(torch.tensor([TASKS.index(tasks[i]) for i in rows]))
            write = encoded.new_tensor([[float(not (tasks[i] == "binding" and position == len(observations[i])-1))] for i in rows])
            address, mask = None, None
            if self.address_encoder is not None:
                eligible = [tasks[i] == "binding" and obs.modality == "symbolic" and len(obs.values) >= 8
                            and obs.scale == 1 and all((obs.available or (True,)*len(obs.values))[:4])
                            and sum(obs.values[:4]) == 1 and all(v in (0, 1) for v in obs.values[:4])
                            for i, obs in zip(rows, events)]
                keys = encoded.new_tensor([list(obs.values[:4]) if valid else [0.0]*4 for valid, obs in zip(eligible, events)])
                address = self.address_encoder(keys)
                mask = torch.tensor(eligible)[:, None]
            output, updated = self.shared_step({key: value[indices] for key, value in state.items()}, encoded, write,
                                                address=address, address_mask=mask)
            state = {key: value.index_copy(0, indices, updated[key]) for key, value in state.items()}
            hidden = hidden.index_copy(0, indices, output)
        return {"categorical": self.typed_categorical(hidden), "numeric": self.typed_numeric(hidden)}

    def forward_sequence(self, inputs):
        state = self.initial(len(inputs))
        for position in range(inputs.shape[1]):
            encoded = self.sequence_encoder(inputs[:, position])
            write = 1 - inputs[:, position, 14:15]
            address, mask = None, None
            if self.address_encoder is not None:
                task_ids = inputs[:, position, 8:13].argmax(-1)
                mask = ((task_ids == 1) | (task_ids == 4))[:, None]
                address = self.address_encoder(inputs[:, position, :4])
            hidden, state = self.shared_step(state, encoded, write, address=address, address_mask=mask)
        return self.sequence_decoder(hidden)

    def core_state_bytes(self, batch=1):
        return sum(value.numel() * value.element_size() for value in self.initial(batch).values())

    def add_adapter(self, rank=8):
        if self.adapter is not None or type(rank) is not int or not 1 <= rank <= self.settings["width"]:
            raise ValueError("Adapter insertion needs a fresh valid rank")
        width = self.settings["width"]
        self.adapter = nn.Sequential(nn.Linear(width, rank, bias=False), nn.Tanh(), nn.Linear(rank, width, bias=False))
        nn.init.zeros_(self.adapter[-1].weight)
        self.settings["adapter_rank"] = rank


@dataclass(frozen=True)
class SharedReadoutConfig:
    kind: str = "shared_sequence_view"
    owner: str = "r1"


class SharedView(nn.Module):
    """A parameter-free interface; the solver registers its owner exactly once."""

    def __init__(self, owner, name="r1"):
        super().__init__()
        if not isinstance(owner, SharedR1) or name != "r1":
            raise ValueError("Shared interfaces require their registered R1 owner")
        object.__setattr__(self, "owner", owner)
        self.owner_name = name

    def train(self, mode=True):
        self.owner.train(mode)
        return super().train(mode)

    def validity(self):
        return self.owner.validity()


class SharedTypedView(SharedView):
    @property
    def programs(self):
        return self.owner.programs

    def export_config(self):
        return {"type": "shared_typed_view", "owner": self.owner_name}

    def forward(self, observations, tasks):
        return self.owner.forward_typed(observations, tasks)

    @torch.no_grad()
    def predict(self, observations, task):
        self.eval()
        if task in self.programs:
            from sera.typed_programs import execute_rule
            try:
                answer = execute_rule(self.programs[task]["rule"], observations)
                return {"kind": EvidenceKind.PREDICTION.value, "task": task, "class": answer,
                        "procedure": self.programs[task]["identity"], "route": "verified-program"}
            except (ValueError, UnicodeError):
                pass
        output = self([observations], [task])
        if task == "motion":
            return {"kind": EvidenceKind.PREDICTION.value, "task": task, "values": output["numeric"][0].tolist(), "units": "m"}
        probabilities = output["categorical"][0].softmax(-1)
        return {"kind": EvidenceKind.PREDICTION.value, "task": task, "class": int(probabilities.argmax()),
                "probabilities": probabilities.tolist(), "route": "shared-r1"}


class SharedSequenceView(SharedView):
    @property
    def config(self):
        return SharedReadoutConfig(owner=self.owner_name)

    def export_config(self):
        return {"type": "shared_sequence_view", "owner": self.owner_name}

    def forward(self, inputs):
        return self.owner.forward_sequence(inputs)

    def state_bytes(self, batch=1):
        return self.owner.core_state_bytes(batch)


class IndependentTypedControl(nn.Module):
    """Explicit extra model for the separate-parameter control condition."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    @property
    def programs(self):
        return self.core.programs

    def export_config(self):
        return {"type": "independent_typed_control", "core": self.core.export_config()}

    def forward(self, observations, tasks):
        return self.core.forward_typed(observations, tasks)

    def predict(self, observations, task):
        return SharedTypedView(self.core).predict(observations, task)

    def validity(self):
        return self.core.validity()


def make_shared_solver(model, **options):
    from sera.solver import Solver
    return Solver(SharedSequenceView(model), components={"r1": model, "typed": SharedTypedView(model)}, **options)


def replace_shared_owner(solver, model):
    if not isinstance(model, SharedR1):
        raise ValueError("A shared solver requires a shared replacement")
    solver.components["r1"] = model
    solver.neural = SharedSequenceView(model)
    solver.components["typed"] = SharedTypedView(model)
