"""A finite typed mutation grammar with explicit parent and migration records."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch

from sera.r1 import RecurrentWorldModel
from sera.r2 import ControlledInstrument
from sera.solver import tensor_digest


@dataclass(frozen=True)
class Mutation:
    operation: str
    component: str = "r1"
    value: int = 4

    def validate(self):
        if self.operation not in {"insert_adapter", "selective_routing", "expand_instrument"}:
            raise ValueError("Mutation is outside the declared architecture grammar")
        if type(self.value) is not int or not 1 <= self.value <= 16:
            raise ValueError("Mutation value exceeds the declared resource bound")


def insert_adapter(model, rank=4):
    if model.adapter is not None:
        raise ValueError("This model already has an adapter; use a separate expansion proposal")
    from sera.shared import SharedR1
    if isinstance(model, SharedR1):
        candidate = copy.deepcopy(model)
        candidate.add_adapter(rank)
        return candidate.eval()
    settings = dict(model.settings)
    candidate = RecurrentWorldModel(**settings, planning_horizon=model.planning_horizon, adapter_rank=rank)
    missing, unexpected = candidate.load_state_dict(model.state_dict(), strict=False)
    if unexpected or set(missing) != {"adapter.0.weight", "adapter.2.weight"}:
        raise ValueError("Adapter migration changed an undeclared parameter")
    return candidate.eval()


def mutate(solver, specification, *, seed=0):
    specification.validate()
    if specification.component not in solver.components:
        raise ValueError("Mutation targets an unknown component")
    torch.manual_seed(seed)
    candidate = copy.deepcopy(solver)
    parent = solver.components[specification.component]
    if specification.operation == "insert_adapter":
        if not isinstance(parent, RecurrentWorldModel):
            raise ValueError("Adapter insertion requires an R1 component")
        child = insert_adapter(parent, specification.value)
        preserving = True
    elif specification.operation == "selective_routing":
        if not isinstance(parent, RecurrentWorldModel) or parent.settings["kind"] not in {"reference", "lowrank_hybrid"}:
            raise ValueError("Selective routing requires a reference memory")
        child = candidate.components[specification.component]
        child.memory.routing = "top1"
        child.settings["routing"] = "top1"
        preserving = False
    else:
        if not isinstance(parent, ControlledInstrument) or specification.value not in {4, 8, 16}:
            raise ValueError("Instrument expansion needs a registered bounded event model")
        child = ControlledInstrument(dimension=specification.value, rank=2, event_kind="kraus", event_rank=2,
                                      complex_valued=parent.complex_valued)
        preserving = False
    from sera.shared import SharedR1, replace_shared_owner
    if isinstance(child, SharedR1):
        replace_shared_owner(candidate, child)
    else:
        candidate.components[specification.component] = child
    candidate.validate()
    record = {"grammar_version": 1, "mutation": asdict(specification),
              "parent_solver": solver.identity(), "parent_component": tensor_digest(parent),
              "child_component": tensor_digest(child), "parent_config": parent.export_config(),
              "child_config": child.export_config(), "function_preserving_at_initialization": preserving,
              "migration": "Old immutable solver remains available; non-preserving candidates require new training and admission"}
    return candidate.eval(), record
