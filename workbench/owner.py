"""Explicit extension of the same R1 owner for observed numerical task contexts."""

import copy
import hashlib
from pathlib import Path

import torch

from experiments.continuing_control.core import InteractiveR1
from sera.session_state import model_identity

PRE_FORECAST_SOURCE = "d0131cb4e99a4026a4785579c8ae5b5a1643d69ea7c90245d9c10c3675dae4d4"
PRE_TASK_SOURCE = "28af3fdc9491d43b23a0c52238e1694cbaef1b2471f6f45990a9ac7af27a3db8"
MIGRATABLE_SOURCES = {PRE_FORECAST_SOURCE, PRE_TASK_SOURCE}


def source_identity():
    folder = Path(__file__).parent
    return hashlib.sha256(b"".join((folder/name).read_bytes() for name in ("owner.py", "streams.py", "task_learning.py"))).hexdigest()


class LiveR1(InteractiveR1):
    def export_config(self):
        return {**super().export_config(), "live_schema": 1, "live_source": self.live_source,
                "live_contexts": sorted(self.live_contexts)}

    @classmethod
    def attach(cls, owner, contexts=None):
        if type(owner) is not InteractiveR1:
            raise ValueError("Expected the unchanged continuing predecessor")
        owner.__class__ = cls
        owner.live_source = source_identity()
        owner.live_contexts = {}
        for name, values in (contexts or {}).items():
            owner.set_context(name, values)
        return owner

    def set_context(self, name, values):
        if len(name) != 16 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("Invalid task context identity")
        value = torch.tensor(values, dtype=torch.float64)
        if value.shape != (5,) or not bool(torch.isfinite(value).all()):
            raise ValueError("Expected three learned coefficients, count and last observation")
        key = "live_stream_"+name
        if name in self.live_contexts:
            getattr(self, key).copy_(value)
        else:
            self.register_buffer(key, value)
            self.live_contexts[name] = key

    def contexts(self):
        return {name: getattr(self, key).tolist() for name, key in self.live_contexts.items()}

    def transform_context(self, name, inputs):
        if name not in self.live_contexts:
            raise ValueError("Task has not been learned")
        a, b, c = getattr(self, self.live_contexts[name])[:3]
        x = torch.tensor(inputs, dtype=torch.float64)
        return (a*x*x+b*x+c).tolist()

    def forecast_context(self, name, history, horizon):
        """Execute acquired coefficients from this actual owner, not an external cache."""
        if name not in self.live_contexts or len(history) < 3 or not 1 <= horizon <= 48:
            raise ValueError("A learned task context and bounded forecast horizon are required")
        coefficients = getattr(self, self.live_contexts[name])[:3]
        values = torch.as_tensor(history, dtype=torch.float64)
        scale = max(float(values[-32:].std(correction=0)), 1e-6)
        anchor = float(values[-1])
        previous, last = float(values[-2]), anchor
        result = []
        for step in range(horizon):
            value = float(coefficients@torch.tensor([last, previous, 1.], dtype=torch.float64))
            value = min(anchor+4*scale, max(anchor-4*scale, value))
            result.append({"step": step+1, "value": value})
            previous, last = last, value
        return result


def base_snapshot(session):
    """Preserve the frozen A08 restore contract as one explicit part of the new owner."""
    result = session.snapshot()
    base = copy.deepcopy(session.owner)
    for key in base.live_contexts.values():
        delattr(base, key)
    base.__class__ = InteractiveR1
    result["owner_sha256"] = model_identity(base)
    return result
