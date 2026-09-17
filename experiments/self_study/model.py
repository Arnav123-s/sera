"""Small learned operators registered on the exact continuing request owner."""
import hashlib
from fractions import Fraction as Q
from pathlib import Path

import torch
from torch import nn

from experiments.stream_curriculum.model import StreamR1
from .algebra import SIZE, vector


def source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class FastMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gram", torch.eye(SIZE, dtype=torch.float64) * 1e-6)
        self.register_buffer("rhs", torch.zeros(SIZE, SIZE, dtype=torch.float64))
        self.register_buffer("observations", torch.zeros((), dtype=torch.int64))
        self.weight = nn.Parameter(torch.zeros(SIZE, SIZE, dtype=torch.float64), requires_grad=False)

    @torch.no_grad()
    def update(self, p, q):
        x, y = (torch.tensor([float(v) for v in vector(a)], dtype=torch.float64) for a in (p, q))
        gram, rhs = self.gram + torch.outer(x, x), self.rhs + torch.outer(x, y)
        weight = torch.linalg.solve(gram, rhs)
        if not torch.isfinite(weight).all():
            raise ValueError("Nonfinite candidate operator")
        self.gram.copy_(gram)
        self.rhs.copy_(rhs)
        self.weight.copy_(weight)
        self.observations.add_(1)

    @torch.no_grad()
    def propose(self, p):
        x = torch.tensor([float(v) for v in vector(p)], dtype=torch.float64)
        return [str(Q(float(v)).limit_denominator(120)) for v in x @ self.weight]


class StudyR1(StreamR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not StreamR1:
            raise ValueError("Restore the actual continuing StreamR1 owner")
        owner.__class__ = cls
        owner.study_maps = nn.ModuleDict({d: FastMap() for d in ("sum", "integral")})
        owner.study_source = source()
        return owner

    def export_config(self):
        return {**super().export_config(), "self_study_source": self.study_source}
