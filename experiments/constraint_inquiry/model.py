"""New protected interfaces on the existing shared recurrent and physical owner."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from experiments.language_inquiry.model import InquiryR1
from sera.storage import digest

from .data import ENTITIES, VOCAB, WORDS, encode


def source():
    root = Path(__file__).parent
    return digest({n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                   for n in ("data.py", "model.py", "settling.py")})


class ConstraintR1(InquiryR1):
    @classmethod
    def attach(cls, owner, seed, kind="ordered"):
        if type(owner) is not InquiryR1 or kind not in ("ordered", "bag"):
            raise ValueError("The exact continuing inquiry owner is required")
        torch.manual_seed(seed)
        owner.__class__ = cls
        width = owner.settings["width"]
        owner.cloud_embedding = nn.Embedding(len(WORDS), 32, padding_idx=0)
        owner.cloud_projection = nn.Linear(32, width)
        owner.cloud_norm = nn.LayerNorm(width)
        owner.cloud_roles = nn.Linear(width, 2)
        owner.cloud_mode = nn.Linear(width, 4)
        owner.cloud_proposal = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 2), nn.Tanh())
        owner.register_buffer("cloud_location_sum", torch.zeros(4, 2, dtype=torch.float64))
        owner.register_buffer("cloud_location_count", torch.zeros(4, dtype=torch.int64))
        owner.cloud_kind, owner.cloud_seed, owner.cloud_source = kind, seed, source()
        return owner

    def export_config(self):
        return {**super().export_config(), "cloud_kind": self.cloud_kind,
                "cloud_seed": self.cloud_seed, "cloud_source": self.cloud_source}

    def binding(self, ids):
        if self.cloud_kind == "bag":
            ids = ids.sort(dim=1, descending=True).values
        x = self.cloud_projection(self.cloud_embedding(ids))
        state = self.initial(len(ids))
        hidden, outputs = x.new_zeros(len(ids), self.settings["width"]), []
        for t in range(ids.shape[1]):
            valid = ids[:, t] != 0
            value, updated = self.shared_step(state, x[:, t], valid.float()[:, None])
            state = {k: torch.where(valid.reshape(-1, *([1] * (v.ndim - 1))), updated[k], v)
                     for k, v in state.items()}
            hidden = torch.where(valid[:, None], value, hidden)
            outputs.append(self.cloud_norm(value))
        token_scores = self.cloud_roles(torch.stack(outputs, 1))
        roles = torch.stack([
            token_scores.masked_fill((ids != VOCAB[e])[:, :, None], -10000).max(1).values
            for e in ENTITIES
        ], 1).transpose(1, 2)
        return roles[:, 0], roles[:, 1], self.cloud_mode(self.cloud_norm(hidden))

    @torch.no_grad()
    def interpret(self, text):
        outputs = self.binding(torch.tensor(encode([text])))
        return {"actor": ENTITIES[int(outputs[0].argmax(-1))],
                "target": ENTITIES[int(outputs[1].argmax(-1))],
                "mode": int(outputs[2].argmax(-1)),
                "scores": [v.softmax(-1)[0].tolist() for v in outputs],
                "scope": "Finite supervised binding; scores are not applicability probabilities"}


def delta(owner):
    return {n: v.detach().clone() for n, v in owner.state_dict().items() if n.startswith("cloud_")}


def apply(owner, values):
    expected = delta(owner)
    if set(values) != set(expected) or any(v.shape != expected[n].shape or v.dtype != expected[n].dtype
                                         or not torch.isfinite(v).all() for n, v in values.items()):
        raise ValueError("Cloud tensor schema changed")
    owner.load_state_dict({**owner.state_dict(), **values})


def language_identity(owner):
    from experiments.language_inquiry.graph import language_identity as inherited
    h = hashlib.sha256((source() + inherited(owner)).encode())
    for n, v in sorted(delta(owner).items()):
        if not n.startswith(("cloud_location_", "cloud_proposal.")):
            h.update(n.encode())
            h.update(v.numpy().tobytes())
    return h.hexdigest()
