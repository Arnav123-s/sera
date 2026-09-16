"""A protected language interface on the actual task-bearing SERA R1 owner."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from experiments.task_transfer.runtime import TaskR1
from sera.storage import digest

from .data import WORDS, encode


def source():
    folder = Path(__file__).parent
    return digest({n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ("model.py", "data.py")})


class InquiryR1(TaskR1):
    @classmethod
    def attach(cls, owner, seed, kind="shared"):
        if type(owner) is not TaskR1 or kind not in ("shared", "bag"):
            raise ValueError("The exact preserved task owner and a declared language path are required")
        torch.manual_seed(seed)
        owner.__class__ = cls
        width = owner.settings["width"]
        owner.inquiry_embedding = nn.Embedding(len(WORDS), width, padding_idx=0)
        owner.inquiry_norm = nn.LayerNorm(width)
        owner.inquiry_head = nn.Linear(width, 5)
        nn.init.normal_(owner.inquiry_embedding.weight, std=.1)
        owner.inquiry_kind, owner.inquiry_seed = kind, seed
        owner.inquiry_source = source()
        owner.inquiry_admitted = []
        return owner

    def export_config(self):
        return {**super().export_config(), "inquiry_source": self.inquiry_source,
                "inquiry_kind": self.inquiry_kind, "inquiry_seed": self.inquiry_seed,
                "inquiry_admitted": list(self.inquiry_admitted)}

    def span_logits(self, ids):
        x = self.inquiry_embedding(ids)
        if self.inquiry_kind == "bag":
            output = x.sum(1)/(ids != 0).sum(1)[:, None]
        else:
            state = self.initial(len(ids))
            output = torch.zeros(len(ids), self.settings["width"])
            for t in range(int((ids != 0).sum(1).max())):
                valid = ids[:, t] != 0
                value, updated = self.shared_step(state, x[:, t], valid.float()[:, None])
                state = {k: torch.where(valid.reshape(-1, *([1]*(v.ndim-1))), updated[k], v)
                         for k, v in state.items()}
                output = torch.where(valid[:, None], value, output)
        return self.inquiry_head(self.inquiry_norm(output))

    @torch.no_grad()
    def spans(self, texts):
        self.eval()
        return self.span_logits(torch.tensor(encode(texts))).softmax(-1).tolist()


def delta(owner):
    return {n: v.detach().clone() for n, v in owner.state_dict().items() if n.startswith("inquiry_")}


def apply(owner, values):
    expected = delta(owner)
    if set(expected) != set(values) or any(expected[n].shape != v.shape or expected[n].dtype != v.dtype
                                         or not torch.isfinite(v).all() for n, v in values.items()):
        raise ValueError("Inquiry interface tensor contract differs")
    full = owner.state_dict()
    full.update(values)
    owner.load_state_dict(full)


def parser_identity(owner):
    h = hashlib.sha256(digest({"source": owner.inquiry_source, "kind": owner.inquiry_kind,
                             "admitted": owner.inquiry_admitted}).encode())
    for n, v in sorted(delta(owner).items()):
        h.update(n.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()
