"""Ordered human-language readouts on the existing shared R1 memory."""

import copy
from pathlib import Path

import torch
from torch import nn

from experiments.human_reading.data import sha
from experiments.human_reading.model import ReadingR1, encode


class BookR1(ReadingR1):
    @classmethod
    def attach(cls, owner, vocabulary, bias):
        if type(owner) is not ReadingR1:
            raise ValueError("The retained human-reading owner is required")
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        owner.__class__ = cls
        width = owner.settings["width"]
        owner.book_heads = nn.ModuleDict({k: nn.Linear(width, len(vocabulary)) for k in ("shared", "pooled")})
        owner.register_buffer("book_mean", torch.zeros(2, width))
        owner.register_buffer("book_scale", torch.ones(2, width))
        owner.book_config = {"vocabulary": vocabulary, "source": sha(Path(__file__)), "selected": "unselected"}
        with torch.no_grad():
            for head in owner.book_heads.values():
                head.weight.zero_()
                head.bias.copy_(bias)
        return owner

    def export_config(self):
        return {**super().export_config(), "human_books": copy.deepcopy(self.book_config)}

    @torch.no_grad()
    def book_features(self, contexts):
        ids = encode([" ".join(c[-16:]) for c in contexts])[:, :16]
        encoded = self.stream_projection(self.reading_embedding(ids))
        mask = ids.ne(0)
        state = self.initial(len(contexts))
        value = encoded[:, 0]
        for t in range(ids.shape[1]):
            candidate, updated = self.shared_step(state, encoded[:, t], mask[:, t].float()[:, None])
            state = {k: torch.where(mask[:, t].reshape(-1, *([1]*(v.ndim-1))), updated[k], v) for k, v in state.items()}
            value = torch.where(mask[:, t, None], candidate, value)
        pooled = (encoded*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1)[:, None]
        return torch.stack((value, pooled), 1)

    def book_logits(self, features, kind):
        index = ("shared", "pooled").index(kind)
        x = (features[:, index]-self.book_mean[index])/self.book_scale[index]
        return self.book_heads[kind](x)


def delta(owner):
    return {k: v.detach().clone() for k, v in owner.state_dict().items() if k.startswith("book_")}


def apply(owner, values):
    expected = delta(owner)
    if set(expected) != set(values) or any(v.shape != expected[k].shape or v.dtype != expected[k].dtype
                                         or not torch.isfinite(v).all() for k, v in values.items()):
        raise ValueError("Book parameter schema changed")
    owner.load_state_dict({**owner.state_dict(), **values})
