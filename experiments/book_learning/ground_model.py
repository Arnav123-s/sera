"""Learned definition bindings on the retained computational owner."""

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.human_reading.data import ROOT, digest, read, sha
from experiments.human_reading.model import encode, words
from experiments.human_reading.runtime import ReadingSession

from .data import OUT
from .ground_data import ROLES
from .model import BookR1
from .model import apply as apply_books

KINDS = ("neural", "lexical", "combined")


def restore_books():
    session = ReadingSession()
    facts, library = session.base.learner.legacy.snapshot(), session.base.learner.library.record()
    selected = read(OUT / "selection.json")
    path = ROOT / selected["checkpoint"]
    if sha(path) != selected["sha256"]:
        raise ValueError("Human book checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    BookR1.attach(session.owner, saved["vocabulary"], torch.zeros(len(saved["vocabulary"])))
    apply_books(session.owner, saved["delta"])
    session.owner.book_config["selected"] = selected["selected"]
    session.owner.book_config["admission"] = "experimental; unigram had lower final loss"
    session.base.learner.rebind(facts, library)
    session.base.assert_owner()
    return session


def lexical(texts):
    values = torch.zeros(len(texts), 1024)
    for i, text in enumerate(texts):
        token = words(text)[:128]
        features = token + [a+"|"+b for a, b in zip(token, token[1:])]
        for word in features:
            values[i, int(digest("GD-001:"+word)[:16], 16) % 1024] += 1
    return F.normalize(values, dim=-1)


class GroundR1(BookR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not BookR1:
            raise ValueError("Definition binding must extend the retained BookR1")
        for p in owner.parameters():
            p.requires_grad_(False)
        owner.__class__ = cls
        width = owner.settings["width"] + owner.reading_embedding.embedding_dim
        owner.ground_heads = nn.ModuleDict({k: nn.Linear(n, len(ROLES)) for k, n in
                                           (("neural", width), ("lexical", 1024), ("combined", width+1024))})
        for head in owner.ground_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        owner.register_buffer("ground_mean", torch.zeros(width))
        owner.register_buffer("ground_scale", torch.ones(width))
        owner.ground_config = {"roles": list(ROLES), "source": sha(Path(__file__)), "selected": "unselected"}
        return owner

    def export_config(self):
        return {**super().export_config(), "grounded_definitions": copy.deepcopy(self.ground_config)}

    @torch.no_grad()
    def ground_features(self, texts):
        ids = encode(texts)
        mask = ids.ne(0)
        embedded = self.reading_embedding(ids)
        encoded = self.stream_projection(embedded)
        state = self.initial(len(texts))
        value = encoded[:, 0]
        for t in range(ids.shape[1]):
            candidate, updated = self.shared_step(state, encoded[:, t], mask[:, t].float()[:, None])
            state = {k: torch.where(mask[:, t].reshape(-1, *([1]*(v.ndim-1))), updated[k], v) for k, v in state.items()}
            value = torch.where(mask[:, t, None], candidate, value)
        mean = (embedded*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1)[:, None]
        return {"neural": torch.cat((value, mean), -1), "lexical": lexical(texts)}

    def ground_logits(self, features, kind):
        neural = (features["neural"]-self.ground_mean)/self.ground_scale
        x = neural if kind == "neural" else features["lexical"] if kind == "lexical" else torch.cat((neural, features["lexical"]), -1)
        return self.ground_heads[kind](x)


def delta(owner):
    return {k: v.detach().clone() for k, v in owner.state_dict().items() if k.startswith("ground_")}


def apply(owner, values):
    expected = delta(owner)
    if set(expected) != set(values) or any(v.shape != expected[k].shape or v.dtype != expected[k].dtype
                                         or not torch.isfinite(v).all() for k, v in values.items()):
        raise ValueError("Definition binding parameter schema changed")
    owner.load_state_dict({**owner.state_dict(), **values})
