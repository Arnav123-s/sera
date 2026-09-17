"""Trainable real-request interface on the exact continuing constraint owner."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from experiments.constraint_inquiry.model import ConstraintR1
from sera.storage import digest

from .data import BUCKETS, MAX_TOKENS, token_id


def source():
    folder = Path(__file__).parent
    return digest({n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in ("data.py", "model.py")})


class StreamR1(ConstraintR1):
    @classmethod
    def attach(cls, owner, vocabulary, seed=2631, kind="shared"):
        if type(owner) is not ConstraintR1 or kind not in ("shared", "pooled"):
            raise ValueError("The exact constraint owner and declared route are required")
        torch.manual_seed(seed)
        width = owner.settings["width"]
        owner.__class__ = cls
        owner.stream_embedding = nn.Embedding(BUCKETS, 48, padding_idx=0)
        nn.init.normal_(owner.stream_embedding.weight, std=.1)
        with torch.no_grad():
            owner.stream_embedding.weight[0].zero_()
        owner.stream_projection = nn.Linear(48, width)
        owner.stream_norm = nn.LayerNorm(width)
        owner.stream_intent = nn.Linear(width, len(vocabulary["intents"]))
        owner.stream_slots = nn.Linear(width, len(vocabulary["tags"]))
        owner.stream_config = {"vocabulary": vocabulary, "seed": seed, "kind": kind, "source": source()}
        return owner

    def export_config(self):
        return {**super().export_config(), "stream": self.stream_config}

    def request_logits(self, ids):
        encoded = self.stream_projection(self.stream_embedding(ids))
        mask = ids != 0
        if self.stream_config["kind"] == "shared":
            state, outputs = self.initial(len(ids)), []
            for t in range(ids.shape[1]):
                valid = mask[:, t]
                value, updated = self.shared_step(state, encoded[:, t], valid.float()[:, None])
                state = {k: torch.where(valid.reshape(-1, *([1] * (v.ndim - 1))), updated[k], v)
                         for k, v in state.items()}
                outputs.append(value)
            hidden = torch.stack(outputs, 1)
        else:
            hidden = encoded
        hidden = self.stream_norm(hidden)
        pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        return self.stream_intent(pooled), self.stream_slots(hidden)


def encode(texts):
    rows = [text.split() for text in texts]
    if any(not 0 < len(row) <= MAX_TOKENS for row in rows):
        raise ValueError("Request outside the declared token-length scope")
    length = max(map(len, rows))
    return torch.tensor([[token_id(word) for word in row] + [0] * (length - len(row)) for row in rows])


def batch(rows, vocabulary):
    x = encode([r["text"] for r in rows])
    intent = torch.tensor([vocabulary["intents"].index(r["intent"]) for r in rows])
    slots = torch.full(x.shape, -100, dtype=torch.long)
    for i, row in enumerate(rows):
        # An unseen development label remains an error and is reported, never taught here.
        values = [vocabulary["tags"].index(tag) if tag in vocabulary["tags"] else -100 for tag in row["tags"]]
        slots[i, :len(values)] = torch.tensor(values)
    return x, intent, slots


def delta(owner):
    return {n: v.detach().clone() for n, v in owner.state_dict().items() if n.startswith("stream_")}


def apply(owner, values):
    expected = delta(owner)
    if set(expected) != set(values) or any(v.shape != expected[n].shape or v.dtype != expected[n].dtype
                                         or not torch.isfinite(v).all() for n, v in values.items()):
        raise ValueError("Changed request-interface tensor schema")
    owner.load_state_dict({**owner.state_dict(), **values})
