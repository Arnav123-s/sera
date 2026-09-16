"""Learned lexical/semantic interfaces use the actual R1 recurrent and fusion path."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from workbench.owner import LiveR1

from .data import MAX_LENGTH, TOKENS, VOCAB, tokens


def fingerprint():
    folder = Path(__file__).parent
    return hashlib.sha256(b"".join((folder/n).read_bytes() for n in ("model.py", "data.py"))).hexdigest()


class LanguageR1(LiveR1):
    @classmethod
    def attach(cls, owner, seed):
        if type(owner) is not LiveR1:
            raise ValueError("Language extension needs the actual live predecessor")
        torch.manual_seed(seed)
        owner.__class__ = cls
        width = owner.settings["width"]
        owner.language_embedding = nn.Embedding(len(TOKENS), width, padding_idx=0)
        owner.language_position = nn.Embedding(MAX_LENGTH, width)
        owner.language_norm = nn.LayerNorm(width)
        owner.language_queries = nn.Parameter(torch.randn(4, width)*.1)
        owner.language_readout = nn.ModuleList([nn.Linear(width, count) for count in (2, 11, 11, 11)])
        nn.init.normal_(owner.language_embedding.weight, std=.1)
        nn.init.normal_(owner.language_position.weight, std=.03)
        owner.language_source = fingerprint()
        owner.language_words = sorted(set(TOKENS)-{"scale", "<unknown>", "<pad>"})
        return owner

    def export_config(self):
        return {**super().export_config(), "language_source": self.language_source,
                "language_vocabulary": TOKENS, "language_words_admitted": self.language_words,
                "language_interface": "shared_recurrent_attention_slots_v1"}

    def language(self, ids, *, reset_memory=False):
        ids = ids[:, :int((ids != 0).sum(1).max())]
        state = self.initial(len(ids))
        encoded = self.language_embedding(ids)+self.language_position(torch.arange(ids.shape[1]))[None]
        hidden = []
        for position in range(ids.shape[1]):
            if reset_memory:
                state = self.initial(len(ids))
            valid = ids[:, position] != 0
            output, next_state = self.shared_step(state, encoded[:, position], valid.float()[:, None])
            state = {key: torch.where(valid.reshape(-1, *([1]*(value.ndim-1))), next_state[key], value) for key, value in state.items()}
            hidden.append(output)
        values = self.language_norm(torch.stack(hidden, 1))
        scores = torch.einsum("btw,sw->bst", values, self.language_queries)/values.shape[-1]**.5
        scores = scores.masked_fill(ids[:, None] == 0, -1e9)
        attention = scores.softmax(-1)
        pooled = torch.einsum("bst,btw->bsw", attention, values)
        return [head(pooled[:, i]) for i, head in enumerate(self.language_readout)]

    @torch.no_grad()
    def interpret(self, texts, *, reset_memory=False):
        self.eval()
        inputs = [tokens(text) for text in texts]
        logits = self.language(torch.tensor([x[0] for x in inputs]), reset_memory=reset_memory)
        probabilities = [value.softmax(-1) for value in logits]
        labels = torch.stack([value.argmax(-1) for value in probabilities], 1).tolist()
        confidence = torch.stack([value.max(-1).values for value in probabilities], 1).min(1).values.tolist()
        return [{"labels": row, "confidence": score, "missing_words": sorted(set(words)-set(self.language_words))}
                for row, score, (_, words) in zip(labels, confidence, inputs)]


def tensors(rows):
    return torch.tensor([tokens(r["text"])[0] for r in rows]), torch.tensor([r["labels"] for r in rows])


def trainable(owner, scope, *, novel_word=None):
    if scope not in ("interface", "shared", "word"):
        raise ValueError("Unsupported adaptation scope")
    for name, parameter in owner.named_parameters():
        enabled = name.startswith("language_") if scope == "interface" else name.startswith(("language_", "memory.", "fusion."))
        if scope == "word":
            enabled = name == "language_embedding.weight"
        parameter.requires_grad_(enabled)
    if scope == "word":
        if novel_word not in VOCAB:
            raise ValueError("Word needs an explicit vocabulary entry")
        row = VOCAB[novel_word]
        def mask(gradient):
            result = torch.zeros_like(gradient)
            result[row] = gradient[row]
            return result
        return owner.language_embedding.weight.register_hook(mask)
    return None
