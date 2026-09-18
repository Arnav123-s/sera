"""Learned source-sentence ranking registered on the actual continuing owner."""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.concept_refinement.runtime import ConceptStudyR1, RefinementSession
from experiments.stream_curriculum.data import token_id

from .data import OUT, read, sha

LEXICAL = 30
WIDTH = LEXICAL + 33
STOP = set("a an the is are was were be been being of in on at to and or by for from with as it its that this these those do does did has have had what which who when where why how".split())


def words(text):
    return re.findall(r"\w+", text.casefold())


def lexical(question, candidates):
    q = [w for w in words(question) if w not in STOP]
    rows = [[w for w in words(c["text"]) if w not in STOP] for c in candidates]
    df = Counter(w for row in rows for w in set(row))
    average = sum(map(len, rows)) / max(1, len(rows))
    scores = []
    for row in rows:
        counts = Counter(row)
        scores.append(sum(math.log(1 + (len(rows) - df[w] + .5) / (df[w] + .5))
                          * counts[w] * 2.2 / (counts[w] + 1.2 * (.25 + .75 * len(row) / max(average, 1)))
                          for w in set(q) if counts[w]))
    maximum = max(scores, default=0.)
    wh = [float(word in words(question)) for word in ("what", "who", "when", "where", "why", "how")]
    q_pairs = set(zip(q, q[1:]))
    q_numbers = {w for w in q if w.isdecimal()}
    result = []
    for i, (sentence, row, score) in enumerate(zip(candidates, rows, scores, strict=True)):
        overlap = len(set(q) & set(row))
        digits = float(any(w.isdecimal() for w in row))
        caps = sum(w[0].isupper() for w in sentence["text"].split()[1:] if w) / max(1, len(row))
        base = [1., score / max(maximum, 1e-8), overlap / max(1, len(set(q))),
                overlap / max(1, len(set(row))), min(len(row), 128) / 128,
                i / max(1, len(rows)-1), float(i == 0),
                len(q_pairs & set(zip(row, row[1:]))) / max(1, len(q_pairs)), digits,
                float(bool(re.search(r"\b\d{4}\b", sentence["text"]))), caps,
                len(q_numbers & set(row)) / max(1, len(q_numbers))]
        result.append(base + wh + [w * digits for w in wh] + [w * caps for w in wh])
    return torch.tensor(result, dtype=torch.float32)


@lru_cache(maxsize=65536)
def hashed(word):
    return token_id(word)


class ReadingR1(ConceptStudyR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not ConceptStudyR1:
            raise ValueError("Restore the retained empirical StudyR1 owner")
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        owner.__class__ = cls
        owner.reading_embedding = nn.Embedding.from_pretrained(
            owner.stream_embedding.weight.detach().clone(), freeze=False, padding_idx=0
        )
        owner.reading_head = nn.Sequential(nn.Linear(WIDTH, 48), nn.Tanh(), nn.Linear(48, 1))
        owner.register_buffer("reading_mean", torch.zeros(WIDTH))
        owner.register_buffer("reading_scale", torch.ones(WIDTH))
        owner.reading_config = {"kind": "shared", "seed": 2901, "source": source()}
        owner.reset_reading("shared", 2901)
        return owner

    def reset_reading(self, kind, seed):
        if kind not in {"lexical", "shared"}:
            raise ValueError("Unknown reading condition")
        torch.manual_seed(seed)
        for layer in self.reading_head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        nn.init.zeros_(self.reading_head[-1].weight)
        nn.init.zeros_(self.reading_head[-1].bias)
        self.reading_config.update(kind=kind, seed=seed)

    def export_config(self):
        return {**super().export_config(), "human_reading": dict(self.reading_config)}

    def reading_logits(self, features, mask):
        x = (features - self.reading_mean) / self.reading_scale
        if self.reading_config["kind"] == "lexical":
            x = torch.cat((x[..., :LEXICAL], torch.zeros_like(x[..., LEXICAL:])), -1)
        value = 4 * features[..., 1] + self.reading_head(x).squeeze(-1)
        return value.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def pooled_reading(self, texts):
        rows = [[hashed(w) for w in words(text)[:128]] or [hashed("empty")] for text in texts]
        width = max(map(len, rows))
        ids = torch.tensor([r + [0] * (width-len(r)) for r in rows])
        mask = ids.ne(0)
        pooled = (self.reading_embedding(ids) * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        return self.stream_projection(pooled)

    def semantic(self, ids):
        mask = ids.ne(0)
        value = (self.reading_embedding(ids) * mask[..., None]).sum(1)
        return F.normalize(value / mask.sum(1).clamp_min(1)[:, None], dim=-1)

    @torch.no_grad()
    def reading_features(self, examples, progress=None):
        x = torch.zeros(len(examples), 16, WIDTH)
        mask = torch.zeros(len(examples), 16, dtype=torch.bool)
        pairs = []
        for i, row in enumerate(examples):
            candidates = row["sentences"]
            if not 1 <= len(candidates) <= 16:
                raise ValueError("One to sixteen source candidates required")
            x[i, :len(candidates), :LEXICAL] = lexical(row["question"], candidates)
            mask[i, :len(candidates)] = True
            pairs.extend((i, j, row["question"], c["text"]) for j, c in enumerate(candidates))
        for start in range(0, len(pairs), 128):
            batch = pairs[start:start+128]
            q = self.pooled_reading([p[2] for p in batch])
            sentence = self.pooled_reading([p[3] for p in batch])
            initial = self.initial(len(batch))
            question_state, state = self.shared_step(initial, q, q.new_ones((len(batch), 1)))
            conditioned, _ = self.shared_step(state, sentence, sentence.new_ones((len(batch), 1)))
            values = (conditioned-question_state)[:, :32]
            x[[p[0] for p in batch], [p[1] for p in batch], LEXICAL:LEXICAL+32] = values
            similarity = (self.semantic(encode([p[2] for p in batch])) *
                          self.semantic(encode([p[3] for p in batch]))).sum(-1)
            x[[p[0] for p in batch], [p[1] for p in batch], -1] = similarity
            if progress and start % 4096 == 0:
                progress(start, len(pairs))
        return x, mask


def source():
    return sha(Path(__file__))


def encode(texts):
    rows = [[hashed(w) for w in words(text)[:96]] or [hashed("empty")] for text in texts]
    return torch.tensor([r + [0] * (96-len(r)) for r in rows])


def load_curriculum(owner):
    from .data import ROOT

    selection = read(OUT / "curriculum-selection.json")
    path = ROOT / selection["checkpoint"]
    if sha(path) != selection["sha256"]:
        raise ValueError("Curriculum checkpoint identity changed")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    with torch.no_grad():
        owner.reading_embedding.weight.copy_(saved["embedding"])
    owner.reading_config["curriculum"] = selection["sha256"]
    owner.reading_embedding.weight.requires_grad_(False)
    return owner


def parent():
    return RefinementSession(read(OUT / "parent-empirical.json"))


def delta(owner):
    return {n: v.detach().clone() for n, v in owner.state_dict().items() if n.startswith("reading_")}


def apply(owner, values):
    expected = delta(owner)
    if (set(values) != set(expected) or any(v.shape != expected[n].shape or v.dtype != expected[n].dtype
                                          or not torch.isfinite(v).all() for n, v in values.items())):
        raise ValueError("Reading weights changed schema or finiteness")
    owner.load_state_dict({**owner.state_dict(), **values})
