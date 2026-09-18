"""Knowledge and procedure weights on the exact acquired owner."""

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.autonomous_discovery.acquire import AcquiredR1

from .common import BOOKS, PREFIXES, RATES, SKILLS, contracts


class GrowingR1(AcquiredR1):
    @classmethod
    def attach(cls, owner, seed):
        if type(owner) is not AcquiredR1:
            raise ValueError("Continue the actual acquired owner")
        for name, parameter in owner.named_parameters():
            parameter.requires_grad_(name.startswith(PREFIXES))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            policy = nn.Sequential(nn.Linear(len(SKILLS)+8, 32, dtype=torch.float64), nn.Tanh(),
                                   nn.Linear(32, 1, dtype=torch.float64))
            nn.init.zeros_(policy[-1].weight)
            nn.init.zeros_(policy[-1].bias)
        owner.__class__ = cls
        owner.growth_policy = policy
        owner.growth_config = {"seed": seed, "contracts": contracts()}
        return owner

    def export_config(self):
        return {**super().export_config(), "continuing_growth": copy.deepcopy(self.growth_config)}


def knowledge(owner):
    return {k: v.detach().clone() for k, v in owner.state_dict().items() if k.startswith(PREFIXES)}


def identity(values):
    h = hashlib.sha256()
    for k, v in sorted(values.items()):
        h.update(k.encode())
        h.update(str((v.shape, v.dtype)).encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def apply(owner, values):
    current = knowledge(owner)
    if set(current) != set(values) or any(current[k].shape != v.shape or current[k].dtype != v.dtype
                                         or not torch.isfinite(v).all() for k, v in values.items()):
        raise ValueError("Changed continuing knowledge schema")
    owner.load_state_dict(values, strict=False)


def logits(owner, skill, rows):
    if skill in BOOKS:
        return owner.book_logits(rows["x"], "shared")
    if skill == "reading":
        return owner.reading_logits(rows["x"], rows["mask"])
    if skill == "methods":
        return owner.quest_policy(rows["x"])
    return owner.stream_intent(rows["x"])


def errors(owner, skill, rows):
    value = logits(owner, skill, rows)
    if skill == "reading":
        return -value.log_softmax(-1).masked_fill(~rows["gold"], -torch.inf).logsumexp(-1)
    return F.cross_entropy(value, rows["y"], reduction="none")


@torch.no_grad()
def measure(owner, data, witness=False):
    records = {}
    for skill in SKILLS:
        row = data[skill]
        value = logits(owner, skill, row)
        prediction = value.argmax(-1)
        good = row["gold"].gather(1, prediction[:, None]).squeeze(1) if skill == "reading" else prediction.eq(row["y"])
        loss = float(errors(owner, skill, row).mean())
        record = {"loss": loss, "score": 1/(1+loss), "correct": int(good.sum()), "count": len(good),
                  "accuracy": float(good.double().mean())}
        if witness:
            record.update(logits=value.numpy(), labels=row["y"].numpy(),
                          gold=row.get("gold", torch.empty(0)).numpy())
        records[skill] = record
    return {"skills": records, "scores": [records[s]["score"] for s in SKILLS],
            "accuracy": [records[s]["accuracy"] for s in SKILLS],
            "macro": float(np.mean([records[s]["score"] for s in SKILLS]))}


def candidates(before, recent, visits):
    rows = []
    for i in range(len(SKILLS)):
        for rate in range(len(RATES)):
            for method in range(2):
                rows.append([float(j == i) for j in range(len(SKILLS))] +
                    [before["scores"][i], before["accuracy"][i], recent[i], visits[i]/128,
                     rate/2, float(method), 1/(1+visits[i]), 1.])
    return torch.tensor(rows, dtype=torch.float64)


def credit(before, after, highwater, chosen):
    old, new, peak = (np.asarray(x, dtype=float) for x in (before["scores"], after["scores"], highwater))
    if any(x.shape != (len(SKILLS),) for x in (old, new, peak)) or not np.isfinite([old, new, peak]).all():
        raise ValueError("Finite independent scores for every strand are required")
    novelty = np.maximum(new-peak, 0)
    transfer = novelty.copy()
    transfer[chosen] = 0
    maintenance = float((new-old).mean())
    bonus, connections = float(.5*novelty.mean()), float(.25*transfer.mean())
    accurate = float(.25*(np.mean(after["accuracy"])-np.mean(before["accuracy"])))
    return {"reward": maintenance+bonus+connections+accurate, "maintenance": maintenance,
            "new_highwater": bonus, "cross_strand_highwater": connections, "accuracy": accurate,
            "highwater": np.maximum(peak, new).tolist()}
