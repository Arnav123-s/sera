"""Explicitly instructed binding rules with disjoint cases and forced overwrites."""

from __future__ import annotations

import numpy as np

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.data import seed_for
from sera.typed_learning import TypedExample
from sera.typed_protocol import semantic_id


def binding_cases(*, seed, count=128, rule="earliest", split="support", family="ordinary", unique=True):
    if rule not in {"earliest", "latest"} or family not in {"ordinary", "long", "composition"}:
        raise ValueError("Unknown binding rule or composition family")
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("Binding case count exceeds the declared budget")
    if split.startswith("validation"):
        bucket = "validation"
    elif split.startswith(("test", "query", "promotion")):
        bucket = "test"
    elif split.startswith(("support", "meta-train")):
        bucket = "support"
    else:
        raise ValueError("Binding split must declare its evidence role")
    rng = np.random.default_rng(seed_for(f"binding-rule-v1/{rule}/{split}/{family}", seed))
    rows, seen = [], set()
    for attempt in range(count * 100):
        length = 12 if family == "long" else 8 if family == "composition" else 6
        query = int(rng.integers(4))
        keys = rng.choice([k for k in range(4) if k != query], size=length)
        positions = sorted(rng.choice(length, size=3 if family == "composition" else 2, replace=False).tolist())
        values = rng.integers(4, size=length)
        for position in positions:
            keys[position] = query
        # First and last interpretations must disagree on every scored example.
        values[positions[-1]] = (values[positions[0]] + int(rng.integers(1, 4))) % 4
        identity = f"{seed}/{split}/{family}/{rule}/{attempt}"
        raw = Provenance("binding-rule-v1", identity, EvidenceKind.OBSERVATION)
        instruction = [float(rule == "earliest"), float(rule == "latest")]
        events = []
        for position, (key, value) in enumerate(zip(keys, values)):
            features = [float(i == key) for i in range(4)] + [float(i == value) for i in range(4)] + instruction
            events.append(Observation("symbolic", tuple(features), position, raw))
        events.append(Observation("symbolic", tuple([float(i == query) for i in range(4)] + [0.0]*4 + instruction),
                                  length, raw, available=(True,)*4 + (False,)*4 + (True,)*2))
        target = int(values[positions[0] if rule == "earliest" else positions[-1]])
        row = TypedExample(tuple(events), "binding", target,
                           Provenance("binding-rule-v1", identity, EvidenceKind.SYNTHETIC), split)
        semantic = semantic_id(row)
        value = int(semantic[:12], 16) % 10
        assigned = "support" if value < 6 else "validation" if value < 8 else "test"
        if (unique and semantic in seen) or assigned != bucket:
            continue
        seen.add(semantic)
        rows.append(row)
        if len(rows) == count:
            return rows
    raise ValueError("Binding generator exhausted its unique-case budget")
