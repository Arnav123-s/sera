"""Finite typed sequence rules acquired from admitted examples, with explicit priors."""

from __future__ import annotations

import torch

from sera.storage import digest

RULES = ("first_binding", "last_binding", "marked_retrieval", "majority")


def execute_rule(rule, inputs):
    if rule not in RULES or inputs.ndim != 2 or inputs.shape[-1] != 20:
        raise ValueError("Sequence rule input contract violated")
    values = inputs[:-1, 4:8].argmax(-1)
    if rule == "majority":
        return int(torch.bincount(values, minlength=4).argmax())
    if rule == "marked_retrieval":
        return int(values[inputs[:-1, 13].argmax()])
    keys = inputs[:-1, :4].argmax(-1)
    query = int(inputs[-1, :4].argmax())
    indices = (keys == query).nonzero().flatten()
    if not len(indices):
        raise ValueError("No matching key in the declared rule domain")
    return int(values[indices[0 if rule == "first_binding" else -1]])


def acquire_rule(support, verification):
    candidates = []
    for rule in RULES:
        if all(execute_rule(rule, x) == int(y) for x, y in zip(support.inputs, support.targets)):
            candidates.append(rule)
    if len(candidates) != 1:
        raise ValueError("Evidence does not identify a unique sequence rule")
    rule = candidates[0]
    if support.dataset_id == verification.dataset_id:
        raise ValueError("Verification must be independent of rule discovery")
    if not all(execute_rule(rule, x) == int(y)
               for x, y in zip(verification.inputs, verification.targets)):
        raise ValueError("Sequence rule failed independent verification")
    return {"kind": "sequence_rule", "rule": rule,
            "signature": "symbolic_sequence[time,20] -> categorical[4]",
            "source": "finite supplied DSL; rule selected from labeled evidence",
            "support_id": support.dataset_id, "verification_id": verification.dataset_id,
            "support_examples": len(support.inputs), "verification_examples": len(verification.inputs),
            "content_sha256": digest([rule, support.dataset_id, verification.dataset_id])}
