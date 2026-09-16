import copy
import itertools

import pytest
import torch

from experiments.continuing_control.core import InteractiveR1
from experiments.grounded_language.data import (
    VOCAB,
    examples,
    execute,
    independent_solutions,
    partition,
    tokens,
)
from experiments.grounded_language.model import LanguageR1, tensors, trainable
from sera.generative import GenerativeSharedR1
from workbench.owner import LiveR1


def small_owner():
    owner = GenerativeSharedR1(width=12, heads=2, memory_dim=4, kind="delta")
    owner = InteractiveR1.extend(owner, window=32)
    LiveR1.attach(owner)
    return LanguageR1.attach(owner, 34001)


def test_supplied_formal_checker_exhaustive():
    for labels in itertools.product(range(2), range(11), range(11), range(11)):
        assert execute(list(labels))["solutions"] == independent_solutions(list(labels))
    assert execute([0, 0, 1, 10])["solutions"] == list(range(11))
    assert execute([1, 0, 1, 10])["solutions"] == []


def test_semantic_split_and_missing_word_boundary():
    banks = {split: examples(151, 128, split) for split in ("train", "development", "final")}
    assert all(partition(*r["labels"][1:]) in {2, 3, 4} for r in banks["train"])
    triples = [{tuple(r["labels"][1:]) for r in bank} for bank in banks.values()]
    assert all(not a & b for a, b in itertools.combinations(triples, 2))
    owner = small_owner()
    assert owner.interpret(["scale x by three then subtract one to get two modulo eleven"])[0]["missing_words"] == ["scale"]
    assert tokens("unknownword")[0][0] == 1
    with pytest.raises(ValueError):
        tokens("x "*100)


def test_word_update_changes_only_explicit_embedding_and_roundtrip():
    owner = small_owner()
    prior = copy.deepcopy(owner.state_dict())
    hook = trainable(owner, "word", novel_word="scale")
    ids, labels = tensors(examples(332, 8, "train", lesson=True))
    optimizer = torch.optim.AdamW([p for p in owner.parameters() if p.requires_grad], lr=.01, weight_decay=0)
    logits = owner.language(ids)
    loss = sum(torch.nn.functional.cross_entropy(v, labels[:, i]) for i, v in enumerate(logits))
    loss.backward()
    optimizer.step()
    hook.remove()
    for name, value in owner.state_dict().items():
        if name == "language_embedding.weight":
            difference = (value != prior[name]).any(1)
            assert difference.nonzero().flatten().tolist() == [VOCAB["scale"]]
        else:
            assert torch.equal(value, prior[name])
    restored = small_owner()
    restored.load_state_dict(owner.state_dict(), strict=True)
    assert restored.interpret(["take x minus two multiply by five and obtain six modulo eleven"]) == owner.interpret(["take x minus two multiply by five and obtain six modulo eleven"])


def test_formal_check_does_not_certify_translation():
    wrong = execute([0, 2, 3, 4])
    assert wrong["formal_verification"]
    assert wrong["solutions"] != independent_solutions([1, 2, 3, 4])
