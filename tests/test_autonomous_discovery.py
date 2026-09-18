from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from experiments.autonomous_discovery.core import (
    WORDS,
    AcquisitionGap,
    canonical,
    certify,
    consequence_span,
    elimination_proposals,
    execute,
    numerical_proposals,
    reference,
    reference_step,
)
from experiments.self_study.algebra import check, independent


def old_relation():
    row = [Q(0)]*len(WORDS)
    for word, value in (("II", 1), ("IM", 1), ("MI", -1)):
        row[WORDS.index(word)] = Q(value)
    return row


def test_separate_summation_checker_satisfies_two_old_contracts():
    p = ["1", "-2", "0", "3"]
    q = list(map(str, reference_step("S", p)))
    assert check("sum", p, q)["accepted"]
    assert independent("sum", p, q)


def test_composed_old_knowledge_and_aliases_do_not_mint_discovery():
    row = old_relation()
    span = consequence_span([{"coefficients": row}])
    composed = [Q(0)]*len(WORDS)
    for word, c in zip(WORDS, row, strict=True):
        if c:
            composed[WORDS.index("S"+word)] += c
    assert certify(composed)["accepted"]
    assert not span.add(composed)
    assert canonical(row) == canonical([7*c for c in row])
    wrong = row[:]
    wrong[WORDS.index("MI")] = 1
    assert not certify(wrong)["accepted"]


def test_two_proposal_procedures_use_execution_values():
    names = [w for w in WORDS if len(w) <= 2]
    columns = [[v for d in range(4) for v in reference(w, [0]*d+[1])] for w in names]
    a, b = numerical_proposals(names, columns), elimination_proposals(names, columns)
    assert a and set(a) == set(b)
    assert all(certify(row)["accepted"] for row in a)
    with pytest.raises(ValueError):
        reference("IIII", [1])


def test_verified_reconstruction_improvement_persists_without_repeat_credit():
    from experiments.self_study.algebra import SIZE, vector

    weight = torch.zeros(SIZE, SIZE, dtype=torch.float64)
    for i in range(SIZE-1):
        weight[i, i+1] = 1/(i+1)

    def propose(p):
        x = torch.tensor(list(map(float, vector(p))), dtype=torch.float64)
        return [str(Q(float(v)).limit_denominator(120)) for v in x @ weight]

    owner = SimpleNamespace(study_maps={"integral": SimpleNamespace(weight=weight, propose=propose)},
                            autonomous_denominators=torch.tensor([120, 120]))
    with pytest.raises(AcquisitionGap):
        execute(owner, "I", ["1/240"], {}, {"primitive_calls": 0})
    counts = {"primitive_calls": 0, "learning": True}
    assert execute(owner, "I", ["1/240"], {}, counts)[1] == Q(1, 240)
    assert owner.autonomous_denominators.tolist() == [240, 120]
    assert len(counts["procedure_updates"]) == 1
    repeat = {"primitive_calls": 0, "learning": True}
    execute(owner, "I", ["1/240"], {}, repeat)
    assert not repeat.get("procedure_updates")
    frozen = {"primitive_calls": 0}
    assert execute(owner, "I", ["1/240"], {}, frozen)[1] == Q(1, 240)
    assert owner.autonomous_denominators.tolist() == [240, 120]
