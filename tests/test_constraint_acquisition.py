from types import SimpleNamespace

from experiments.autonomous_discovery.acquire import construct
from experiments.autonomous_discovery.core import reference_step
from experiments.self_study.algebra import SIZE, check, independent


def test_candidate_is_constructed_from_retained_operator_and_constraint_feedback():
    integral = SimpleNamespace(propose=lambda p: list(map(str, reference_step("I", p))))
    unknown = SimpleNamespace(propose=lambda p: ["0"]*SIZE)
    owner = SimpleNamespace(study_maps={"integral": integral, "sum": unknown})
    for p in ([1], [0, 0, 1]):
        record = construct(owner, "sum", p)
        assert record["acquired"]
        assert check("sum", p, record["candidate"])["accepted"]
        assert independent("sum", p, record["candidate"])
        assert record["feedback_kind"].endswith("automated supervision")
    known = construct(owner, "integral", [1])
    assert not known["acquired"]
