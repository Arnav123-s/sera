"""Scientific admission is invariant to bounded inverse-optimizer bookkeeping."""

import copy

import pytest

from experiments.gap_inquiry import ROOT, digest, read
from experiments.intervention_portability import STRICT, compare


@pytest.fixture
def receipt():
    path = ROOT / "runs/IU-study-001/final/completed/mixed-47101-information.json"
    if not path.exists():
        pytest.skip("Restore the exact current intervention evidence")
    return read(path)["result"]["qualification"]


def rehash(value):
    value["receipt"] = digest({k: v for k, v in value.items() if k != "receipt"})
    return value


def test_optimizer_bookkeeping_is_not_a_changed_admission(receipt):
    new = copy.deepcopy(receipt)
    new["independent_fit"]["iterations"] += 1
    new["independent_fit"]["message"] = "Different platform convergence description"
    rehash(new)
    assert not STRICT(receipt, new)
    assert compare(receipt, new, emit=False)


def test_bounded_inverse_fit_rounding_is_rechecked_in_prediction_space(receipt):
    new = copy.deepcopy(receipt)
    new["independent_fit"]["weights"][0] += 1e-9
    rehash(new)
    assert compare(receipt, new, emit=False)
    new["independent_fit"]["weights"][0] += 1e-3
    rehash(new)
    assert not compare(receipt, new, emit=False)


@pytest.mark.parametrize("field", ["accepted", "source", "goal", "data", "model", "guard", "likelihood", "adequacy", "receipt"])
def test_meaningful_changes_remain_rejected(receipt, field):
    new = copy.deepcopy(receipt)
    new["independent_fit"]["iterations"] += 1
    if field == "accepted":
        new["accepted"] = not new["accepted"]
    elif field == "source":
        new["source"] = "different-source"
    elif field == "goal":
        new["goal_id"] = "different-goal"
    elif field == "data":
        new["audit"][0]["groups"][0]["plus"] += 1
    elif field == "model":
        new["weight_identity"] = "different-weights"
    elif field == "guard":
        new["independent_fit"]["qualified"] = not new["independent_fit"]["qualified"]
    elif field == "likelihood":
        new["independent_fit"]["nll"] += .0001
    elif field == "adequacy":
        new["adequacy"]["passed"] = not new["adequacy"]["passed"]
    if field != "receipt":
        rehash(new)
    assert not compare(receipt, new, emit=False)


def test_portable_successor_keeps_exact_owner_and_answer(receipt):
    from scripts.intervention_live import restore
    session, growth = restore()
    audit = read(ROOT / "research-continuation/47_intervention_understanding/audit.json")
    assert session.identity() == audit["owner"]
    assert growth.base.base.base.base.base.base.grounded.owner is session.owner
    answer = session.answer("mixed-47101-information")
    original = read(ROOT / "runs/IU-study-001/final/completed/mixed-47101-information.json")["result"]["answer"]
    for field in ("status", "goal_id", "original_goal", "program", "model", "selected", "revision", "alternatives", "model_adequacy"):
        assert answer[field] == original[field]
    assert answer["plus_probability"] == pytest.approx(original["plus_probability"], abs=1e-12)
    assert answer["model_confidence"]["local_standard_error"] == pytest.approx(original["model_confidence"]["local_standard_error"], abs=1e-10, rel=1e-8)
