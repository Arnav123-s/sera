import copy

import pytest
import torch

from experiments.book_learning.ground_data import GROUND
from experiments.book_learning.runtime import PREMISES, GroundedSession, number
from experiments.human_reading.data import read
from sera.session_state import model_identity


@pytest.fixture(scope="module")
def session():
    return GroundedSession()


def request(role="force", identifier="force-check"):
    rows = read(GROUND / "teaching.json")["train"]
    row = next(r for r in rows if r["term"] == role and r["label"] == role and "entry" not in r)
    return {"id": identifier, "goal": "Check motion under the attributed physical meaning",
            "term": row["term"], "definition": row["text"], "source": row["source"],
            "value": "6", "unit": "N", "mass": "3", "time": "2",
            "assumptions": {**dict.fromkeys(PREMISES, True), "net_force": True}}


def test_human_definition_reaches_existing_certified_operator_without_observations(session):
    before = model_identity(session.owner)
    observations = copy.deepcopy(session.base.subjects)
    result = session.imagine(request())
    assert result["status"] == "CONDITIONAL_RESULT"
    assert result["binding"]["admission"] == "TAUGHT_SOURCE_REUSE"
    assert result["branches"]["base"]["result"]["position"] == "4"
    assert result["branches"]["base"]["result"]["velocity"] == "4"
    assert result["branches"]["opposite_drive"]["result"]["position"] == "-4"
    assert result["branches"]["double_mass"]["result"]["position"] == "2"
    assert not result["branches"]["base"]["lexical_interpretation_certified"]
    assert model_identity(session.owner) == before and session.base.subjects == observations
    session.base.assert_owner()


@pytest.mark.parametrize("change", [
    {"unit": "kg"}, {"mass": "0"}, {"mass": "-1"}, {"time": "-2"},
    {"assumptions": {}}, {"acceleration": "19"}, {"source": "unverified replacement"},
])
def test_incompatible_physical_premises_never_certify_execution(session, change):
    result = session.imagine(request(identifier="premise-check") | change)
    assert result["status"] == "RETAINED_OPEN" and "branches" not in result
    assert result["original_goal_preserved"]


def test_nonphysical_mass_sense_cannot_become_mechanics_by_supplying_kilograms(session):
    row = next(r for r in session.bank if r["label"] == "other" and "Eucharist" in r["text"])
    value = request(identifier="sense-check") | {"term": row["term"], "definition": row["text"], "source": row["source"], "unit": "kg"}
    result = session.imagine(value)
    assert result["status"] == "RETAINED_OPEN"
    assert not result["binding"]["admitted"]


def test_interrupted_source_acquisition_retains_exact_original_task(session):
    class Unavailable:
        def search(self, *args, **kwargs):
            raise OSError("independent unavailable-source probe")
    value = request(identifier="source-outage") | {"term": "hysteresis", "definition": "History dependence", "public_gap": "hysteresis"}
    result = session.imagine(value, acquire=True, client=Unavailable())
    assert result["status"] == "RETAINED_OPEN"
    assert result["request"] == value
    assert result["investigation"]["question"] == value["goal"]
    assert result["investigation"]["weight_updates"] == 0


def test_persisted_source_meaning_and_owner_cannot_be_silently_changed(session):
    session.imagine(request(identifier="restore-check"))
    snapshot = session.snapshot()
    assert GroundedSession(snapshot).snapshot() == snapshot
    changed = copy.deepcopy(snapshot)
    first = next(iter(changed["concepts"].values()))
    first["type"] = "mass"
    with pytest.raises(ValueError, match="source"):
        GroundedSession(changed)
    changed = copy.deepcopy(snapshot)
    changed["identity"]["owner"] = "0"*64
    with pytest.raises(ValueError, match="owner"):
        GroundedSession(changed)


@pytest.mark.parametrize("value", [True, float("nan"), "1/0", "1e300", "-1/0", "2/100000000000000"])
def test_rational_domain_rejects_unsupported_quantities(value):
    with pytest.raises(ValueError):
        number(value)


def test_parameters_belong_to_one_actual_owner_and_runtime_does_not_train(session):
    assert not any(p.requires_grad for p in session.owner.parameters())
    assert isinstance(session.owner.ground_heads[session.kind].weight, torch.nn.Parameter)
    assert session.owner is session.base.owner is session.base.study.owner
