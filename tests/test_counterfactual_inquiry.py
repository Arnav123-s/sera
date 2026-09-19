import copy
from fractions import Fraction as Q

import pytest

from experiments.counterfactual_core import Z, admitted, at, curve, describe, digest


def test_model_limit_does_not_admit_the_zero_endpoint():
    c = {"curve": curve(1 / Z), "conditions": {"positive": [curve(Z)], "integer": [], "nonzero": []}}
    description = describe(tuple(c["curve"]["n"]), tuple(c["curve"]["d"]))
    assert description["zero_right_limit"] == "+infinity"
    assert admitted(c, Q(1, 1000))
    assert not admitted(c, 0)
    assert not admitted(c, -1)


def test_cancellation_preserves_original_applicability_guard():
    c = {"curve": curve((Z - 1) / (Z - 1)), "conditions": {"positive": [], "integer": [], "nonzero": [curve(Z - 1)]}}
    assert at(c["curve"], 1) == 1
    assert not admitted(c, 1)
    assert admitted(c, 2)


def test_discrete_time_is_not_silently_continued_to_real_time():
    c = {"curve": curve(Z * (Z - 1) / 2), "conditions": {"positive": [curve(Z)], "integer": [curve(Z)], "nonzero": []}}
    assert not admitted(c, Q(3, 2))
    assert admitted(c, 3) and at(c["curve"], 3) == 3


def test_commitment_cannot_be_reused_for_a_changed_predictor():
    from experiments.counterfactual_check import validate_event
    event = {"predictor": "old", "question": {"id": "goal"}, "candidates": []}
    event["commitment"] = digest(event)
    with pytest.raises(ValueError, match="predictor"):
        validate_event(event, "new")


def test_thresholds_and_reversals_remain_distinct_from_poles():
    c = curve((Z - 2) ** 2)
    analysis = describe(tuple(c["n"]), tuple(c["d"]))
    assert analysis["zeros"] == [{"interval": ["2", "2"], "multiplicity": 2}]
    assert analysis["stationary_points"] == [{"interval": ["2", "2"], "multiplicity": 1}]
    assert analysis["poles"] == []


@pytest.fixture(scope="module")
def committed_case():
    from experiments.counterfactual_core import propose
    from experiments.counterfactual_loop import commit
    from experiments.gap_inquiry import ROOT, read
    path = ROOT / "runs/CI-study-001/frontier.json"
    if not path.exists():
        pytest.skip("Optional published counterfactual frontier")
    frontier = read(path)
    question = next(q for q in frontier["questions"] if q["domain"] == "motion_0" and q["axis"] == "m" and q["target"] == "a0")
    return commit(propose(question, frontier["matrices"]), "test-predictor"), frontier["matrices"]


def test_proposer_does_not_read_reference_worlds_or_solutions(committed_case, monkeypatch):
    from experiments.counterfactual_core import propose
    from experiments.self_chosen import equations
    event, matrices = committed_case
    def prohibited(*args, **kwargs):
        raise AssertionError("Reference solution leaked into imagination")
    monkeypatch.setattr(equations, "symbolic_world", prohibited)
    monkeypatch.setattr(equations, "reference_rows", prohibited)
    assert propose(event["question"], matrices)["candidates"] == event["candidates"]


def test_independent_checker_rejects_changed_derivative_and_guard(committed_case):
    from experiments.counterfactual_check import check_candidate, validate_event
    event, _ = committed_case
    assert validate_event(event, "test-predictor")["accepted"]
    c = copy.deepcopy(next(c for c in event["candidates"] if c["conditions"]["nonzero"]))
    c["conditions"]["nonzero"] = []
    assert not check_candidate(event["question"], c)["accepted"]
    c = copy.deepcopy(event["candidates"][0])
    c["analysis"]["derivative"] = curve(12345 * Z)
    assert not check_candidate(event["question"], c)["accepted"]


def test_interrupted_investigation_resumes_and_rejects_forged_evidence(committed_case):
    from experiments.counterfactual_check import Simulation, replay_investigation
    from experiments.counterfactual_loop import investigate
    event, matrices = committed_case
    source = Simulation(event, 45678, omitted=True)
    snapshots = []
    class Interrupted(Exception):
        pass
    def boundary(state):
        snapshots.append(state)
        if len(state["observations"]) == 1:
            raise Interrupted()
    with pytest.raises(Interrupted):
        investigate(event, matrices, source, "entropy", [0.] * 8, 45678, boundary)
    resumed = investigate(event, matrices, source, "entropy", [0.] * 8, 45678, lambda _: None, snapshots[-1])
    assert replay_investigation(resumed, source, matrices)["accepted"]
    forged = copy.deepcopy(resumed)
    forged["observations"][0]["value"] += 1
    forged["id"] = digest({k: v for k, v in forged.items() if k != "id"})
    with pytest.raises(ValueError, match="replay differs"):
        replay_investigation(forged, source, matrices)


def test_scope_saturation_moves_to_other_questions():
    from experiments.counterfactual_loop import next_question
    frontier = [{"id": "a", "domain": "one", "models": [1, 2]}, {"id": "b", "domain": "two", "models": [1]}]
    first = next_question(frontier, {})
    second = next_question(frontier, {first["id"]: first})
    assert first["id"] != second["id"]
    assert next_question(frontier, {q["id"]: q for q in frontier}) is None


def test_constant_and_dependent_backgrounds_have_distinct_generalization(committed_case):
    from experiments.counterfactual_check import check_generalization
    from experiments.counterfactual_core import generalized_shape
    event, matrices = committed_case
    claims = [generalized_shape(matrices, c, event["question"]["target"]) for c in event["candidates"]]
    useful = [c for c in claims if c and not c["tautological_control"]]
    assert useful and all(check_generalization(c)["accepted"] for c in useful)
    assert any(c["shape"] == {"n": ["1"], "d": ["0", "1"]} for c in useful)
