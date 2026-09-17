"""Meaningful empirical-evidence boundaries and actual-owner restoration contracts."""

import copy

import numpy as np
import pytest
import torch

from experiments.concept_refinement import physical
from experiments.concept_refinement.audit import observations
from experiments.concept_refinement.common import EXPERIMENT, RUNS, read
from experiments.concept_refinement.data import bank, features, identity, step
from experiments.concept_refinement.runtime import RefinementSession, arrays, validate_observations


@pytest.fixture(scope="module")
def source():
    path = RUNS / "dev.json"
    rows = read(path) if path.exists() else bank("dev")
    return next(r for r in rows if r["family"] == "delayed" and r["condition"] == "clean")


@pytest.fixture(scope="module")
def saved(source):
    if not (EXPERIMENT / "selection.json").exists() or not (RUNS / "final.json").exists():
        pytest.skip("Run the frozen concept study before actual-owner integration tests")
    session = RefinementSession()
    result = session.autonomous_refine(
        "original-task", "body-1", source["future_controls"], observations(source)
    )
    assert result["status"] == "EMPIRICAL_PREDICTION"
    return session.snapshot()


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", "IMAGINED"),
        ("units", ["ft", "m/s", "1"]),
        ("subject", "another-body"),
        ("velocity", float("nan")),
        ("available", [True, 1]),
        ("source", ""),
    ],
)
def test_observation_contract(source, field, value):
    items = observations(source)
    items[2][field] = value
    with pytest.raises(ValueError):
        validate_observations("body-1", items)


def test_clock_and_unavailable_value(source):
    items = observations(source)
    items[2]["time"] = items[1]["time"]
    with pytest.raises(ValueError, match="clock"):
        validate_observations("body-1", items)
    items = observations(source)
    items[2]["available"][1] = False
    with pytest.raises(ValueError, match="null"):
        validate_observations("body-1", items)


def test_no_assessor_values_in_features(source):
    values, masks, controls = arrays(observations(source))
    x = features(values[:-1], controls, masks[:-1])
    changed = copy.deepcopy(source)
    changed["assessor_hidden"] = [999] * len(changed["assessor_hidden"])
    changed["assessor_parameters"] = {"tau": 1234}
    changed["truth"] = [[999, 999]] * 12
    v, m, u = arrays(observations(changed))
    assert np.array_equal(x, features(v[:-1], u, m[:-1]))


def test_simple_teacher_has_no_hidden_transient():
    for split in ("train", "dev"):
        path = RUNS / f"{split}.json"
        for row in read(path) if path.exists() else bank(split):
            if row["family"] != "simple":
                continue
            assert all(value == 0 for value in row["assessor_hidden"])
            if split == "dev" and row["condition"] == "clean":
                model = physical.fit(row["values"], row["masks"], row["controls"], force="instant")
                np.testing.assert_allclose(
                    physical.predict(model, row["values"], row["future_controls"]),
                    row["truth"],
                    atol=1e-6,
                    rtol=0,
                )


@pytest.mark.parametrize("tamper", ["weights", "tau", "history", "source", "extra_weight"])
def test_restore_rejects_changed_support(saved, tamper):
    snapshot = copy.deepcopy(saved)
    model = snapshot["subjects"]["body-1"]["models"][0]
    if tamper == "weights":
        snapshot["weights"][model["weight_key"]][0] += 0.1
    if tamper == "tau":
        model["tau"] = 20
    if tamper == "history":
        snapshot["subjects"]["body-1"]["observations"][4]["velocity"] += 0.5
    if tamper == "source":
        snapshot["source"] = "wrong"
    if tamper == "extra_weight":
        snapshot["weights"]["unowned"] = [1, 2]
    with pytest.raises(ValueError):
        RefinementSession(snapshot)


def test_branch_and_live_weight_validation(saved):
    session = RefinementSession(saved)
    before = identity(session.snapshot())
    a = session.predict("body-1", [0.2] * 12)
    b = session.predict("body-1", [1.0] * 12)
    assert a["position_velocity"] != b["position_velocity"]
    assert identity(session.snapshot()) == before
    assert not a["occurrence_established"]
    with torch.no_grad():
        next(iter(session.owner.empirical_weights.values()))[0] += 0.1
    with pytest.raises(ValueError, match="Weights"):
        session.predict("body-1", [0.2])


def test_goal_reuse_and_exact_route(saved, source):
    session = RefinementSession(saved)
    weight_count = len(session.owner.empirical_weights)
    result = session.autonomous_refine("original-task", "body-1", source["future_controls"])
    assert result["attempts"][-1]["learning"]["status"] == "REUSED"
    assert len(session.owner.empirical_weights) == weight_count
    with pytest.raises(ValueError, match="immutable"):
        session.autonomous_refine("original-task", "body-1", [0.3])
    exact = session.exact_motion([2, 3, 1], "3", "5", "-1")
    assert exact["status"] == "CERTIFIED_ALGEBRA" and exact["result"]["position"] == "125/4"
    assert session.predict("unseen", [0.3])["status"] == "MISSING_KNOWLEDGE"


def test_rejected_acquisition_is_transactional(saved, source):
    session = RefinementSession(saved)
    before = identity(session.snapshot())
    with pytest.raises(ValueError):
        session.observe("body-1", observations(source))
    assert identity(session.snapshot()) == before


def test_constant_command_does_not_identify_gain_and_bias(saved):
    session = RefinementSession(saved)
    parameters = {"b": 1.2, "w": 0.1, "d": 0.2, "c": 0.0, "tau": 0.4, "q": 0.0}
    state = np.array([10.0, 0.0, 0.0])
    states = [state[:2].tolist()]
    for _ in range(80):
        state = step(state, 0.8, parameters)
        states.append(state[:2].tolist())
    row = {
        "id": "development/constant-command",
        "values": states,
        "controls": [0.8] * 80,
        "future_controls": [0.8] * 12,
        "masks": [[True, True]] * 81,
    }
    session.observe("ambiguous", observations(row, "ambiguous"))
    result = session.refine("ambiguous")
    assert result["status"] == "UNIDENTIFIABLE_CAUSE"
    assert session.subjects["ambiguous"]["active"] is None
