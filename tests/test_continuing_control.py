import copy

import numpy as np
import pytest
import torch

from experiments.continuing_control.control import decide
from experiments.continuing_control.core import ContinuingSession, InteractiveR1, sensor
from experiments.continuing_control.environment import World, specification
from sera.contracts import EvidenceKind, Observation, Provenance
from sera.generative import GenerativeSharedR1


def tiny_parent():
    model = GenerativeSharedR1(width=16, heads=2, memory_dim=4)
    model.generator_mean[0, :4] = torch.tensor([.25, -.4, .8, .3])
    return model


def fixture():
    parent = tiny_parent()
    session = ContinuingSession(InteractiveR1.extend(parent, 12))
    world = World(specification(9219, "gain_reversal", 60))
    for index in range(3):
        if index:
            world.step(0)
        session.admit(sensor(world.measure(), index, str(index)), 0 if index else None)
    return parent, session, world


def test_imagined_points_are_not_factual_and_actions_are_required():
    _, session, world = fixture()
    imagined = Observation("numeric", tuple(world.measure()), 3,
                           Provenance("a08-paid-sensor", "hypothesis", EvidenceKind.PREDICTION), units="m")
    with pytest.raises(ValueError, match="paid"):
        session.admit(imagined, 1)
    with pytest.raises(ValueError, match="executed action"):
        session.admit(sensor(world.measure(), 3, "sensor"))


def test_plans_are_conditional_and_invalidated_by_real_changes():
    _, session, world = fixture()
    before = copy.deepcopy(session.snapshot())
    plan = decide(session, world.goals(4), "mpc")
    assert session.snapshot()["buffers"] == before["buffers"]
    assert session.events == before["events"]
    session.validate_plan(plan)
    session.owner.interaction_mean.add_(.01)
    with pytest.raises(ValueError, match="Stale"):
        session.validate_plan(plan)
    with pytest.raises(ValueError, match="Privileged"):
        decide(session, world.goals(4), "mpc", oracle_coefficients=world.coefficients())


def test_online_updates_and_replay_keep_exact_state_and_cumulative_costs():
    parent, session, world = fixture()
    for index in range(3, 20):
        action = (-1, 0, 1)[index % 3]
        world.step(action)
        session.admit(sensor(world.measure(), index, str(index)), action)
    original = parent.state_dict()
    assert all(torch.equal(v, session.owner.state_dict()[k]) for k, v in original.items())
    restored = ContinuingSession.restore(parent, session.snapshot())
    assert restored.snapshot()["buffers"] == session.snapshot()["buffers"]
    assert restored.work["paid_sensor_measurements"] == 20
    assert restored.work["executed_actions"] == 19
    assert restored.work["factual_replay_events"] == 20
    assert np.linalg.norm(restored.owner.interaction_mean.numpy()-np.array([.85, .06, 0.])) > .01
    wrong = session.snapshot()
    wrong["work"]["paid_sensor_measurements"] = 0
    with pytest.raises(ValueError, match="costs"):
        ContinuingSession.restore(parent, wrong)


def test_identical_visible_state_does_not_expose_hidden_velocity():
    spec = specification(9821, "gain_reversal", 60)
    other = {**spec, "velocity": -spec["velocity"]}
    a, b = World(spec), World(other)
    assert np.array_equal(a.measure(), b.measure())
    assert not np.array_equal(a.step(0), b.step(0))


def test_observed_innovations_revise_a_reversed_action_response():
    parent = tiny_parent()
    session = ContinuingSession(InteractiveR1.extend(parent, 12, detect_change=True))
    world = World(specification(19919, "gain_reversal", 60))
    session.admit(sensor(world.measure(), 0, "initial"))
    for index in range(1, 61):
        action = (-1, 1, 0)[index % 3]
        world.step(action)
        session.admit(sensor(world.measure(), index, str(index)), action)
    assert int(session.owner.interaction_changes) >= 1
    assert float(session.owner.interaction_mean[1]) < -.04
    assert len(session.events) == 61  # A change revises sufficient statistics, never erases factual history.
    assert ContinuingSession.restore(parent, session.snapshot()).snapshot()["buffers"] == session.snapshot()["buffers"]


def test_restored_continuing_state_makes_identical_future_decisions():
    parent, original, world = fixture()
    restored = ContinuingSession.restore(parent, original.snapshot())
    paired_world = copy.deepcopy(world)
    for _ in range(8):
        a = decide(original, world.goals(4), "mpc")
        b = decide(restored, paired_world.goals(4), "mpc")
        assert a["actions"] == b["actions"]
        assert a["particles"] == b["particles"]
        for session, environment, plan in ((original, world, a), (restored, paired_world, b)):
            environment.step(plan["action"])
            session.admit(sensor(environment.measure(), len(session.events), str(environment.time)), plan["action"])
    assert original.snapshot()["buffers"] == restored.snapshot()["buffers"]
    assert original.work["paid_sensor_measurements"] == restored.work["paid_sensor_measurements"] == 11


def test_independent_complex_plane_auditor_matches_the_conditional_plan():
    from experiments.continuing_control.audit import IndependentFit, independent_plan

    _, session, world = fixture()
    center, radius = session.owner.geometry()
    fit = IndependentFit(center, 12, True, False)
    for event in session.events:
        fit.update(event["values"], event["action"])
    plan = decide(session, world.goals(4), "mpc")
    assert independent_plan(fit, radius, world.goals(4), plan) < 1e-12
