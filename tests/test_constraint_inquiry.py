"""Actual-owner lifecycle and evidence-boundary checks for continuous investigation."""

import copy

import numpy as np
import pytest
import torch

from experiments.constraint_inquiry.data import encode
from experiments.constraint_inquiry.model import language_identity
from experiments.constraint_inquiry.runtime import ConstraintRuntime
from experiments.constraint_inquiry.settling import context, endpoints, energy, step
from experiments.language_inquiry.study import ROOT, read
from sera.session_state import model_identity


@pytest.fixture(scope="module")
def runtime():
    selected = ROOT / "runs/CI-fits-final-001/selected.json"
    if not selected.exists():
        pytest.skip("Requires the qualified local parent and new fitted interfaces")
    return ConstraintRuntime(read(selected))


def test_actual_owner_and_existing_interfaces(runtime):
    learner = runtime.base.session.learner
    assert runtime.owner is learner.solver.neural.owner is learner.solver.components["typed"].owner
    assert runtime.owner is learner.session.owner is runtime.base.session.owner
    assert len(runtime.base.session.tasks) == 6


@pytest.mark.parametrize("text", ["can it reach beacon", "can alien reach orbit", "can orbit orbit reach", ""])
def test_unknown_and_ambiguous_language_stays_missing(text):
    with pytest.raises(ValueError):
        encode([text])


def test_imagining_does_not_create_facts_and_missing_is_explicit(runtime):
    before = model_identity(runtime.owner)
    count = len(runtime.base.session.learner.session.events)
    answer = runtime.ask("missing", "can orbit reach beacon")
    assert answer["status"] == "MISSING_LOCATION"
    assert model_identity(runtime.owner) == before
    assert len(runtime.base.session.learner.session.events) == count


def test_role_reversal_and_event_are_different_requests(runtime):
    assert runtime.ask("reverse", "can beacon reach orbit")["status"] == "MISSING_DYNAMICS"
    assert runtime.ask("event", "did orbit reach beacon")["status"] == "MISSING_EVENT_EVIDENCE"
    assert runtime.ask("malformed", "reach can not orbit beacon")["status"] == "MISSING_LANGUAGE"


def test_unqualified_surface_family_is_not_silently_accepted(runtime):
    assert runtime.ask("withheld-wording", "please can orbit reach beacon")["status"] == "MISSING_LANGUAGE"


def test_grounded_acquisition_reopens_request(runtime):
    before = language_identity(runtime.owner)
    answer = runtime.acquire("beacon")
    assert len(answer["paid_observations"]) == 3
    assert runtime.jobs["missing"]["status"] == "pending"
    assert language_identity(runtime.owner) == before
    assert runtime.jobs["event"]["status"] == "MISSING_EVENT_EVIDENCE"


def test_factual_gate_rejects_relabelled_imagination_duplicate_and_bad_units(runtime):
    original = copy.deepcopy(runtime.observations[0])
    for changes in ({}, {"kind": "conditional_imagination"}, {"units": "pixels"},
                    {"source": "web-description"}, {"values": [float("nan"), 0.], "record_id": "nan"}):
        row = {**original, **changes}
        before = model_identity(runtime.owner)
        with pytest.raises(ValueError):
            runtime.admit(row)
        assert model_identity(runtime.owner) == before


def test_refinement_does_not_mutate_owner(runtime):
    before = model_identity(runtime.owner)
    answer = runtime.work_on("missing", 4)
    assert answer["steps"] == 4
    assert answer["result"]["occurrence"] == "NOT_ESTABLISHED"
    assert model_identity(runtime.owner) == before


def test_residual_stop_keeps_qualified_evidence_status(runtime):
    runtime.ask("fast", "please beacon can be reached by orbit")
    before = model_identity(runtime.owner)
    answer = runtime.solve("fast")
    assert answer["status"] == "complete"
    assert answer["steps"] <= 12
    assert answer["result"]["occurrence"] == "NOT_ESTABLISHED"
    assert "UNCALIBRATED" in answer["result"]["applicability"]
    assert model_identity(runtime.owner) == before


def test_resume_and_independent_copy_are_exact(runtime):
    saved = runtime.snapshot()
    restored = ConstraintRuntime(runtime.checkpoint, saved)
    original = runtime.work_on("missing", 3)
    resumed = restored.work_on("missing", 3)
    assert original == resumed
    duplicate = copy.deepcopy(runtime.owner)
    assert duplicate is not runtime.owner
    assert duplicate.interpret("can orbit reach beacon") == runtime.owner.interpret("can orbit reach beacon")


def test_changed_partial_state_rejected(runtime):
    record = runtime.snapshot()
    record["jobs"]["missing"]["points"][0][0] += .01
    with pytest.raises(ValueError, match="replay"):
        ConstraintRuntime(runtime.checkpoint, record)


def test_unrelated_evidence_does_not_stale_branch(runtime):
    before = runtime.jobs["missing"]["dependency"]
    runtime.acquire("marker")
    assert runtime.jobs["missing"]["dependency"] == before
    assert runtime.jobs["missing"]["status"] == "pending"


def test_correction_stales_only_dependent_targets(runtime):
    row = {**runtime.observations[0], "record_id": "beacon:correction:0"}
    row["values"] = [row["values"][0] + .1, row["values"][1]]
    runtime.admit(row)
    assert runtime.jobs["missing"]["status"] == "stale"
    old = copy.deepcopy(runtime.jobs["missing"])
    runtime.rebase("missing")
    assert runtime.history[-1] == old
    assert runtime.jobs["missing"]["steps"] == 0


def test_changed_encoder_invalidates_its_dependent_branch(runtime):
    candidate = copy.copy(runtime)
    candidate.owner = copy.deepcopy(runtime.owner)
    candidate.jobs = copy.deepcopy(runtime.jobs)
    with torch.no_grad():
        candidate.owner.cloud_roles.weight[0, 0].add_(.1)
    candidate.sync()
    assert candidate.jobs["missing"]["status"] == "stale"
    assert runtime.jobs["missing"]["status"] == "pending"


def test_static_parameter_branches_and_gradient(runtime):
    model = context(runtime.owner, [0., 0.])
    controls = torch.tensor([[.2, -.3]], dtype=torch.float64, requires_grad=True)
    analytic, = torch.autograd.grad(energy(model, controls).sum(), controls)
    for j in range(2):
        offset = torch.zeros_like(controls)
        offset[0, j] = 1e-5
        numeric = (energy(model, controls + offset) - energy(model, controls - offset)) / 2e-5
        assert float(numeric.detach()) == pytest.approx(float(analytic[0, j]), abs=1e-8)
    actual = endpoints(model, controls).detach().numpy()[0]
    for i, (a, g, b) in enumerate(model["parameters"]):
        angle, velocity = model["angle"], model["velocity"]
        for u in controls.detach().numpy()[0]:
            velocity = np.clip(a * velocity + g * u + b, -.65, .65)
            angle += velocity
        expected = np.array(model["center"]) + model["radius"] * np.array([np.cos(angle), np.sin(angle)])
        np.testing.assert_allclose(actual[i], expected, atol=1e-14, rtol=0)
    assert torch.isfinite(step(model, controls)).all()


def test_actual_new_motion_evidence_invalidates_prior_plan(runtime):
    before = len(runtime.base.session.learner.session.events)
    runtime.advance(0)
    assert runtime.jobs["missing"]["status"] == "stale"
    assert len(runtime.base.session.learner.session.events) == before + 1
    restored = ConstraintRuntime(runtime.checkpoint, runtime.snapshot())
    assert model_identity(restored.owner) == model_identity(runtime.owner)
