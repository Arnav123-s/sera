"""Actual shared-owner acquisition, independent mathematics and lifecycle checks."""

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from sera.contracts import EvidenceKind, Provenance
from sera.event_ir import Event, EventRole
from sera.generative import (
    CLASSES,
    DIMENSIONS,
    SCHEMA,
    GenerativeSharedR1,
    SharedGenerativeSession,
)
from sera.shared import SharedR1, make_shared_solver, replace_shared_owner
from sera.solver import SolverStore
from sera.storage import digest
from sera.world_graph import ExecutionBudget, SharedOwnerRef


def event(position, phase, xy, *, label=False, entity="trajectory", record_id=None):
    return Event(SCHEMA, position, (float(phase), *map(float, xy)),
                 Provenance("independent-fixture", record_id or f"{'label' if label else 'obs'}-{position}",
                            EvidenceKind.VERIFIED if label else EvidenceKind.SYNTHETIC),
                 entity_refs=(entity,), role=EventRole.TARGET if label else EventRole.OBSERVATION)


@pytest.fixture
def solver():
    torch.manual_seed(713)
    base = SharedR1(width=8, heads=2, memory_dim=2)
    return make_shared_solver(GenerativeSharedR1.from_shared(base)).eval()


def independent_design(kind, phases):
    rows = []
    for t in phases:
        c, s = math.cos(math.pi*t), math.sin(math.pi*t)
        if kind == "circle":
            pair = [[1, 0, c, -s], [0, 1, s, c]]
        elif kind == "ellipse":
            pair = [[1, 0, c, s, 0, 0], [0, 1, 0, 0, c, s]]
        elif kind == "line":
            pair = [[1, 0, t, 0], [0, 1, 0, t]]
        else:
            pair = [[1, 0, c, -s, t*c, -t*s], [0, 1, s, c, t*s, t*c]]
        rows.extend(pair)
    return np.asarray(rows)


def test_conversion_owns_all_old_bytes_flags_and_views_without_extra_parameters():
    base = SharedR1(width=8, heads=2, memory_dim=2)
    next(base.parameters()).requires_grad_(False)
    converted = GenerativeSharedR1.from_shared(base)
    assert type(base) is SharedR1
    assert {n: p.requires_grad for n, p in base.named_parameters()} == {
        n: p.requires_grad for n, p in converted.named_parameters()}
    assert {n for n, _ in base.named_parameters()} == {n for n, _ in converted.named_parameters()}
    for name, old in base.state_dict().items():
        new = converted.state_dict()[name]
        assert torch.equal(old, new) and old.dtype == new.dtype
        assert not old.numel() or old.data_ptr() != new.data_ptr()
    combined = make_shared_solver(converted)
    combined.validate()
    assert combined.neural.owner is converted is combined.components["typed"].owner
    assert {id(p) for p in combined.parameters()} == {id(p) for p in converted.parameters()}
    clone = copy.deepcopy(combined)
    clone.validate()
    assert clone.neural.owner is clone.components["r1"] is clone.components["typed"].owner
    assert clone.components["r1"] is not converted


def test_online_posterior_matches_independent_observation_space_gaussian(solver):
    session = SharedGenerativeSession(solver)
    phases = [-.7, -.2, .15, .55, .9]
    ys = np.array([[.2, -.7], [1.1, .3], [.7, 1.2], [-.4, .5], [-.8, -.2]])
    for i, (phase, xy) in enumerate(zip(phases, ys)):
        session.observe(event(i, phase, xy))
    owner = solver.components["r1"]
    evidence = []
    y = ys.ravel()
    for k, (kind, dimension) in enumerate(zip(CLASSES, DIMENSIONS)):
        a = independent_design(kind, phases)
        np.testing.assert_allclose(owner._design(phases)[k, :, :, :dimension].reshape(-1, dimension), a,
                                   rtol=1e-14, atol=1e-14)
        marginal = owner.generator_noise**2 * np.eye(len(y)) + 4*a @ a.T
        mean = 4*a.T @ np.linalg.solve(marginal, y)
        covariance = 4*np.eye(dimension) - 16*a.T @ np.linalg.solve(marginal, a)
        np.testing.assert_allclose(owner.generator_mean[k, :dimension], mean, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(owner.generator_covariance[k, :dimension, :dimension], covariance,
                                   rtol=1e-8, atol=1e-10)
        evidence.append(-.5*(len(y)*math.log(2*math.pi) + np.linalg.slogdet(marginal)[1]
                              + y @ np.linalg.solve(marginal, y)))
    np.testing.assert_allclose(owner.generator_log_evidence, evidence, rtol=1e-9, atol=1e-8)
    weights = np.exp(evidence - np.max(evidence))
    np.testing.assert_allclose(owner.generator_class_probabilities, weights/weights.sum(), atol=1e-10)
    assert session.situation.budget.counts["factual_observations"] == len(phases)


def test_predictions_branches_and_old_routes_retain_the_same_owner(solver):
    session = SharedGenerativeSession(solver)
    inputs = torch.randn(3, 4, 20)
    before = solver.neural(inputs).detach().clone()
    old_bytes = {k: v.clone() for k, v in solver.components["r1"].state_dict().items()
                 if not k.startswith("generator_")}
    for i, t in enumerate([-.5, 0., .5]):
        session.observe(event(i, t, [math.cos(math.pi*t), math.sin(math.pi*t)]))
    assert torch.equal(before, solver.neural(inputs))
    for key, value in old_bytes.items():
        assert torch.equal(value, solver.components["r1"].state_dict()[key])
    identity, factual = solver.identity(), session.snapshot()["sha256"]
    direct = session.predict([.2, .7])
    branch = session.fork()
    predicted = branch.predict([.2, .7])
    for key in direct:
        assert torch.equal(direct[key], predicted[key])
    direct["class_probabilities"].zero_()
    assert solver.components["r1"].generator_class_probabilities.sum() == pytest.approx(1)
    assert solver.identity() == identity and session.snapshot()["sha256"] == factual
    with pytest.raises(ValueError, match="cannot admit"):
        branch.observe(event(3, 1., [0., 1.]))
    assert len(branch.situation.assumptions) == 2


def test_correction_preserves_facts_rebuilds_posterior_and_restores_predecessor(solver):
    session = SharedGenerativeSession(solver)
    first = event(0, 0., [5., 2.])
    session.observe(first)
    session.observe(event(1, .5, [0., 1.]))
    before, prior_snapshot = solver.identity(), session.snapshot()["sha256"]
    corrected = session.correct(event(0, 0., [1., 0.], label=True))
    assert session.situation.observations[0] == first
    assert len(session.situation.labels) == 1 and solver.identity() != before
    assert corrected.record["parent_solver_sha256"] == before
    prior_solver, prior_session = corrected.predecessor.restore()
    assert prior_solver.identity() == before and prior_session.snapshot()["sha256"] == prior_snapshot
    fresh = make_shared_solver(GenerativeSharedR1(width=8, heads=2, memory_dim=2))
    reference = SharedGenerativeSession(fresh)
    reference.observe(event(0, 0., [1., 0.]))
    reference.observe(event(1, .5, [0., 1.]))
    for name in ("gram", "rhs", "square_sum", "mean", "covariance", "class_probabilities"):
        assert torch.equal(getattr(solver.components["r1"], "generator_"+name),
                           getattr(fresh.components["r1"], "generator_"+name))
    assert session.situation.budget.counts["factual_observations"] == 2
    assert session.situation.budget.counts["verified_labels"] == 1


def test_full_store_restore_reconnects_and_resumes_acquired_state(solver, tmp_path):
    next(solver.parameters()).requires_grad_(False)
    session = SharedGenerativeSession(solver)
    session.observe(event(0, -.25, [.5, -.6]))
    session.observe(event(1, .25, [.5, .6]))
    record = session.snapshot()
    store = SolverStore(tmp_path / "solver")
    store.initialize(solver)
    restored = store.load()
    assert restored.identity() == solver.identity()
    assert {n: p.requires_grad for n, p in restored.named_parameters()} == {
        n: p.requires_grad for n, p in solver.named_parameters()}
    assert restored.neural.owner is restored.components["typed"].owner is restored.components["r1"]
    resumed = SharedGenerativeSession.restore(restored, json.loads(json.dumps(record)))
    for key, value in session.predict([.1]).items():
        assert torch.equal(value, resumed.predict([.1])[key])
    resumed.observe(event(2, .75, [-.5, .6]))
    assert len(session.situation.observations) == 2 and len(resumed.situation.observations) == 3
    with pytest.raises(ValueError, match="factual snapshot"):
        SharedGenerativeSession(restored)


@pytest.mark.parametrize("change", ["prediction", "entity", "mask", "phase", "duplicate"])
def test_ineligible_or_inconsistent_evidence_never_changes_owner(solver, change):
    session = SharedGenerativeSession(solver)
    first = event(0, 0., [1., 0.])
    session.observe(first)
    proposal = event(1, .5, [0., 1.])
    if change == "prediction":
        proposal = replace(proposal, provenance=Provenance("model", "guess", EvidenceKind.PREDICTION))
    elif change == "entity":
        proposal = replace(proposal, entity_refs=("different",))
    elif change == "mask":
        proposal = replace(proposal, available=(True, True, False))
    elif change == "phase":
        proposal = event(0, .1, [1., 0.], label=True)
    else:
        proposal = first
    identity = solver.identity()
    with pytest.raises(ValueError):
        (session.correct if change == "phase" else session.observe)(proposal)
    assert solver.identity() == identity and len(session.situation.observations) == 1


def test_rejection_budget_exhaustion_and_stale_references_are_transactional(solver):
    session = SharedGenerativeSession(solver)
    session.observe(event(0, 0., [1., 0.]))
    old_branch = session.fork()
    old_situation = session.situation
    identity = solver.identity()
    with pytest.raises(ValueError, match="rejected"):
        session.observe(event(1, .5, [0., 1.]), validator=lambda *_: False)
    assert solver.identity() == identity and session.situation is old_situation
    assert session.situation.budget.counts["factual_observations"] == 1
    session.situation.budget = ExecutionBudget(6)  # reaches replay after copy + visit + four solves
    with pytest.raises(InterruptedError):
        session.observe(event(1, .5, [0., 1.]))
    assert solver.identity() == identity and session.situation is old_situation
    session.situation.budget = ExecutionBudget()
    session.observe(event(1, .5, [0., 1.]))
    with pytest.raises(ValueError):
        old_branch.predict([0.])
    with pytest.raises(ValueError):
        old_situation.snapshot()
    owner = copy.deepcopy(solver.components["r1"])
    replace_shared_owner(solver, owner)
    with pytest.raises(ValueError):
        session.predict([0.])


def test_validator_mutation_is_rejected_before_commit(solver):
    session = SharedGenerativeSession(solver)
    identity = solver.identity()

    def mutate(candidate, _):
        with torch.no_grad():
            next(candidate.parameters()).add_(1)
        return True

    with pytest.raises(ValueError, match="mutated"):
        session.observe(event(0, 0., [1., 0.]), validator=mutate)
    assert solver.identity() == identity and session.situation.observations == ()


def test_restore_checks_numerical_evidence_not_only_self_consistent_hashes(solver):
    session = SharedGenerativeSession(solver)
    session.observe(event(0, 0., [1., 0.]))
    record = session.snapshot()
    owner = solver.components["r1"]
    owner.generator_rhs.mul_(2)
    owner._refresh_posterior()  # internally self-consistent, but no longer explains recorded evidence
    record["payload"]["mechanism"] = session.situation.mechanism.bind(
        owner.generator_graph(), SharedOwnerRef.from_solver(solver)).record()
    record["sha256"] = digest(record["payload"])
    with pytest.raises(ValueError, match="sufficient statistics"):
        SharedGenerativeSession.restore(solver, record)


def test_silent_generator_precision_change_is_rejected(solver):
    solver.float()
    with pytest.raises(ValueError, match="float64"):
        solver.validate()
