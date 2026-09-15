"""Mechanism checks for prospective conditional distillation."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.cross_route_transfer.data import (
    admit_training,
    circle_cases,
    partition_audit,
    state_digest,
    teacher_identity,
    typed_motion,
)
from experiments.cross_route_transfer.learning import configure, fit, motion_loss
from experiments.cross_route_transfer.study import EVIDENCE, parent
from sera.shared import replace_shared_owner
from sera.shared_learning import SharedEvidence
from sera.world_graph import SharedOwnerRef


@pytest.fixture(scope="module")
def solver():
    torch.set_num_threads(1)
    return parent()


def test_conditional_teaching_is_not_observed_evidence(solver):
    owner = solver.components["r1"]
    before = state_digest(owner)
    cases = circle_cases(owner, seed=7201, count=16, split="support-condition-test", conditional=True)
    admit_training(cases, allow_conditional=True, teacher=teacher_identity(owner))
    with pytest.raises(ValueError, match="Unbound"):
        admit_training(cases, allow_conditional=False, teacher=teacher_identity(owner))
    with pytest.raises(ValueError, match="Unbound"):
        admit_training(cases, allow_conditional=True, teacher="wrong")
    with pytest.raises(ValueError, match="not independently"):
        typed_motion(cases[0])
    with pytest.raises(ValueError, match="cannot enter"):
        admit_training([replace(cases[0], split="evaluation-motion")], allow_conditional=True,
                       teacher=teacher_identity(owner))
    assert state_digest(owner) == before


def test_teacher_execution_matches_independent_declared_circle(solver):
    owner = solver.components["r1"]
    generated = circle_cases(owner, seed=771, count=17, split="support-generated", conditional=True)
    reference = circle_cases(owner, seed=771, count=17, split="support-reference")
    assert np.max(np.abs(np.asarray([r.target for r in generated])-np.asarray([r.target for r in reference]))) < 1e-12
    assert all(r.teacher is not None for r in generated)
    with pytest.raises(ValueError, match="Repeated"):
        partition_audit({"first": reference, "second": reference})


def test_shared_gradient_and_parent_independence(solver):
    original = solver.identity()
    descendant = copy.deepcopy(solver)
    owner = descendant.components["r1"]
    state_before = state_digest(owner)
    old_owner_identity = SharedOwnerRef.from_solver(descendant).identity
    configure(owner, "shared")
    rows = circle_cases(owner, seed=823, count=8, split="support-gradients", conditional=True)
    loss = motion_loss(owner, rows)
    loss.backward()
    active = {name for name, p in owner.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    for prefix in ("memory.", "fusion.", "typed_encoder.adapters.numeric.", "typed_numeric."):
        assert any(n.startswith(prefix) for n in active), prefix
    optimizer = torch.optim.SGD([p for p in owner.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    assert state_digest(owner) != state_before
    replace_shared_owner(descendant, owner)
    assert descendant.neural.owner is descendant.components["typed"].owner is owner
    assert old_owner_identity != SharedOwnerRef.from_solver(descendant).identity
    assert solver.identity() == original
    assert all(torch.equal(value, owner.state_dict()[name]) for name, value in solver.components["r1"].state_dict().items()
               if name.startswith("generator_"))


def test_readout_scope_really_freezes_shared_memory(solver):
    owner = copy.deepcopy(solver.components["r1"])
    parameters = configure(owner, "readout")
    assert len(parameters) == 2
    assert all(name.startswith("typed_numeric.") for name, p in owner.named_parameters() if p.requires_grad)


def test_interrupted_optimizer_continues_exactly(solver):
    owner = solver.components["r1"]
    cfg = {"scope": "readout", "steps": 4, "select_every": 2, "batch_size": 8,
           "replay_batch": 8, "learning_rate": .0003, "replay_weight": 1.0,
           "anchor_weight": 2.0, "anchor_batch": 8}
    observed = circle_cases(owner, seed=926, count=8, split="support-resume")
    dreams = circle_cases(owner, seed=927, count=8, split="support-resume-dream", conditional=True)
    oracle = circle_cases(owner, seed=927, count=8, split="support-resume-oracle")
    dev = circle_cases(owner, seed=928, count=8, split="development-resume")
    evidence = SharedEvidence.load(EVIDENCE)
    checkpoints = []

    def callback(step, model, optimizer, best_state, history, draw_rng, replay_rng, anchor_rng, work):
        original = owner.state_dict()
        def delta(state):
            return {n: v.detach().clone() for n, v in state.items() if not torch.equal(v, original[n])}
        checkpoints.append(copy.deepcopy({"step": step, "model_delta": delta(model.state_dict()),
                                         "best_delta": delta(best_state), "optimizer": optimizer.state_dict(),
                                         "history": history, "draw_rng": draw_rng, "replay_rng": replay_rng,
                                         "anchor_rng": anchor_rng, "torch_rng": torch.get_rng_state(),
                                         "work": work, "training_config": cfg, "arm": "integrated", "seed": 93}))
    full, report = fit(owner, evidence, observed, dreams, oracle, dev, cfg, "integrated", 93, callback)
    resumed, continued = fit(owner, evidence, observed, dreams, oracle, dev, cfg, "integrated", 93,
                             resume=checkpoints[0])
    assert state_digest(full) == state_digest(resumed)
    assert report["history"] == continued["history"]
    assert report["work"] == continued["work"]
    with pytest.raises(ValueError, match="completed"):
        fit(owner, evidence, observed, dreams, oracle, dev, cfg, "integrated", 93, resume=checkpoints[-1])
