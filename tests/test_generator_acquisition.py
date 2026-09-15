"""Independent likelihood/design controls and provenance/replay regressions for GG-ACT."""

import copy
import hashlib
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.generative_memory import acquisition as act
from experiments.generative_memory.core import CIRCLE, LINE, canonical, design, digest
from sera.contracts import EvidenceKind, Provenance
from sera.event_ir import EventRole
from sera.shared import SharedR1, make_shared_solver
from sera.world_graph import ExecutionBudget, LiveSituation, SharedOwnerRef


@pytest.fixture(autouse=True)
def one_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_sequential_joint_class_and_coefficient_posterior_matches_batch_gaussian():
    actions = (act.Action(0, -.7, "target", .04), act.Action(1, -.1, "target", .09),
               act.Action(2, .2, "nuisance", 400.), act.Action(3, .8, "target", .01))
    observations = np.array([[1.2, -.4], [.3, .7], [35., -.2], [1.1, 1.8]])
    masses = np.array([1., 2., 3., 4.])
    posterior = act.Posterior(actions, max_observations=4, class_priors=masses)
    assert masses.tolist() == [1., 2., 3., 4.]  # caller-owned prior is not mutated
    for i, (action, y) in enumerate(zip(actions, observations)):
        posterior.observe(act.observed_event(action, y, i, f"event-{i}"))
    record = posterior.snapshot()["payload"]
    log_mass = []
    noise = np.diag(np.repeat([a.variance for a in actions], 2))
    y = observations.ravel()
    for spec, actual in zip(act.SPECS, record["models"]):
        matrix = np.vstack([design(spec, np.array([a.t])) * (a.channel == "target")
                            for a in actions])
        prior = np.eye(matrix.shape[1]) * 4
        covariance = noise + matrix @ prior @ matrix.T
        mean = prior @ matrix.T @ np.linalg.solve(covariance, y)
        conditional = prior - prior @ matrix.T @ np.linalg.solve(covariance, matrix @ prior)
        np.testing.assert_allclose(actual["mean"], mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual["covariance"], conditional, rtol=1e-9, atol=1e-10)
        log_mass.append(-.5 * (len(y) * math.log(2 * math.pi)
                              + np.linalg.slogdet(covariance)[1]
                              + y @ np.linalg.solve(covariance, y)))
    log_mass = np.asarray(log_mass) + np.log(masses / masses.sum())
    expected = np.exp(log_mass - np.max(log_mass))
    expected /= expected.sum()
    np.testing.assert_allclose(posterior.weights, expected, rtol=1e-10, atol=1e-10)
    assert act.Posterior.restore(posterior.snapshot()).snapshot() == posterior.snapshot()


def test_nuisance_is_high_entropy_but_zero_information_and_consumes_budget():
    actions = act.permitted_grid(points=5)
    posterior = act.Posterior(actions, max_observations=2)
    nuisance = next(a for a in actions if a.channel == "nuisance")
    before = posterior.snapshot()["payload"]
    score = act.acquisition_scores(posterior, nuisance, order=5)
    assert score["information_gain"] == score["class_information_gain"] == score["parameter_information_gain"] == 0
    assert score["predictive_entropy"] == pytest.approx(math.log(2 * math.pi * math.e * nuisance.variance))
    entropy_choice, _ = act.choose(posterior, "predictive_entropy", np.random.default_rng(1), quadrature_order=5)
    useful_choice, _ = act.choose(posterior, "information_gain", np.random.default_rng(1), quadrature_order=5)
    assert entropy_choice.channel == "nuisance" and useful_choice.channel == "target"
    result = posterior.observe(act.observed_event(nuisance, [200., -400.], 0, "nuisance"))
    after = posterior.snapshot()["payload"]
    assert before["models"] == after["models"]
    assert before["log_class_weights"] == after["log_class_weights"]
    assert posterior.observations_used == 1 and after["observation_cost"] == 1
    assert not result["all_models_inadequate_alarm"]
    assert len(set(result["pre_observation_log_likelihoods"])) == 1


def test_single_class_information_is_parameter_information_not_class_disagreement():
    actions = act.permitted_grid(points=3)
    posterior = act.Posterior(actions, specs=(LINE,), max_observations=2)
    action = actions[0]
    score = act.acquisition_scores(posterior, action, order=3)
    a = design(LINE, np.array([action.t]))
    expected = .5 * np.linalg.slogdet(np.eye(2) + a @ (4 * np.eye(a.shape[1])) @ a.T / action.variance)[1]
    assert score["information_gain"] == pytest.approx(expected, abs=1e-12)
    assert score["parameter_information_gain"] == pytest.approx(expected, abs=1e-12)
    assert score["class_information_gain"] == pytest.approx(0, abs=1e-12)
    assert score["disagreement"] == 0


def test_shared_class_joint_density_does_not_factor_into_marginal_mixtures():
    prediction = {"component_means": np.array([[2., 2.], [-2., -2.]]),
                  "component_covariances": np.array([np.eye(2) * .1, np.eye(2) * .1]),
                  "log_weights": np.log([.5, .5])}
    y = np.array([2., -2.])
    joint = act._mixture_logp(y, prediction)
    marginal_density = .5 / math.sqrt(2 * math.pi * .1) * (1 + math.exp(-16 / .2))
    independent_marginal_logp = 2 * math.log(marginal_density)
    assert independent_marginal_logp - joint > 70


def test_tiny_prior_class_can_recover_in_log_space_and_menu_alarm_is_separate():
    actions = act.permitted_grid(points=5, noise=.001)
    posterior = act.Posterior(actions, specs=(CIRCLE, LINE), class_priors=[1., 1e-300], max_observations=5)
    for i, action in enumerate(actions[:4]):
        posterior.observe(act.observed_event(action, [3 * action.t, 0.], i, f"e-{i}"))
    assert posterior.weights[1] > .999
    diagnostic = posterior.observe(act.observed_event(actions[4], [1e5, -1e5], 4, "shock"))
    assert diagnostic["all_models_inadequate_alarm"]
    assert sum(posterior.weights) == pytest.approx(1)
    assert max(posterior.weights) > .99  # confidence within the menu does not establish adequacy


def test_observation_contract_rejects_imagination_duplicates_spoofing_and_overbudget():
    actions = act.permitted_grid(points=3)
    posterior = act.Posterior(actions, max_observations=2)
    event = act.observed_event(actions[0], [1., 2.], 0, "first")
    imagined = replace(event, role=EventRole.HYPOTHETICAL,
                       provenance=Provenance("model", "imagined", EvidenceKind.PREDICTION))
    before = posterior.snapshot()
    branch = act.hypothetical_prediction(posterior, actions[0])
    assert not branch["learning_eligible"] and branch["observation_count"] == 0
    assert posterior.snapshot() == before
    for invalid in (imagined, replace(event, sequence=1),
                    replace(event, values=(99, *event.values[1:])),
                    replace(event, values=(event.values[0], event.values[1], .9, *event.values[3:]))):
        with pytest.raises(ValueError):
            posterior.observe(invalid)
        assert posterior.snapshot() == before
    posterior.observe(event)
    with pytest.raises(ValueError, match="Duplicate"):
        posterior.observe(replace(event, sequence=1))
    posterior.observe(act.observed_event(actions[1], [1., 2.], 1, "second"))
    with pytest.raises(ValueError, match="budget"):
        posterior.observe(act.observed_event(actions[2], [1., 2.], 2, "third"))


def test_snapshot_replay_rejects_forged_posterior_even_with_recomputed_digest():
    posterior = act.Posterior(act.permitted_grid(points=3), max_observations=2)
    posterior.observe(act.observed_event(posterior.actions[0], [1., 2.], 0, "observed"))
    frozen = posterior.snapshot()
    assert act.Posterior.restore(copy.deepcopy(frozen)).snapshot() == frozen
    broken = copy.deepcopy(frozen)
    broken["payload"]["models"][0]["mean"][0] += 1
    broken["sha256"] = digest(broken["payload"])
    with pytest.raises(ValueError, match="replay"):
        act.Posterior.restore(broken)


def test_disagreement_control_does_not_pay_for_entropy_quadrature(monkeypatch):
    posterior = act.Posterior(act.permitted_grid(points=3), max_observations=2)
    monkeypatch.setattr(act, "acquisition_scores", lambda *_: pytest.fail("unneeded entropy integration"))
    action, decision = act.choose(posterior, "disagreement", np.random.default_rng(1))
    assert action.index in [a.index for a in posterior.available]
    assert decision["density_evaluations"] == 0


def test_external_mixture_intervals_and_joint_nll_reduce_to_known_gaussian():
    covariance = np.array([[1., .4], [.4, 2.]])
    prediction = {"mean": [0., 0.], "covariance": covariance.tolist(),
                  "component_means": [[0., 0.]], "component_covariances": [covariance.tolist()],
                  "log_weights": [0.], "class_weights": [1.]}
    observed = np.array([[0., 0.], [3., 0.], [0., 3.], [1., 2.], [-1., -2.]])
    world = {"query_t": [-2., -1., 0., 1., 2.], "query_truth": np.zeros((5, 2)).tolist(),
             "query_observed": observed.tolist()}
    actions = act.permitted_grid(points=3)
    result = act._query_metrics([prediction] * 5, world, actions, [0, 1])
    expected_bound = 1.959963984540054 * np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(result["marginal_95_upper"], np.tile(expected_bound, (5, 1)), atol=1e-11)
    np.testing.assert_allclose(result["marginal_95_lower"], np.tile(-expected_bound, (5, 1)), atol=1e-11)
    expected_nll = [.5 * (2 * math.log(2 * math.pi) + np.linalg.slogdet(covariance)[1]
                         + y @ np.linalg.solve(covariance, y)) for y in observed]
    np.testing.assert_allclose(result["per_query_observed_joint_nll"], expected_nll, atol=1e-12)
    assert result["regions"]["all"]["marginal_95_coverage"] == .8


def test_omitted_exception_is_invisible_outside_its_assessor_only_window():
    query = [-1., -.4, 0., .64, 1., 1.8]
    world = act._assessor(21, "test-private", "menu_exception", act.permitted_grid(), query, 4.)
    base = (design(CIRCLE, np.asarray(query)) @ world["private_coefficients"]).reshape(-1, 2)
    actual = np.asarray(world["query_truth"])
    np.testing.assert_array_equal(actual[[0, 1, 2, 4, 5]], base[[0, 1, 2, 4, 5]])
    assert np.linalg.norm(actual[3] - base[3]) > 1


def test_owner_adapter_uses_existing_owner_and_restores_explicit_side_state():
    torch.manual_seed(7)
    owner = SharedR1(width=8, heads=2, memory_dim=2)
    solver = make_shared_solver(owner)
    reference = SharedOwnerRef.from_solver(solver)
    posterior = act.Posterior(act.permitted_grid(points=3), max_observations=2)
    graph = act.graph_for(posterior)
    live = LiveSituation("one-world", graph, reference, budget=ExecutionBudget(100_000))
    adapter = act.OwnerBoundAcquisition(live, posterior)
    parameter_ids = {id(p) for p in solver.parameters()}
    before = live.snapshot()["sha256"]
    branch = adapter.branch(posterior.actions[0])
    assert branch["shared_owner_mechanism"] == live.mechanism.record()
    assert live.snapshot()["sha256"] == before and len(live.observations) == 0
    adapter.observe(act.observed_event(posterior.actions[0], [1., 2.], 0, "event"))
    assert adapter.situation.owner.owner is owner and list(reference.parameters()) == []
    assert {id(p) for p in solver.parameters()} == parameter_ids
    frozen = adapter.snapshot()
    restored_live = LiveSituation.restore(frozen["payload"]["situation"], graph, reference,
                                          situation_id="one-world")
    restored = act.OwnerBoundAcquisition.restore(frozen, restored_live)
    assert restored.situation.owner.owner is owner
    assert restored.posterior.snapshot() == adapter.posterior.snapshot()
    assert restored_live.budget.counts["replayed_observations"] == 1
    next(owner.parameters()).data.add_(.125)
    with pytest.raises(ValueError, match="parameters"):
        adapter.choose("random", np.random.default_rng(1))


def test_owner_adapter_rejects_same_bytes_different_owner_object():
    owner = SharedR1(width=8, heads=2, memory_dim=2)
    posterior = act.Posterior(act.permitted_grid(points=3), max_observations=2)
    live = LiveSituation("one", act.graph_for(posterior), SharedOwnerRef(owner))
    adapter = act.OwnerBoundAcquisition(live, posterior)
    clone = copy.deepcopy(owner)
    live.owner = SharedOwnerRef(clone)
    with pytest.raises(ValueError, match="another parameter owner"):
        adapter.require_current()


def tiny_protocol():
    protocol = {"experiment_id": "GG-ACT-001/test", "source_hashes": act.source_hashes(),
                "local_cap_seconds": 20, "policies": list(act.POLICIES),
                "prior_variance": 4., "alarm_alpha": .001}
    for i, phase in enumerate(("plumbing", "development", "final")):
        protocol[phase] = {"environment_seeds": [100 + i], "families": ["circle"],
                           "grid": {"points": 5, "low": -1.5, "high": 1.5,
                                    "noise": .05, "nuisance_noise": 20.},
                           "query_count": 7, "observation_budget": 3,
                           "initial_indices": [1], "policy_seeds": [811 + i],
                           "quadrature_order": 3}
    return protocol


def test_tiny_study_replays_all_scores_posteriors_and_external_vectors(tmp_path):
    output = tmp_path / "run"
    protocol = tiny_protocol()
    manifest = act.run_study(protocol, output)
    assert manifest["status"] == "complete" and len(manifest["records"]) == 5
    records = [json.loads((output / r["path"]).read_text()) for r in manifest["records"]]
    for record in records:
        assert record["observation_count"] == 3
        assert record["trace"][0]["action"] == records[0]["trace"][0]["action"]
        for step in record["trace"]:
            assert step["observed_event"]["provenance"]["record_id"] == f"action-{step['action']['index']}"
            assert set(step["posterior"]["payload"]["contract"]) >= {"actions", "specs", "source_hashes"}
            assert len(step["query_metrics"]["per_query_observed_joint_nll"]) == 7
    replay = act.replay_study(output)
    assert replay["status"] == "passed" and replay["observation_steps_replayed"] == 15
    path = output / manifest["records"][0]["path"]
    broken = records[0]
    broken["trace"][0]["query_prediction"][0]["mean"][0] += .1
    path.write_text(canonical(broken) + "\n", encoding="utf-8")
    info = manifest["artifact_files"][manifest["records"][0]["path"]]
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info["bytes"] = path.stat().st_size
    (output / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="External raw prediction"):
        act.replay_study(output)


def test_protocol_seed_overlap_and_stale_source_fail_before_output_creation(tmp_path):
    protocol = tiny_protocol()
    protocol["final"]["environment_seeds"] = protocol["development"]["environment_seeds"]
    with pytest.raises(ValueError, match="seed namespaces"):
        act.run_study(protocol, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    protocol = tiny_protocol()
    protocol["source_hashes"]["experiments/generative_memory/acquisition.py"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen source"):
        act.run_study(protocol, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()
