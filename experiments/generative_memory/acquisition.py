"""GG-ACT-001: fixed-policy inquiry into one partially observed supplied-phase world.

The learner sees only the public experiment contract and observed Event records.
Assessor-only functions are at the end; their world coefficients/seeds never enter
Posterior or choose(). Gaussian class/parameter updates are conjugate. Mixture
entropy and information gain use finite Gauss-Hermite quadrature, not exact EIG.
The optional owner adapter retains this analytic posterior as explicit side state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sera.contracts import EvidenceKind, Provenance
from sera.event_ir import Event, EventRole, EventSchema

from .core import CIRCLE, ELLIPSE, LINE, canonical, design, digest

POLICIES = ("random", "space_filling", "predictive_entropy", "disagreement", "information_gain")
SCHEMA = EventSchema("gg-act-observed-action-and-pair", 1, "numeric", 5, "dimensionless")
SPECS = (CIRCLE, ELLIPSE, LINE,
         {"kind": "radial_orbit", "name": "linear-radius-orbit", "frequency": 1})


def source_hashes():
    folder = Path(__file__).parent
    return {f"experiments/generative_memory/{name}":
            hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("core.py", "acquisition.py")}


def owner_seam_hashes():
    """Pin the explicit common-owner seam separately from the analytic decoder."""
    folder = Path(__file__).resolve().parents[2] / "src" / "sera"
    return {f"src/sera/{name}": hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("contracts.py", "event_ir.py", "world_graph.py", "session_state.py",
                         "shared.py", "storage.py")}


@dataclass(frozen=True)
class Action:
    index: int
    t: float
    channel: str
    variance: float
    cost: float = 1.0

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Action indices must be nonnegative integers")
        if self.channel not in {"target", "nuisance"}:
            raise ValueError("Unknown measurement channel")
        if not math.isfinite(self.t) or not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("Finite coordinates and positive observation variance required")
        if self.cost != 1:
            raise ValueError("GG-ACT-001 uses one shared unit observation cost per action")

    def record(self):
        return asdict(self)


def permitted_grid(points=15, low=-1.5, high=1.5, noise=.05, nuisance_noise=20.):
    if type(points) is not int or not 2 <= points <= 256 or not low < high:
        raise ValueError("Invalid finite experiment grid")
    if noise <= 0 or nuisance_noise <= 0:
        raise ValueError("Noise standard deviations must be positive")
    t = np.linspace(low, high, points)
    return tuple(Action(i + offset, float(x), channel, sigma ** 2)
                 for channel, sigma, offset in
                 (("target", noise, 0), ("nuisance", nuisance_noise, points))
                 for i, x in enumerate(t))


def observed_event(action, value, sequence, record_id, source="GG-ACT-001/assessor"):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("One finite observed pair is required")
    return Event(SCHEMA, sequence,
                 (action.index, action.t, action.variance, *value.tolist()),
                 Provenance(source, record_id, EvidenceKind.SYNTHETIC))


def _logsumexp(values, axis=None):
    largest = np.max(values, axis=axis, keepdims=True)
    result = largest + np.log(np.exp(values - largest).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else float(result.item())


def _normal_logp(values, mean, covariance):
    residual = np.asarray(values) - mean
    chol = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(chol, residual.reshape(-1, 2).T).T
    q = np.sum(standardized ** 2, axis=1)
    result = -.5 * (2 * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + q)
    return result.reshape(residual.shape[:-1])


def _mixture_logp(values, prediction):
    terms = np.asarray([
        logw + _normal_logp(values, mean, covariance)
        for logw, mean, covariance in zip(prediction["log_weights"],
                                        prediction["component_means"],
                                        prediction["component_covariances"])
    ])
    return _logsumexp(terms, axis=0)


def _normal_entropy(covariance):
    return math.log(2 * math.pi * math.e) + .5 * np.linalg.slogdet(covariance)[1]


class Posterior:
    """Finite model-class mixture with a Gaussian coefficient posterior per class."""

    def __init__(self, actions, *, specs=SPECS, prior_variance=4., class_priors=None,
                 max_observations=10, alarm_alpha=.001):
        actions = tuple(actions)
        if (not actions or len(actions) > 512 or any(type(a) is not Action for a in actions)
                or len({a.index for a in actions}) != len(actions)):
            raise ValueError("Unique finite permitted actions are required")
        if type(max_observations) is not int or not 1 <= max_observations <= len(actions):
            raise ValueError("Invalid observation budget")
        if not math.isfinite(prior_variance) or prior_variance <= 0 or not 0 < alarm_alpha < 1:
            raise ValueError("Invalid Gaussian prior or diagnostic threshold")
        specs = json.loads(canonical(list(specs)))
        if not specs or len(specs) > 32 or len({s["name"] for s in specs}) != len(specs):
            raise ValueError("A bounded menu of uniquely named generators is required")
        matrices = [design(spec, np.array([a.t for a in actions])) for spec in specs]
        if any(not np.isfinite(a).all() or not 1 <= a.shape[1] <= 32 for a in matrices):
            raise ValueError("Invalid or excessive supplied generator design")
        weights = (np.ones(len(specs)) if class_priors is None
                   else np.array(class_priors, dtype=float, copy=True))
        if weights.shape != (len(specs),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Positive class priors must match the menu")
        prior_masses = weights.tolist()
        log_weights = np.log(weights)
        log_weights -= _logsumexp(log_weights)
        weights = np.exp(log_weights)
        self.actions = actions
        self._contract = {
            "schema": SCHEMA.record(), "actions": [a.record() for a in actions],
            "specs": specs, "prior_variance": float(prior_variance),
            "class_priors": weights.tolist(), "class_prior_masses": prior_masses,
            "max_observations": max_observations,
            "alarm_alpha": float(alarm_alpha), "source_hashes": source_hashes(),
            "precision": "float64 analytic posterior",
        }
        self._means = [np.zeros(a.shape[1]) for a in matrices]
        self._covariances = [np.eye(a.shape[1]) * prior_variance for a in matrices]
        self._log_weights = log_weights
        self._events, self._diagnostics = [], []
        self._cost = 0.

    @property
    def contract(self):
        return json.loads(canonical(self._contract))

    @property
    def weights(self):
        return np.exp(self._log_weights.copy())

    @property
    def observed_indices(self):
        return tuple(int(row["values"][0]) for row in self._events)

    @property
    def available(self):
        used = set(self.observed_indices)
        return tuple(a for a in self.actions if a.index not in used)

    @property
    def observations_used(self):
        return len(self._events)

    def action(self, index):
        for action in self.actions:
            if action.index == index:
                return action
        raise ValueError("Observation or request is outside the permitted grid")

    def _matrix(self, spec, action):
        a = design(spec, np.array([action.t]))
        return np.zeros_like(a) if action.channel == "nuisance" else a

    def predict(self, action):
        if type(action) is not Action or self.action(action.index) != action:
            raise ValueError("Prediction requires the declared action contract")
        means, covariances = [], []
        for spec, mean, covariance in zip(self._contract["specs"],
                                          self._means, self._covariances):
            a = self._matrix(spec, action)
            means.append(a @ mean)
            covariances.append(a @ covariance @ a.T + action.variance * np.eye(2))
        means, covariances = np.asarray(means), np.asarray(covariances)
        mean = np.einsum("k,kd->d", self.weights, means)
        between = means - mean
        covariance = (np.einsum("k,kij->ij", self.weights, covariances)
                      + np.einsum("k,ki,kj->ij", self.weights, between, between))
        return {"mean": mean, "covariance": covariance, "component_means": means,
                "component_covariances": covariances, "log_weights": self._log_weights.copy(),
                "class_weights": self.weights}

    def validate_event(self, event):
        if type(event) is not Event or event.schema != SCHEMA or not event.eligible_observation:
            raise ValueError("Only eligible observed action/pair events may update the posterior")
        if not all(event.available) or event.sequence != self.observations_used:
            raise ValueError("Evidence must be complete and sequential")
        if self.observations_used >= self._contract["max_observations"]:
            raise ValueError("Observation budget exhausted")
        if event.values[0] != int(event.values[0]):
            raise ValueError("Action index must be integral")
        action = self.action(int(event.values[0]))
        if event.values[1:3] != (action.t, action.variance):
            raise ValueError("Observed coordinate or noise contract differs")
        if (action.index in self.observed_indices
                or event.evidence_key in {(r["provenance"]["source"], r["provenance"]["record_id"])
                                         for r in self._events}):
            raise ValueError("Duplicate observation/action")
        return action

    def observe(self, event):
        action = self.validate_event(event)
        prediction, y = self.predict(action), np.asarray(event.values[3:])
        log_likelihood, tails, means, covariances = [], [], [], []
        for spec, mean, covariance, mu, predictive_cov in zip(
                self._contract["specs"], self._means, self._covariances,
                prediction["component_means"], prediction["component_covariances"]):
            residual = y - mu
            q = float(residual @ np.linalg.solve(predictive_cov, residual))
            tails.append(float(math.exp(-.5 * q)))  # chi-square survival, df=2
            log_likelihood.append(float(_normal_logp(y, mu, predictive_cov)))
            if action.channel == "nuisance":
                means.append(mean.copy())
                covariances.append(covariance.copy())
                continue
            a = self._matrix(spec, action)
            gain = np.linalg.solve(predictive_cov, a @ covariance).T
            means.append(mean + gain @ residual)
            remaining = np.eye(len(mean)) - gain @ a
            updated = remaining @ covariance @ remaining.T + action.variance * gain @ gain.T
            covariances.append((updated + updated.T) / 2)
        log_weights = self._log_weights.copy()
        if action.channel == "target":
            log_weights += log_likelihood
            log_weights -= _logsumexp(log_weights)
        if (not np.isfinite(log_weights).all() or any(not np.isfinite(x).all()
                for x in means + covariances + [np.asarray(log_likelihood)])):
            raise ValueError("Nonfinite posterior update; no evidence admitted")
        diagnostic = {
            "action_index": action.index, "channel": action.channel,
            "pre_observation_log_likelihoods": log_likelihood,
            "pre_observation_component_tail_probabilities_df2": tails,
            "pre_observation_mixture_logp": float(_mixture_logp(y, prediction)),
            "all_models_inadequate_alarm": bool(action.channel == "target"
                                               and max(tails) < self._contract["alarm_alpha"]),
            "interpretation": "Prequential all-component residual diagnostic; not P(outside menu). "
                              "Threshold is uncorrected across repeated adaptive tests.",
        }
        # Commit only after all numerical calculations succeeded.
        self._means, self._covariances, self._log_weights = means, covariances, log_weights
        self._events.append(event.record())
        self._diagnostics.append(diagnostic)
        self._cost += action.cost
        return json.loads(canonical(diagnostic))

    def snapshot(self):
        payload = {
            "format": "sera-gg-acquisition-posterior-v1", "contract": self.contract,
            "models": [{"mean": m.tolist(), "covariance": p.tolist()}
                       for m, p in zip(self._means, self._covariances)],
            "log_class_weights": self._log_weights.tolist(), "class_weights": self.weights.tolist(),
            "observed_events": self._events, "diagnostics": self._diagnostics,
            "observation_cost": self._cost,
        }
        payload = json.loads(canonical(payload))
        return {"payload": payload, "sha256": digest(payload)}

    @classmethod
    def restore(cls, record):
        if (type(record) is not dict or set(record) != {"payload", "sha256"}
                or digest(record["payload"]) != record["sha256"]):
            raise ValueError("Posterior snapshot integrity failure")
        payload, contract = record["payload"], record["payload"]["contract"]
        if contract["source_hashes"] != source_hashes():
            raise ValueError("Posterior interpreter sources changed")
        result = cls([Action(**a) for a in contract["actions"]], specs=contract["specs"],
                     prior_variance=contract["prior_variance"], class_priors=contract["class_prior_masses"],
                     max_observations=contract["max_observations"], alarm_alpha=contract["alarm_alpha"])
        for row in payload["observed_events"]:
            result.observe(Event.from_record(row))
        if result.snapshot() != record:
            raise ValueError("Posterior payload is inconsistent with observed-event replay")
        return result


@lru_cache(maxsize=30)
def _quadrature(order):
    if type(order) is not int or not 2 <= order <= 31:
        raise ValueError("Quadrature order must be between 2 and 31")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    grid = np.asarray(np.meshgrid(nodes, nodes, indexing="ij")).reshape(2, -1).T * math.sqrt(2)
    quadrature_weights = np.outer(weights, weights).ravel() / math.pi
    grid.flags.writeable = quadrature_weights.flags.writeable = False
    return grid, quadrature_weights


def acquisition_scores(posterior, action, order=9):
    """Joint mixture entropy quadrature; includes within-class parameter information."""
    grid, quadrature_weights = _quadrature(order)
    prediction = posterior.predict(action)
    noise_entropy = math.log(2 * math.pi * math.e * action.variance)
    if action.channel == "nuisance":
        return {"predictive_entropy": noise_entropy, "disagreement": 0.,
                "information_gain": 0., "class_information_gain": 0.,
                "parameter_information_gain": 0., "density_evaluations": 0,
                "quadrature_order": order, "integration": "analytic zero-design Gaussian channel"}
    entropy, conditional_entropy = 0., 0.
    for weight, mean, covariance in zip(prediction["class_weights"],
                                         prediction["component_means"],
                                         prediction["component_covariances"]):
        samples = mean + grid @ np.linalg.cholesky(covariance).T
        entropy -= weight * float(quadrature_weights @ _mixture_logp(samples, prediction))
        conditional_entropy += weight * _normal_entropy(covariance)
    difference = prediction["component_means"] - prediction["mean"]
    disagreement = float(np.einsum("k,ki,ki->", prediction["class_weights"], difference, difference))
    k = len(prediction["class_weights"])
    return {
        "predictive_entropy": float(entropy), "disagreement": disagreement,
        "information_gain": float(entropy - noise_entropy),
        "class_information_gain": float(entropy - conditional_entropy),
        "parameter_information_gain": float(conditional_entropy - noise_entropy),
        "density_evaluations": k * k * order * order, "quadrature_order": order,
        "integration": "deterministic tensor Gauss-Hermite approximation, not exact EIG",
    }


def choose(posterior, policy, rng, *, quadrature_order=9, deadline=None):
    """Only public posterior/action metadata and an independent policy RNG are admitted."""
    if policy not in POLICIES:
        raise ValueError("Unknown fixed policy")
    if posterior.observations_used >= posterior.contract["max_observations"]:
        raise ValueError("Observation budget exhausted")
    actions = posterior.available
    if not actions:
        raise ValueError("No permitted actions remain")
    start = time.perf_counter()
    if deadline is not None and start >= deadline:
        raise TimeoutError("Acquisition computation cap reached")
    rows = []
    if policy == "random":
        selected = actions[int(rng.integers(len(actions)))]
        rows = [{"action_index": a.index, "sampling_probability": 1 / len(actions)} for a in actions]
    elif policy == "space_filling":
        # Strong fixed baseline uses known zero-design metadata, available equally to all policies.
        useful = [a for a in actions if a.channel == "target"] or list(actions)
        observed = [posterior.action(i).t for i in posterior.observed_indices
                    if posterior.action(i).channel == "target"]
        for a in useful:
            distance = min(abs(a.t - x) for x in observed) if observed else abs(a.t)
            rows.append({"action_index": a.index, "score": float(distance)})
        selected = posterior.action(max(rows, key=lambda r: (r["score"], -r["action_index"]))["action_index"])
    else:
        for a in actions:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Acquisition computation cap reached")
            if policy == "disagreement":
                prediction = posterior.predict(a)
                difference = prediction["component_means"] - prediction["mean"]
                value = float(np.einsum("k,ki,ki->", prediction["class_weights"], difference, difference))
                score = {"disagreement": value, "density_evaluations": 0,
                         "integration": "analytic between-class mean disagreement only"}
            else:
                score = acquisition_scores(posterior, a, quadrature_order)
            # All approximations, including small negative MI estimates, remain in the raw trace.
            score.update(action_index=a.index, score=score[policy] / a.cost)
            rows.append(score)
        selected = posterior.action(max(rows, key=lambda r: (r["score"], -r["action_index"]))["action_index"])
    return selected, {
        "policy": policy, "fixed_policy": True, "candidate_scores": rows,
        "density_evaluations": sum(r.get("density_evaluations", 0) for r in rows),
        "permitted_remaining_indices": [a.index for a in actions],
        "selection_wall_seconds": time.perf_counter() - start,
    }


def hypothetical_prediction(posterior, action):
    before = posterior.snapshot()["sha256"]
    prediction = posterior.predict(action)
    result = {
        "provenance": "hypothetical", "parent_factual_posterior": before,
        "model_version": digest(posterior.contract), "simulation_depth": 1,
        "action": action.record(), "assumptions": "Same declared generator menu, priors and noise model.",
        "prediction": {key: value.tolist() for key, value in prediction.items()},
        "observation_count": 0, "learning_eligible": False,
    }
    if posterior.snapshot()["sha256"] != before:
        raise AssertionError("Prediction modified factual memory")
    return result


def graph_for(posterior):
    """A real executable-definition seam; analytic coefficients remain explicit side state."""
    from sera.world_graph import ExecutableDefinition, ExecutableGraph

    interpreter = digest({**posterior.contract["source_hashes"], **owner_seam_hashes()})
    definition = ExecutableDefinition.build("gg-act-contract", posterior.contract, SCHEMA, interpreter)
    return ExecutableGraph(SCHEMA, interpreter, (definition,))


class OwnerBoundAcquisition:
    """References one existing LiveSituation/SharedOwnerRef; creates no neural owner."""

    def __init__(self, situation, posterior):
        from sera.world_graph import LiveSituation

        if type(situation) is not LiveSituation or type(posterior) is not Posterior:
            raise ValueError("An existing factual situation and analytic posterior are required")
        if situation.graph != graph_for(posterior):
            raise ValueError("Situation graph does not own this analytic experiment contract")
        if [e.record() for e in situation.observations] != posterior.snapshot()["payload"]["observed_events"]:
            raise ValueError("Factual prefixes differ")
        self.situation, self.posterior = situation, posterior
        self._mechanism = situation.mechanism
        self._owner = situation.owner.owner
        self._seam_sources = owner_seam_hashes()

    def require_current(self):
        self.situation.owner.require_current(self._owner)
        if (self.posterior.contract["source_hashes"] != source_hashes()
                or self._seam_sources != owner_seam_hashes()):
            raise ValueError("Acquisition interpreter sources changed; replay and rebind")
        self._mechanism.validate(self.situation.graph, self.situation.owner)
        if self.situation.mechanism != self._mechanism:
            raise ValueError("Acquisition binding changed; replay and explicitly rebind")
        if [e.record() for e in self.situation.observations] != self.posterior.snapshot()["payload"]["observed_events"]:
            raise ValueError("Analytic and shared factual prefixes diverged")

    def observe(self, event):
        self.require_current()
        self.posterior.validate_event(event)
        self.situation.budget.add("analytic_replay_and_update_steps", self.posterior.observations_used + 1)
        # Prepare side-state first so a numerical rejection cannot admit a half-update.
        successor = Posterior.restore(self.posterior.snapshot())
        diagnostic = successor.observe(event)
        self.situation.observe(event)
        self.posterior = successor
        return diagnostic

    def branch(self, action):
        self.require_current()
        result = hypothetical_prediction(self.posterior, action)
        branch = self.situation.fork()
        mean = result["prediction"]["mean"]
        event = Event(SCHEMA, len(self.situation.observations),
                      (action.index, action.t, action.variance, *mean),
                      Provenance("GG-ACT-001/model", digest(result), EvidenceKind.PREDICTION),
                      role=EventRole.HYPOTHETICAL)
        branch.assume(event)
        result["shared_hypothetical_situation"] = branch.snapshot()
        result["shared_owner_mechanism"] = self._mechanism.record()
        return result

    def choose(self, policy, rng, **options):
        self.require_current()
        if policy in {"information_gain", "predictive_entropy"}:
            order = options.get("quadrature_order", 9)
            _quadrature(order)
            count = sum(a.channel == "target" for a in self.posterior.available)
            count *= len(self.posterior.contract["specs"]) ** 2 * order ** 2
            self.situation.budget.add("acquisition_density_evaluations", count)
        else:
            self.situation.budget.add("acquisition_candidate_visits", len(self.posterior.available))
        return choose(self.posterior, policy, rng, **options)

    def snapshot(self):
        self.require_current()
        payload = {"format": "sera-owner-bound-acquisition-v1",
                   "situation": self.situation.snapshot(), "analytic_side_state": self.posterior.snapshot(),
                   "seam_sources": self._seam_sources, "mechanism": self._mechanism.record(),
                   "claim": "Existing common owner and factual event log; analytic side state is not neural transfer."}
        return {"payload": payload, "sha256": digest(payload)}

    @classmethod
    def restore(cls, record, situation):
        # The caller restores the actual live situation with its existing owner first.
        if (set(record) != {"payload", "sha256"} or digest(record["payload"]) != record["sha256"]
                or record["payload"]["format"] != "sera-owner-bound-acquisition-v1"):
            raise ValueError("Bound acquisition snapshot integrity failure")
        payload = record["payload"]
        if payload["seam_sources"] != owner_seam_hashes():
            raise ValueError("Shared seam sources changed")
        current = situation.snapshot()
        if (current["payload"] != payload["situation"]["payload"]
                or payload["mechanism"] != situation.mechanism.record()):
            raise ValueError("Restoration requires the same factual situation and complete mechanism")
        old_budget = payload["situation"]["budget"]
        if (current["budget"]["limit"] != old_budget["limit"] or any(
                current["budget"]["counts"].get(key, 0) < count
                for key, count in old_budget["counts"].items())):
            raise ValueError("Restoration cannot reset the shared execution budget")
        return cls(situation, Posterior.restore(payload["analytic_side_state"]))


# Assessor and artifact runner. These functions never expose a world to choose().
def _rng(seed, namespace):
    integer = int.from_bytes(hashlib.sha256(f"{namespace}/{seed}".encode()).digest()[:8], "little")
    return np.random.default_rng(integer)


def _assessor(seed, namespace, family, actions, query_t, prior_variance):
    table = {s["kind"] if s["kind"] != "expression" else s["name"]: s for s in SPECS}
    spec = table["circle" if family == "menu_exception" else family]
    rng = _rng(seed, namespace + "/world/" + family)
    coefficients = rng.normal(0, math.sqrt(prior_variance), design(spec, np.array([0.])).shape[1])

    def clean(t):
        values = (design(spec, np.asarray(t)) @ coefficients).reshape(-1, 2)
        if family == "menu_exception":
            phase = (np.asarray(t) - .35) / .6
            window = np.where((phase > 0) & (phase < 1), np.sin(np.pi * phase) ** 2, 0.)
            values += window[:, None] * np.column_stack((1.5 * np.sin(7 * np.pi * np.asarray(t)),
                                                         .7 * np.cos(5 * np.pi * np.asarray(t))))
        return values

    observed = {}
    for action in actions:
        noise_rng = _rng(seed, f"{namespace}/{family}/observation/{action.index}")
        mean = clean([action.t])[0] if action.channel == "target" else np.zeros(2)
        observed[action.index] = mean + math.sqrt(action.variance) * noise_rng.normal(size=2)
    query_truth = clean(query_t)
    noise_variance = next(a.variance for a in actions if a.channel == "target")
    query_rng = _rng(seed, f"{namespace}/{family}/independent-query-noise")
    return {
        "private_family": family, "private_environment_seed": seed,
        "private_coefficients": coefficients.tolist(), "base_spec": spec,
        "query_t": list(query_t), "query_truth": query_truth.tolist(),
        "query_observed": (query_truth + math.sqrt(noise_variance)
                           * query_rng.normal(size=query_truth.shape)).tolist(),
        "query_noise_variance": noise_variance,
        "potential_observation_pairs": {str(k): v.tolist() for k, v in observed.items()},
        "interpretation": "Assessor-only potential outcomes; only chosen observations enter learner.",
    }


def _prediction_record(posterior, query_t, noise_variance):
    # Query records are computed only after each update for scoring; never fed into acquisition.
    means, covariances = [], []
    for spec, mean, covariance in zip(posterior.contract["specs"],
                                      posterior._means, posterior._covariances):
        a = design(spec, np.asarray(query_t)).reshape(len(query_t), 2, -1)
        means.append(a @ mean)
        covariances.append(a @ covariance @ np.swapaxes(a, 1, 2) + noise_variance * np.eye(2))
    component_means = np.stack(means, axis=1)
    component_covariances = np.stack(covariances, axis=1)
    predicted = []
    for means, covariances in zip(component_means, component_covariances):
        mu = np.einsum("k,ki->i", posterior.weights, means)
        centered = means - mu
        cov = (np.einsum("k,kij->ij", posterior.weights, covariances)
               + np.einsum("k,ki,kj->ij", posterior.weights, centered, centered))
        predicted.append({"mean": mu.tolist(), "covariance": cov.tolist(),
                          "component_means": means.tolist(),
                          "component_covariances": covariances.tolist(),
                          "log_weights": posterior._log_weights.tolist(),
                          "class_weights": posterior.weights.tolist()})
    return predicted


def _query_metrics(predictions, world, actions, initial_indices):
    """Independent external assessment, including exact-CDF marginal mixture intervals.

    Gaussian-mixture quantiles use 48 bisections of SciPy's normal CDF, not a
    moment-matched Gaussian interval. Numerical precision is finite. Assessment
    targets and all resulting metrics are kept outside Posterior and choose().
    """
    from scipy.special import ndtr

    weights = np.asarray([p["class_weights"] for p in predictions])
    means = np.asarray([p["component_means"] for p in predictions])
    covariance = np.asarray([p["component_covariances"] for p in predictions])
    sigma = np.sqrt(np.diagonal(covariance, axis1=2, axis2=3))
    lower_start = np.min(means - 12 * sigma, axis=1)
    upper_start = np.max(means + 12 * sigma, axis=1)
    bounds = []
    for probability in (.025, .975):
        lower, upper = lower_start.copy(), upper_start.copy()
        for _ in range(48):
            middle = (lower + upper) / 2
            cdf = np.sum(weights[:, :, None] * ndtr((middle[:, None, :] - means) / sigma), axis=1)
            lower = np.where(cdf < probability, middle, lower)
            upper = np.where(cdf >= probability, middle, upper)
        bounds.append((lower + upper) / 2)
    low, high = bounds
    observed, truth = np.asarray(world["query_observed"]), np.asarray(world["query_truth"])
    center = np.asarray([p["mean"] for p in predictions])
    squared_error = (center - truth) ** 2
    nll = np.asarray([-float(_mixture_logp(y, p)) for y, p in zip(observed, predictions)])
    covered = (observed >= low) & (observed <= high)
    query_t = np.asarray(world["query_t"])
    target_t = [a.t for a in actions if a.channel == "target"]
    prefix_t = [a.t for a in actions if a.index in initial_indices and a.channel == "target"]
    inside = (query_t >= min(target_t)) & (query_t <= max(target_t))
    prefix = ((query_t >= min(prefix_t)) & (query_t <= max(prefix_t))
              if prefix_t else np.zeros(len(query_t), dtype=bool))
    masks = {"all": np.ones(len(query_t), dtype=bool), "prefix_window": prefix,
             "outside_prefix_within_grid": inside & ~prefix, "outside_grid": ~inside,
             "assessment_exception_window": (query_t > .35) & (query_t < .95)}
    regions = {}
    for name, mask in masks.items():
        if not mask.any():
            regions[name] = {"query_count": 0}
            continue
        regions[name] = {"query_count": int(mask.sum()), "clean_mse": float(squared_error[mask].mean()),
                         "observed_joint_nll": float(nll[mask].mean()),
                         "marginal_95_coverage": float(covered[mask].mean()),
                         "marginal_95_width": float((high - low)[mask].mean())}
    return {"regions": regions, "per_query_squared_error": squared_error.tolist(),
            "per_query_observed_joint_nll": nll.tolist(), "marginal_95_lower": low.tolist(),
            "marginal_95_upper": high.tolist(), "marginal_95_covered": covered.tolist(),
            "interpretation": "Finite independent query observations; raw Bayesian equal-tail mixture intervals. "
                              "No calibration guarantee under adaptive sampling or menu inadequacy."}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _validate_protocol(protocol):
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Frozen source hashes differ; create a new protocol revision")
    root = Path(__file__).resolve().parents[2]
    for name, expected in protocol.get("frozen_input_hashes", {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen protocol input differs: {name}")
    if tuple(protocol["policies"]) != POLICIES or not 0 < protocol["local_cap_seconds"] <= 150:
        raise ValueError("GG-ACT-001 requires the five declared fixed policies and bounded execution")
    seen = set()
    for phase in ("plumbing", "development", "final"):
        config = protocol[phase]
        seeds = config["environment_seeds"]
        if (not seeds or any(type(seed) is not int for seed in seeds)
                or len(set(seeds)) != len(seeds) or seen.intersection(seeds)):
            raise ValueError("Distinct development/final seed namespaces are required")
        seen.update(seeds)
        if (len(config["policy_seeds"]) != len(seeds)
                or any(type(seed) is not int for seed in config["policy_seeds"])
                or len(set(config["policy_seeds"])) != len(seeds)):
            raise ValueError("Independent policy RNG seeds must be assigned per paired-world replicate")
        actions = permitted_grid(**config["grid"])
        _quadrature(config["quadrature_order"])
        if (not 1 <= config["observation_budget"] <= len(actions)
                or len(set(config["initial_indices"])) != len(config["initial_indices"])
                or len(config["initial_indices"]) >= config["observation_budget"]
                or any(index not in {a.index for a in actions if a.channel == "target"}
                       for index in config["initial_indices"])):
            raise ValueError("Invalid shared prefix or observation budget")
        if not 5 <= config["query_count"] <= 257:
            raise ValueError("Independent evaluation grid is bounded")
        if not set(config["families"]).issubset({"circle", "ellipse", "line", "radial_orbit", "menu_exception"}):
            raise ValueError("Unknown assessor family")


def _summary_input(record):
    """Retain only summary inputs in RAM; all high-volume raw vectors stay on disk."""
    result = {key: value for key, value in record.items()
              if key not in {"trace", "prior_posterior", "final_posterior", "policy_rng_state"}}
    result["trace"] = [{"query_metrics": {"regions": step["query_metrics"]["regions"]},
                        "diagnostic": step["diagnostic"],
                        "posterior": {"payload": {"class_weights": step["posterior"]["payload"]["class_weights"]}}}
                       for step in record["trace"]]
    return result


def _summary(records):
    """Frozen exploratory evaluator: paired worlds are the resampling unit."""
    complete = [record for record in records if record["status"] == "complete"]
    rows = []
    for record in complete:
        last = record["trace"][-1]
        metrics = last["query_metrics"]["regions"]["all"]
        rows.append({"case": record["case"], "family": record["case"].split("-", 2)[2],
                     "policy": record["policy"], **metrics,
                     "nuisance_observations": record["nuisance_observations"],
                     "alarm_any": any(s["diagnostic"]["all_models_inadequate_alarm"] for s in record["trace"]),
                     "final_max_class_weight": max(last["posterior"]["payload"]["class_weights"]),
                     "selection_wall_seconds": record["selection_wall_seconds"],
                     "density_evaluations": record["density_evaluations"],
                     "retained_side_state_bytes": record["retained_side_state_bytes"]})
    aggregates, pairs, curves = [], [], []
    for family in sorted({r["family"] for r in rows}):
        family_records = [record for record in complete if record["case"].split("-", 2)[2] == family]
        for policy in POLICIES:
            cell = [r for r in rows if r["family"] == family and r["policy"] == policy]
            if not cell:
                continue
            metrics = {key: float(np.mean([r[key] for r in cell])) for key in cell[0]
                       if key not in {"case", "family", "policy"}}
            aggregates.append({"family": family, "policy": policy, "world_count": len(cell), **metrics})
            per_policy = [r for r in family_records if r["policy"] == policy]
            for step in range(min(len(r["trace"]) for r in per_policy)):
                for region in sorted(per_policy[0]["trace"][step]["query_metrics"]["regions"]):
                    observations = [r["trace"][step]["query_metrics"]["regions"][region] for r in per_policy]
                    metrics = {key: float(np.mean([r[key] for r in observations])) for key in observations[0]}
                    curves.append({"family": family, "policy": policy, "observations": step + 1,
                                   "region": region, "world_count": len(per_policy), **metrics})
        treatment = {r["case"]: r for r in rows if r["family"] == family and r["policy"] == "information_gain"}
        for control in POLICIES[:-1]:
            baseline = {r["case"]: r for r in rows if r["family"] == family and r["policy"] == control}
            common = sorted(set(treatment) & set(baseline))
            if not common:
                continue
            for metric in ("clean_mse", "observed_joint_nll", "marginal_95_coverage", "nuisance_observations"):
                differences = np.asarray([treatment[key][metric] - baseline[key][metric] for key in common])
                rng = _rng(130041, f"frozen-exploratory-bootstrap/{family}/{control}/{metric}")
                resampled = differences[rng.integers(len(common), size=(1000, len(common)))].mean(axis=1)
                pairs.append({"family": family, "control": control, "metric": metric,
                              "paired_worlds": common, "IG_minus_control_raw": differences.tolist(),
                              "mean_difference": float(differences.mean()),
                              "bootstrap_95_percentile_interval": np.quantile(resampled, [.025, .975]).tolist()})
    return {"complete_policy_records": len(complete), "raw_final_rows": rows,
            "aggregates": aggregates, "paired_comparisons": pairs, "observation_curves": curves,
            "interpretation": "Exploratory paired-world bootstrap, 1000 draws; no multiplicity correction. "
                              "Surrogate acquisition values are not external prediction evidence."}


def run_study(protocol, output, *, phase="plumbing"):
    """Explicit local invocation; defaults to plumbing, never launches future runs."""
    _validate_protocol(protocol)
    if phase not in {"plumbing", "development", "final"}:
        raise ValueError("Unknown seed namespace")
    config = protocol[phase]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    deadline = start + protocol["local_cap_seconds"]
    actions = permitted_grid(**config["grid"])
    query_t = np.linspace(-1.8, 1.8, config["query_count"]).tolist()
    _atomic_json(output / "protocol.json", protocol)
    sources = output / "sources"
    sources.mkdir()
    for name in ("core.py", "acquisition.py"):
        (sources / name).write_bytes((Path(__file__).parent / name).read_bytes())
    records, result_records, interrupted = [], [], None
    progress = {"status": "running", "phase": phase, "protocol_sha256": digest(protocol),
                "completed_policy_records": 0, "planned_policy_records":
                len(config["environment_seeds"]) * len(config["families"]) * len(protocol["policies"])}
    _atomic_json(output / "progress.json", progress)
    for replicate_index, seed in enumerate(config["environment_seeds"]):
        for family in config["families"]:
            if time.perf_counter() >= deadline:
                interrupted = "local wall cap before next world"
                break
            case = f"{phase}-{seed}-{family}"
            assessor_start = time.perf_counter()
            world = _assessor(seed, phase, family, actions, query_t, protocol["prior_variance"])
            preparation = time.perf_counter() - assessor_start
            world["preparation_wall_seconds"] = preparation
            _atomic_json(output / "cases" / case / "assessor-only.json", world)
            for policy in protocol["policies"]:
                posterior = Posterior(actions, prior_variance=protocol["prior_variance"],
                                      max_observations=config["observation_budget"],
                                      alarm_alpha=protocol["alarm_alpha"])
                policy_seed = config["policy_seeds"][replicate_index]
                policy_rng = _rng(policy_seed, f"{phase}/fixed-policy/{policy}")
                trace = []
                prior_snapshot = posterior.snapshot()
                path = output / "cases" / case / f"{policy}.json"

                def case_record(status):
                    return {
                        "experiment": protocol["experiment_id"], "phase": phase, "case": case,
                        "policy": policy, "policy_seed": policy_seed,
                        "replicate_index": replicate_index, "status": status, "prior_posterior": prior_snapshot,
                        "trace": trace, "final_posterior": posterior.snapshot(),
                        "policy_rng_state": policy_rng.bit_generator.state,
                        "observation_count": posterior.observations_used,
                        "nuisance_observations": sum(s["action"]["channel"] == "nuisance" for s in trace),
                        "density_evaluations": sum(s["decision"]["density_evaluations"] for s in trace),
                        "selection_wall_seconds": sum(s["decision"]["selection_wall_seconds"] for s in trace),
                        "posterior_update_wall_seconds": sum(s["costs"]["posterior_update_wall_seconds"] for s in trace),
                        "hypothetical_wall_seconds": sum(s["costs"]["hypothetical_wall_seconds"] for s in trace),
                        "external_evaluation_wall_seconds": sum(s["costs"]["external_evaluation_wall_seconds"] for s in trace),
                        "completed_checkpoint_wall_seconds": sum(s["costs"]["checkpoint_wall_seconds"] for s in trace),
                        "target_component_updates": len(SPECS) * sum(s["action"]["channel"] == "target" for s in trace),
                        "hypothetical_prediction_branches": len(trace), "shared_owner_bound": False,
                        "retained_side_state_bytes": len(canonical(posterior.snapshot()).encode()),
                    }

                _atomic_json(path, case_record("running"))
                for step in range(config["observation_budget"]):
                    try:
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("Local wall cap before next paid observation")
                        if step < len(config["initial_indices"]):
                            action = posterior.action(config["initial_indices"][step])
                            decision = {"policy": "common-prefix", "candidate_scores": [],
                                        "density_evaluations": 0, "selection_wall_seconds": 0.}
                        else:
                            action, decision = choose(posterior, policy, policy_rng,
                                                      quadrature_order=config["quadrature_order"],
                                                      deadline=deadline)
                    except TimeoutError as error:
                        interrupted = str(error)
                        break
                    began = time.perf_counter()
                    branch = hypothetical_prediction(posterior, action)
                    branch_wall = time.perf_counter() - began
                    event = observed_event(action, world["potential_observation_pairs"][str(action.index)],
                                           step, f"action-{action.index}")
                    began = time.perf_counter()
                    diagnostic = posterior.observe(event)
                    update_wall = time.perf_counter() - began
                    began = time.perf_counter()
                    prediction = _prediction_record(posterior, query_t, config["grid"]["noise"] ** 2)
                    metrics = _query_metrics(prediction, world, actions, config["initial_indices"])
                    evaluation_wall = time.perf_counter() - began
                    trace.append({
                        "step": step, "action": action.record(), "decision": decision,
                        "hypothetical_before_observation": branch, "observed_event": event.record(),
                        "diagnostic": diagnostic, "posterior": posterior.snapshot(),
                        "query_prediction": prediction, "query_metrics": metrics,
                        "costs": {"hypothetical_wall_seconds": branch_wall,
                                  "posterior_update_wall_seconds": update_wall,
                                  "external_evaluation_wall_seconds": evaluation_wall,
                                  "checkpoint_wall_seconds": 0.},
                    })
                    began = time.perf_counter()
                    _atomic_json(path, case_record("running"))
                    trace[-1]["costs"]["checkpoint_wall_seconds"] = time.perf_counter() - began
                record = case_record("capped" if interrupted else "complete")
                _atomic_json(path, record)
                result_records.append(_summary_input(record))
                records.append({"path": str(path.relative_to(output)).replace("\\", "/"),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "status": record["status"], "observations": record["observation_count"]})
                progress["completed_policy_records"] += record["status"] == "complete"
                _atomic_json(output / "progress.json", progress)
                if interrupted:
                    break
            if interrupted:
                break
        if interrupted:
            break
    _atomic_json(output / "summary.json", _summary(result_records))
    progress["status"] = "capped" if interrupted else "complete"
    _atomic_json(output / "progress.json", progress)
    import scipy
    manifest = {
        "experiment": protocol["experiment_id"], "phase": phase,
        "status": "capped" if interrupted else "complete", "interruption": interrupted,
        "records": records, "protocol_sha256": digest(protocol), "source_hashes": source_hashes(),
        "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu,
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "platform": platform.platform()},
        "source_decoder_bytes": sum((sources / name).stat().st_size
                                    for name in ("core.py", "acquisition.py")),
        "artifact_files": {str(p.relative_to(output)).replace("\\", "/"):
                           {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in output.rglob("*") if p.is_file()},
        "cost_boundary": "Audit bytes and retained analytic-state bytes reported separately. "
                         "Python/NumPy/SciPy/Torch installation bytes are shared runtime, not counted "
                         "as a self-contained packaged decoder. No full compression claim.",
        "claim_boundary": "Fixed policies, supplied phase/menu and noise, analytic side state. "
                          "No learned applicability, neural transfer or learned investigator.",
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def replay_study(output):
    output = Path(output)
    started = time.perf_counter()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Replay source version mismatch")
    for name, info in manifest["artifact_files"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Raw artifact integrity failure")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    if digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Protocol digest differs")
    config = protocol[manifest["phase"]]
    checked, observed, summaries = 0, 0, []
    for item in manifest["records"]:
        record = json.loads((output / item["path"]).read_text(encoding="utf-8"))
        world = json.loads((output / item["path"]).with_name("assessor-only.json").read_text(encoding="utf-8"))
        posterior = Posterior.restore(record["prior_posterior"])
        expected_prior = Posterior(permitted_grid(**config["grid"]), prior_variance=protocol["prior_variance"],
                                   max_observations=config["observation_budget"], alarm_alpha=protocol["alarm_alpha"])
        if posterior.snapshot() != expected_prior.snapshot():
            raise ValueError("Initial learner state differs from the frozen public contract")
        policy_seed = config["policy_seeds"][record["replicate_index"]]
        if policy_seed != record["policy_seed"]:
            raise ValueError("Independent policy RNG seed differs")
        policy_rng = _rng(policy_seed, f"{record['phase']}/fixed-policy/{record['policy']}")
        for index, step in enumerate(record["trace"]):
            if index < len(config["initial_indices"]):
                action = posterior.action(config["initial_indices"][index])
                decision = {"policy": "common-prefix", "candidate_scores": [],
                            "density_evaluations": 0, "selection_wall_seconds": 0.}
            else:
                action, decision = choose(posterior, record["policy"], policy_rng,
                                          quadrature_order=config["quadrature_order"])
            if (step["step"] != index or step["action"] != action.record()
                    or {k: v for k, v in step["decision"].items() if k != "selection_wall_seconds"}
                    != {k: v for k, v in decision.items() if k != "selection_wall_seconds"}):
                raise ValueError("Fixed policy or full candidate score replay differs")
            branch = step["hypothetical_before_observation"]
            if hypothetical_prediction(posterior, action) != branch:
                raise ValueError("Hypothetical prediction or factual parent replay differs")
            event = observed_event(action, world["potential_observation_pairs"][str(action.index)],
                                   index, f"action-{action.index}")
            if event.record() != step["observed_event"]:
                raise ValueError("Observed potential outcome replay differs")
            if posterior.observe(event) != step["diagnostic"] or posterior.snapshot() != step["posterior"]:
                raise ValueError("Observed posterior or residual diagnostic replay differs")
            predictions = _prediction_record(posterior, world["query_t"], world["query_noise_variance"])
            metrics = _query_metrics(predictions, world, posterior.actions, config["initial_indices"])
            if predictions != step["query_prediction"] or metrics != step["query_metrics"]:
                raise ValueError("External raw prediction or evaluation replay differs")
            observed += 1
        if (posterior.snapshot() != record["final_posterior"]
                or posterior.observations_used != record["observation_count"]
                or policy_rng.bit_generator.state != record["policy_rng_state"]):
            raise ValueError("Final replay differs")
        summaries.append(_summary_input(record))
        checked += 1
    if _summary(summaries) != json.loads((output / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("Exploratory aggregate and paired comparison replay differs")
    return {"records_replayed": checked, "observation_steps_replayed": observed,
            "status": "passed", "wall_seconds": time.perf_counter() - started,
            "source_hashes": source_hashes(), "protocol_sha256": digest(protocol),
            "scope": "Every fixed action and candidate score, branch, posterior, raw external prediction, "
                     "coverage/NLL vector and exploratory summary replayed exactly; artifact hashes checked. "
                     "No world re-query; stored potential outcomes remain assessor-only."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plumbing", "development", "final"), default="plumbing")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        result = replay_study(args.output)
        _atomic_json(args.output / "replay-report.json", result)
    else:
        if args.protocol is None:
            parser.error("--protocol is required for execution")
        import torch
        torch.set_num_threads(1)
        result = run_study(json.loads(args.protocol.read_text(encoding="utf-8")),
                           args.output, phase=args.phase)
        result = {key: result[key] for key in ("status", "phase", "wall_seconds", "cpu_seconds")}
    print(canonical(result))


if __name__ == "__main__":
    main()
