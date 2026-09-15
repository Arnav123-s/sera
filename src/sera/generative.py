"""A supplied linear-Gaussian grammar inside the existing shared numeric owner.

This adds persistent analytic acquisition, not neural transfer or learned semantic
applicability. All four alternatives explain the same observed trajectory. Phase,
grammar, units, coefficient priors and observation noise are supplied. Conditional
branches complete coordinates at a supplied phase; they are not causal simulators.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import platform
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from sera.contracts import EvidenceKind, Provenance
from sera.event_ir import Event, EventRole, EventSchema
from sera.shared import SharedR1, replace_shared_owner
from sera.storage import canonical, digest
from sera.world_graph import (
    ExecutableDefinition,
    ExecutableGraph,
    ExecutionBudget,
    LiveSituation,
    SharedOwnerRef,
    interpreter_fingerprint,
)

CLASSES = ("circle", "ellipse", "line", "radial_growth")
DIMENSIONS = (4, 6, 4, 6)
GRAMMAR = "phase-pi-t-linear-gaussian-v1"
SCHEMA = EventSchema("phase-world", 1, "numeric", 3, "dimensionless",
                     entity_slots=1, encoding="phase-pi-t-xy-v1")
_EMPTY = "0" * 64


def source_manifest():
    """Explicit local interpreter closure plus the declared external runtime."""
    root = Path(__file__).parent
    sources = {f"sera/{path.name}": path.read_bytes() for path in sorted(root.glob("*.py"))}
    sources["runtime"] = canonical({"python": platform.python_version(),
                                     "implementation": platform.python_implementation(),
                                     "torch": str(torch.__version__)}).encode()
    return sources


def _tensor_hash(value):
    return bytes(value.detach().cpu().tolist()).hex()


def _write_hash(target, value):
    target.copy_(torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device=target.device))


def _chain(events):
    value = _EMPTY
    for event in events:
        value = digest([value, event.record()])
    return value


class GenerativeSharedR1(SharedR1):
    """SharedR1 with one registered Bayesian workspace and no extra parameters."""

    def __init__(self, *args, generator_noise=0.05, generator_prior_variance=4.0,
                 generator_grammar=GRAMMAR, generator_interpreter_sha256=None,
                 generator_parent_sha256=None, generator_requires_grad=None, **kwargs):
        super().__init__(*args, **kwargs)
        if generator_requires_grad is not None:
            parameters = dict(self.named_parameters())
            if (type(generator_requires_grad) is not dict or set(generator_requires_grad) != set(parameters)
                    or any(type(flag) is not bool for flag in generator_requires_grad.values())):
                raise ValueError("Generator checkpoint gradient flags must match every parameter")
            for name, parameter in parameters.items():
                parameter.requires_grad_(generator_requires_grad[name])
        self._initialize_generator(generator_noise, generator_prior_variance, generator_grammar,
                                   generator_interpreter_sha256, generator_parent_sha256)

    def _initialize_generator(self, noise, prior, grammar, interpreter, parent):
        if (type(noise) not in (int, float) or not math.isfinite(noise) or noise <= 0
                or type(prior) not in (int, float) or not math.isfinite(prior) or prior <= 0):
            raise ValueError("Generator noise and coefficient prior variance must be positive")
        actual = interpreter_fingerprint(source_manifest())
        if grammar != GRAMMAR or (interpreter is not None and interpreter != actual):
            raise ValueError("Generator grammar or complete interpreter fingerprint changed")
        if parent is not None and (type(parent) is not str or len(parent) != 64
                                   or any(char not in "0123456789abcdef" for char in parent)):
            raise ValueError("Generator parent requires a complete owner identity")
        self.generator_noise, self.generator_prior_variance = float(noise), float(prior)
        self.generator_grammar, self.generator_interpreter_sha256 = grammar, actual
        self.generator_parent_sha256 = parent
        device = next(self.parameters()).device
        shapes = {"gram": (4, 6, 6), "rhs": (4, 6), "square_sum": (),
                  "mean": (4, 6), "covariance": (4, 6, 6), "log_evidence": (4,),
                  "class_probabilities": (4,)}
        for name, shape in shapes.items():
            self.register_buffer("generator_" + name, torch.zeros(shape, dtype=torch.float64, device=device))
        for name in ("observations", "labels", "map_class"):
            self.register_buffer("generator_" + name, torch.zeros((), dtype=torch.int64, device=device))
        for name in ("observation_chain", "label_chain", "entity_hash"):
            self.register_buffer("generator_" + name, torch.zeros(32, dtype=torch.uint8, device=device))
        self._refresh_posterior()
        self.validity()

    @classmethod
    def from_shared(cls, owner, *, generator_noise=0.05, generator_prior_variance=4.0):
        """Clone an existing trained SharedR1; preserve its old state byte for byte.

        A deepcopy preserves dtype, device, program data, optional adapters, flags
        and module alias relationships. The class has the same Python object
        layout; only its added registered workspace and exported type differ.
        """
        if type(owner) is not SharedR1:
            raise ValueError("Conversion needs an ordinary SharedR1, not an existing descendant")
        owner.validity()
        before_config = owner.export_config()
        parent = SharedOwnerRef(owner).identity
        result = copy.deepcopy(owner)
        result.__class__ = cls
        result._initialize_generator(generator_noise, generator_prior_variance, GRAMMAR, None, parent)
        if SharedR1.export_config(result) != before_config:
            raise ValueError("Conversion altered the preexisting shared configuration")
        for name, value in owner.state_dict().items():
            cloned = result.state_dict()[name]
            if (value.dtype != cloned.dtype or value.device != cloned.device
                    or not torch.equal(value, cloned) or (value.numel() and value.data_ptr() == cloned.data_ptr())):
                raise ValueError("Conversion failed to independently preserve existing state")
        return result

    def export_config(self):
        config = super().export_config()
        if not hasattr(self, "generator_grammar"):
            return config
        return {**config, "type": "generative_shared_r1", "generator_noise": self.generator_noise,
                "generator_prior_variance": self.generator_prior_variance,
                "generator_grammar": self.generator_grammar,
                "generator_interpreter_sha256": self.generator_interpreter_sha256,
                "generator_parent_sha256": self.generator_parent_sha256,
                "generator_requires_grad": {name: p.requires_grad for name, p in self.named_parameters()}}

    def _design(self, phases):
        t = torch.as_tensor(phases, dtype=torch.float64, device=self.generator_gram.device)
        if t.ndim != 1 or not len(t) or len(t) > 4096 or not torch.isfinite(t).all():
            raise ValueError("Generator queries need 1 to 4096 finite scalar phases")
        c, s = torch.cos(math.pi * t), torch.sin(math.pi * t)
        a = t.new_zeros(4, len(t), 2, 6)
        # Circle: x=cx+a*cos-b*sin; y=cy+a*sin+b*cos.
        a[0, :, 0, 0], a[0, :, 1, 1] = 1, 1
        a[0, :, 0, 2], a[0, :, 0, 3] = c, -s
        a[0, :, 1, 2], a[0, :, 1, 3] = s, c
        # Ellipse: independent constant, cosine and sine coefficients per axis.
        a[1, :, 0, 0], a[1, :, 1, 1] = 1, 1
        a[1, :, 0, 2], a[1, :, 0, 3] = c, s
        a[1, :, 1, 4], a[1, :, 1, 5] = c, s
        # Line: x=cx+vx*t; y=cy+vy*t.
        a[2, :, 0, 0], a[2, :, 1, 1] = 1, 1
        a[2, :, 0, 2], a[2, :, 1, 3] = t, t
        # Affine complex amplitude: strict radial growth is the collinear subset.
        a[3] = a[0]
        a[3, :, 0, 4], a[3, :, 0, 5] = t*c, -t*s
        a[3, :, 1, 4], a[3, :, 1, 5] = t*s, t*c
        return a

    def _posterior(self):
        means = torch.zeros_like(self.generator_mean)
        covariances = torch.zeros_like(self.generator_covariance)
        evidence = torch.zeros_like(self.generator_log_evidence)
        noise2, prior = self.generator_noise**2, self.generator_prior_variance
        for index, dimension in enumerate(DIMENSIONS):
            eye = torch.eye(dimension, dtype=means.dtype, device=means.device)
            precision = self.generator_gram[index, :dimension, :dimension] / noise2 + eye / prior
            chol = torch.linalg.cholesky(precision)
            covariance = torch.cholesky_inverse(chol)
            rhs = self.generator_rhs[index, :dimension] / noise2
            mean = torch.cholesky_solve(rhs[:, None], chol).flatten()
            means[index, :dimension], covariances[index, :dimension, :dimension] = mean, covariance
            logdet = 2 * chol.diagonal().log().sum() + dimension * math.log(prior)
            evidence[index] = -.5 * (2 * int(self.generator_observations) * math.log(2*math.pi*noise2)
                                      + logdet + self.generator_square_sum/noise2 - rhs @ mean)
        return means, covariances, evidence, evidence.softmax(0)

    @torch.no_grad()
    def _refresh_posterior(self):
        values = self._posterior()
        for name, value in zip(("mean", "covariance", "log_evidence", "class_probabilities"), values):
            getattr(self, "generator_" + name).copy_(value)
        self.generator_map_class.copy_(values[-1].argmax())

    def validity(self):
        report = super().validity()
        # SharedR1 calls validity while constructing its own modules.
        if not hasattr(self, "generator_gram"):
            return report
        for name, value in self.named_buffers():
            if name.startswith("generator_") and not torch.isfinite(value).all():
                raise ValueError("Generator state must remain finite")
            if name.startswith("generator_") and value.is_floating_point() and value.dtype != torch.float64:
                raise ValueError("Generator floating state requires float64 precision")
        if (int(self.generator_observations) < 0 or int(self.generator_observations) > 4096
                or int(self.generator_labels) < 0 or int(self.generator_labels) > 4096
                or self.generator_square_sum < 0):
            raise ValueError("Generator evidence counts or squared observations are invalid")
        recomputed = self._posterior()
        for name, value in zip(("mean", "covariance", "log_evidence", "class_probabilities"), recomputed):
            if not torch.allclose(getattr(self, "generator_" + name), value, atol=1e-8, rtol=1e-8):
                raise ValueError("Generator posterior differs from its registered sufficient statistics")
        if int(self.generator_map_class) != int(recomputed[-1].argmax()):
            raise ValueError("Generator class state differs from its posterior")
        return {**report, "generator_integrity_error": 0.0}

    @torch.no_grad()
    def forward_generator(self, phases):
        a = self._design(phases)
        means = torch.einsum("knij,kj->kni", a, self.generator_mean)
        covariance = torch.einsum("knip,kpq,knjq->knij", a, self.generator_covariance, a)
        covariance += torch.eye(2, dtype=a.dtype, device=a.device) * self.generator_noise**2
        probabilities = self.generator_class_probabilities
        mean = torch.einsum("k,kni->ni", probabilities, means)
        delta = means - mean
        mixture_covariance = torch.einsum("k,knij->nij", probabilities, covariance)
        mixture_covariance += torch.einsum("k,kni,knj->nij", probabilities, delta, delta)
        # Outputs are detached newly allocated tensors, never writable buffer aliases.
        return {"mean": mean, "covariance": mixture_covariance, "class_means": means,
                "class_covariances": covariance, "class_probabilities": probabilities.clone()}

    def forward(self, *args, route="world", **kwargs):
        if route == "generator":
            return self.forward_generator(*args, **kwargs)
        return super().forward(*args, route=route, **kwargs)

    @torch.no_grad()
    def _add_coordinates(self, phase, coordinates):
        a = self._design([phase]).reshape(4, 2, 6)
        y = a.new_tensor(coordinates)
        self.generator_gram.add_(torch.einsum("kni,knj->kij", a, a))
        self.generator_rhs.add_(torch.einsum("kni,n->ki", a, y))
        self.generator_square_sum.add_(y @ y)

    @torch.no_grad()
    def _observe_event(self, event):
        self._add_coordinates(event.values[0], event.values[1:])
        self.generator_observations.add_(1)
        _write_hash(self.generator_observation_chain,
                    digest([_tensor_hash(self.generator_observation_chain), event.record()]))
        _write_hash(self.generator_entity_hash, digest(event.entity_refs))
        self._refresh_posterior()

    @torch.no_grad()
    def _correct_evidence(self, observations, labels):
        self.generator_gram.zero_()
        self.generator_rhs.zero_()
        self.generator_square_sum.zero_()
        replacements = {label.sequence: label for label in labels}
        for event in observations:
            effective = replacements.get(event.sequence, event)
            self._add_coordinates(event.values[0], effective.values[1:])
        self.generator_labels.fill_(len(labels))
        _write_hash(self.generator_label_chain, _chain(labels))
        self._refresh_posterior()

    def generator_graph(self):
        definitions = []
        bodies = (
            {"x": "cx+a*cos(pi*t)-b*sin(pi*t)", "y": "cy+a*sin(pi*t)+b*cos(pi*t)"},
            {"x": "cx+a*cos(pi*t)+b*sin(pi*t)", "y": "cy+c*cos(pi*t)+d*sin(pi*t)"},
            {"x": "cx+vx*t", "y": "cy+vy*t"},
            {"x": "cx+(a+c*t)*cos(pi*t)-(b+d*t)*sin(pi*t)",
             "y": "cy+(a+c*t)*sin(pi*t)+(b+d*t)*cos(pi*t)"},
        )
        for name, dimension, body in zip(CLASSES, DIMENSIONS, bodies):
            definitions.append(ExecutableDefinition.build(name, {"family": body, "coefficients": dimension,
                                                                  "prior_variance": self.generator_prior_variance},
                                                            SCHEMA, self.generator_interpreter_sha256))
        mixture = ExecutableDefinition.build("generator", {
            "grammar": self.generator_grammar, "noise_std": self.generator_noise,
            "class_prior": "uniform", "inference": "conjugate-linear-Gaussian-v1",
            "workspace": "registered-r1-generator-buffers", "conditioning": "supplied-phase",
        }, SCHEMA, self.generator_interpreter_sha256, dependencies=tuple(item.reference for item in definitions))
        return ExecutableGraph(SCHEMA, self.generator_interpreter_sha256, tuple(definitions) + (mixture,))


@dataclass(frozen=True, slots=True)
class FrozenGenerativePredecessor:
    """Immutable checkpoint bytes; each restoration creates an independent solver."""

    checkpoint: bytes
    checkpoint_sha256: str
    solver_sha256: str
    situation_json: str

    @classmethod
    def capture(cls, solver, situation):
        payload = {"schema_version": 3, "neural_state": solver.neural.state_dict(),
                   "neural_interface": solver.neural.export_config(), "skills": solver.skills,
                   "version": solver.version, "components": {
                       name: {"config": module.export_config(), "state": module.state_dict()}
                       for name, module in solver.components.items()}}
        stream = io.BytesIO()
        torch.save(payload, stream)
        raw = stream.getvalue()
        return cls(raw, hashlib.sha256(raw).hexdigest(), solver.identity(), canonical(situation.snapshot()))

    def restore(self):
        from sera.connected import restore_component
        from sera.solver import Solver
        if hashlib.sha256(self.checkpoint).hexdigest() != self.checkpoint_sha256:
            raise ValueError("Immutable predecessor checkpoint integrity failure")
        payload = torch.load(io.BytesIO(self.checkpoint), weights_only=True, map_location="cpu")
        components = {}
        ordered = sorted(payload["components"].items(), key=lambda item: item[1]["config"]["type"].endswith("_view"))
        for name, item in ordered:
            module = restore_component(item["config"], components)
            module.load_state_dict(item["state"])
            components[name] = module
        neural = restore_component(payload["neural_interface"], components)
        neural.load_state_dict(payload["neural_state"])
        solver = Solver(neural, skills=payload["skills"], components=components, version=payload["version"]).eval()
        solver.validate()
        if solver.identity() != self.solver_sha256:
            raise ValueError("Restored predecessor identity differs")
        return solver, SharedGenerativeSession.restore(solver, json.loads(self.situation_json))


@dataclass(frozen=True, slots=True)
class GenerativeUpdate:
    predecessor: FrozenGenerativePredecessor
    record_json: str

    @property
    def record(self):
        return json.loads(self.record_json)


class SharedGenerativeSession:
    """Evidence-safe staged acquisition with a solver-bound complete owner guard.

    Validators are trusted callbacks, not a hostile-code sandbox. Source claims
    remain the caller's responsibility. Corrections preserve observations and use
    the last independently verified label at a position as its effective value.
    """

    def __init__(self, solver, situation_id="phase-world", *, budget=None, max_events=4096):
        self.solver = solver
        owner = self._owner()
        if int(owner.generator_observations) or int(owner.generator_labels):
            raise ValueError("An acquired workspace requires its factual snapshot to resume")
        self.situation = LiveSituation(situation_id, owner.generator_graph(), SharedOwnerRef.from_solver(solver),
                                       budget=budget, max_events=max_events)
        self._lock = threading.RLock()

    def _owner(self):
        owner = self.solver.components["r1"]
        if not isinstance(owner, GenerativeSharedR1):
            raise ValueError("Generative session requires the registered generative shared owner")
        return owner

    @staticmethod
    def _aligned(owner, situation):
        if (int(owner.generator_observations) != len(situation.observations)
                or int(owner.generator_labels) != len(situation.labels)
                or _tensor_hash(owner.generator_observation_chain) != _chain(situation.observations)
                or _tensor_hash(owner.generator_label_chain) != _chain(situation.labels)):
            raise ValueError("Registered acquisition state differs from admitted factual evidence")
        if situation.observations and _tensor_hash(owner.generator_entity_hash) != digest(
            situation.observations[0].entity_refs
        ):
            raise ValueError("Registered trajectory entity differs from admitted observations")

    def _guard(self):
        self.situation._guard()
        self._aligned(self._owner(), self.situation)

    @staticmethod
    @torch.no_grad()
    def _verify_recorded_statistics(owner, situation):
        """Reconcile persisted numerical state with the actual recorded evidence.

        Hash consistency alone cannot establish this relationship. The replay is
        bounded by the factual capacity and charged separately from acquisition.
        """
        observations, labels = situation.observations, situation.labels
        for row in observations + labels:
            if (row.schema != SCHEMA or not all(row.available)
                    or row.entity_refs != observations[0].entity_refs):
                raise ValueError("Recorded generator evidence has incompatible schema or entity")
        if any(row.values[0] != observations[row.sequence].values[0] for row in labels):
            raise ValueError("Recorded correction changed the conditioning phase")
        situation.budget.add("generator_statistics_replay_visits", len(observations))
        effective = {row.sequence: row for row in labels}
        gram, rhs = torch.zeros_like(owner.generator_gram), torch.zeros_like(owner.generator_rhs)
        square_sum = torch.zeros_like(owner.generator_square_sum)
        for row in observations:
            a = owner._design([row.values[0]]).reshape(4, 2, 6)
            y = a.new_tensor(effective.get(row.sequence, row).values[1:])
            gram.add_(torch.einsum("kni,knj->kij", a, a))
            rhs.add_(torch.einsum("kni,n->ki", a, y))
            square_sum.add_(y @ y)
        for name, expected in (("gram", gram), ("rhs", rhs), ("square_sum", square_sum)):
            if not torch.equal(getattr(owner, "generator_"+name), expected):
                raise ValueError("Registered sufficient statistics differ from recorded evidence")
        owner.validity()

    def snapshot(self):
        with self._lock:
            self._guard()
            return self.situation.snapshot()

    @classmethod
    def restore(cls, solver, record, *, situation_id=None, budget=None):
        result = cls.__new__(cls)
        result.solver, result._lock = solver, threading.RLock()
        owner = result._owner()
        situation_id = record["payload"]["situation_id"] if situation_id is None else situation_id
        result.situation = LiveSituation.restore(record, owner.generator_graph(), SharedOwnerRef.from_solver(solver),
                                                 situation_id=situation_id, budget=budget)
        result._guard()
        result._verify_recorded_statistics(owner, result.situation)
        return result

    def _check_event(self, event, *, correction):
        self._guard()
        self.situation._check_event(event)
        if event.schema != SCHEMA or not all(event.available):
            raise ValueError("Generator acquisition needs fully available dimensionless phase and coordinates")
        if self.situation.observations and event.entity_refs != self.situation.observations[0].entity_refs:
            raise ValueError("One generator workspace explains one trajectory entity")
        if correction:
            if not event.eligible_label or event.sequence >= len(self.situation.observations):
                raise ValueError("Generator correction needs independent eligible evidence at an observed position")
            if event.values[0] != self.situation.observations[event.sequence].values[0]:
                raise ValueError("A coordinate correction cannot silently change its conditioning phase")
        elif not event.eligible_observation or event.sequence != len(self.situation.observations):
            raise ValueError("Generator training admits only factual observations in sequence")

    def observe(self, event, *, validator=None):
        return self._update(event, correction=False, validator=validator)

    def correct(self, event, *, validator=None):
        return self._update(event, correction=True, validator=validator)

    def _update(self, event, *, correction, validator):
        with self._lock:
            self._check_event(event, correction=correction)
            situation, owner = self.situation, self._owner()
            parent, parent_mechanism = self.solver.identity(), situation.mechanism.identity
            budget = situation.budget
            budget.add("solver_copies")
            candidate = copy.deepcopy(self.solver)
            proposed = candidate.components["r1"]
            rows = len(situation.observations) if correction else 1
            budget.add("generator_observation_visits", rows)
            budget.add("generator_posterior_solves", len(CLASSES))
            if correction:
                proposed._correct_evidence(situation.observations, situation.labels + (event,))
            else:
                proposed._observe_event(event)
            proposed.validity()
            migrated = situation.migrate(proposed.generator_graph(), SharedOwnerRef(proposed),
                                           expected_mechanism=parent_mechanism)
            # Tentative admission has a separate cost; factual acquisition is
            # charged only after validation. Prior replay uses the shared ledger.
            budget.add("proposed_admissions")
            migrated.budget = ExecutionBudget(1)
            (migrated.admit_label if correction else migrated.observe)(event)
            migrated.budget = budget
            self._aligned(proposed, migrated)
            candidate.validate()
            candidate_identity = candidate.identity()
            candidate_state = migrated.snapshot()["sha256"]
            budget.add("candidate_validations")
            if validator is not None and validator(candidate, migrated) is not True:
                raise ValueError("Generative candidate validation rejected the update")
            if candidate.identity() != candidate_identity or migrated.snapshot()["sha256"] != candidate_state:
                raise ValueError("Generative validator mutated the staged candidate")
            self._guard()
            if self.solver.identity() != parent or self._owner() is not owner:
                raise ValueError("Shared solver changed during generative candidate validation")
            budget.add("predecessor_serializations")
            predecessor = FrozenGenerativePredecessor.capture(self.solver, situation)
            budget.add("verified_labels" if correction else "factual_observations")
            replacement_ref = SharedOwnerRef(proposed)
            # No fallible replay remains after this in-memory pointer replacement.
            replace_shared_owner(self.solver, proposed)
            object.__setattr__(replacement_ref, "_solver", situation.owner._solver)
            migrated.owner = replacement_ref
            self.situation = migrated
            record = {"operation": "verified-correction" if correction else "observation-update",
                      "parent_solver_sha256": parent, "successor_solver_sha256": candidate_identity,
                      "parent_mechanism_sha256": parent_mechanism,
                      "successor_mechanism": migrated.mechanism.record(),
                      "event_sha256": event.identity, "acquired_observations": len(migrated.observations),
                      "acquired_labels": len(migrated.labels), "budget": budget.record(),
                      "predecessor_checkpoint_bytes": len(predecessor.checkpoint)}
            return GenerativeUpdate(predecessor, canonical(record))

    def predict(self, phases):
        with self._lock:
            self._guard()
            self.situation.budget.add("generator_predictions")
            return self._owner().forward_generator(phases)

    def fork(self):
        with self._lock:
            self._guard()
            return GenerativeConditionalBranch(self.situation.fork())


class GenerativeConditionalBranch:
    """Conditional queries share the owner and budget, never the factual write path."""

    def __init__(self, situation):
        self.situation = situation

    def predict(self, phases):
        self.situation.mechanism.validate(self.situation.graph, self.situation.owner)
        # Validate the entire batch before adding assumptions.
        owner = self.situation.owner.owner
        values = owner._design(phases)
        phases = torch.as_tensor(phases, dtype=torch.float64).tolist()
        if len(self.situation.observations) + len(self.situation.assumptions) + len(phases) > self.situation.max_events:
            raise ValueError("Conditional branch capacity exhausted")
        self.situation.budget.add("generator_predictions")
        entity = self.situation.observations[0].entity_refs if self.situation.observations else ("unobserved",)
        for phase in phases:
            index = len(self.situation.observations) + len(self.situation.assumptions)
            event = Event(SCHEMA, index, (float(phase), 0.0, 0.0),
                          Provenance("generator-condition", str(index), EvidenceKind.PREDICTION),
                          (True, False, False), entity, EventRole.HYPOTHETICAL)
            self.situation.assume(event)
        return owner.forward_generator(values.new_tensor(phases))

    def observe(self, event):
        self.situation.observe(event)

    def admit_label(self, event):
        self.situation.admit_label(event)
