"""Owned executable definitions and factual situations, without learned semantics.

Definitions contain immutable JSON data, never mutable kernel registries. Their
interpreter manifest is declared by the caller; this module does not certify its
completeness or execute arbitrary programs. Migration replays event records only,
not neural latent vectors. These contracts are not a hostile-code sandbox.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

from torch import nn

from sera.contracts import EvidenceKind
from sera.event_ir import Event, EventRole, EventSchema
from sera.session_state import model_identity
from sera.shared import SharedR1
from sera.storage import canonical, digest


def _hash(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("A full SHA-256 fingerprint is required")
    return value


def interpreter_fingerprint(sources):
    """Hash an explicit manifest of named interpreter source bytes."""
    if not sources or any(type(name) is not str or not name or type(raw) is not bytes
                          for name, raw in sources.items()):
        raise ValueError("An interpreter needs a named source-byte manifest")
    return digest({name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()})


class SharedOwnerRef(nn.Module):
    """No registered parameters; copying this reference does not clone its owner."""

    def __init__(self, owner, *, solver=None):
        super().__init__()
        if not isinstance(owner, SharedR1):
            raise ValueError("An executable view requires the existing SharedR1 owner")
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_solver", None if solver is None else weakref.ref(solver))
        self.require_current()

    @classmethod
    def from_solver(cls, solver):
        solver.validate()
        if "r1" not in solver.components:
            raise ValueError("Solver has no registered shared owner")
        return cls(solver.components["r1"], solver=solver)

    @property
    def owner(self):
        self.require_current()
        return self._owner

    def require_current(self, registered_owner=None):
        if registered_owner is not None and registered_owner is not self._owner:
            raise ValueError("Executable view belongs to another parameter owner")
        if self._solver is not None:
            solver = self._solver()
            if (solver is None or "r1" not in solver.components
                    or solver.components["r1"] is not self._owner):
                raise ValueError("Registered shared owner changed; explicitly rebind the view")
        return self._owner

    @property
    def identity(self):
        owner = self.require_current()
        # Reuse SERA's identity format without its tensor-version cache. A raw
        # tensor data write can change bytes without incrementing that version.
        inputs = SimpleNamespace(export_config=owner.export_config, state_dict=owner.state_dict)
        return model_identity(inputs)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


@dataclass(frozen=True, slots=True)
class DependencyRef:
    name: str
    sha256: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise ValueError("A dependency requires a stable name")
        _hash(self.sha256)

    def record(self):
        return {"name": self.name, "sha256": self.sha256}


def _dependencies(values):
    values = tuple(values)
    if any(type(value) is not DependencyRef for value in values):
        raise ValueError("Dependencies require immutable content references")
    if len({value.name for value in values}) != len(values):
        raise ValueError("Duplicate dependency name")
    return tuple(sorted(values, key=lambda value: value.name))


@dataclass(frozen=True, slots=True)
class ExecutableDefinition:
    name: str
    body_json: str
    schema_sha256: str
    interpreter_sha256: str
    dependencies: tuple[DependencyRef, ...] = ()

    def __post_init__(self):
        if type(self.name) is not str or not self.name or type(self.body_json) is not str:
            raise ValueError("Definition requires a name and a canonical JSON body")
        _hash(self.schema_sha256)
        _hash(self.interpreter_sha256)
        body = canonical(json.loads(self.body_json))
        if len(body.encode()) > 1024 * 1024:
            raise ValueError("Executable definition exceeds one MiB")
        object.__setattr__(self, "body_json", body)
        object.__setattr__(self, "dependencies", _dependencies(self.dependencies))

    @classmethod
    def build(cls, name, body, schema, interpreter, *, dependencies=()):
        return cls(name, canonical(body), schema.identity, interpreter, dependencies)

    @property
    def body(self):
        return json.loads(self.body_json)

    def record(self):
        return {"name": self.name, "body": self.body,
                "schema_sha256": self.schema_sha256,
                "interpreter_sha256": self.interpreter_sha256,
                "dependencies": [value.record() for value in self.dependencies]}

    @property
    def identity(self):
        return digest(self.record())

    @property
    def reference(self):
        return DependencyRef(self.name, self.identity)

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {
            "name", "body", "schema_sha256", "interpreter_sha256", "dependencies"
        }:
            raise ValueError("Invalid executable definition record")
        return cls(record["name"], canonical(record["body"]), record["schema_sha256"],
                   record["interpreter_sha256"],
                   tuple(DependencyRef(**row) for row in record["dependencies"]))


@dataclass(frozen=True, slots=True)
class ExecutableGraph:
    schema: EventSchema
    interpreter_sha256: str
    definitions: tuple[ExecutableDefinition, ...] = ()

    def __post_init__(self):
        if type(self.schema) is not EventSchema:
            raise ValueError("An executable graph requires an event schema")
        _hash(self.interpreter_sha256)
        definitions = tuple(self.definitions)
        if len(definitions) > 4096 or any(type(item) is not ExecutableDefinition
                                        for item in definitions):
            raise ValueError("Invalid or excessive executable definitions")
        table = {item.name: item for item in definitions}
        if len(table) != len(definitions):
            raise ValueError("Duplicate executable name")
        for item in definitions:
            if (item.schema_sha256 != self.schema.identity
                    or item.interpreter_sha256 != self.interpreter_sha256):
                raise ValueError("Executable schema or interpreter fingerprint differs")
            for dependency in item.dependencies:
                if (dependency.name not in table
                        or table[dependency.name].identity != dependency.sha256):
                    raise ValueError("Missing or stale executable dependency")
        pending, complete = set(), set()

        def visit(name):
            if name in pending:
                raise ValueError("Cyclic executable dependency")
            if name in complete:
                return
            if len(pending) >= 64:
                raise ValueError("Executable dependency depth exceeds 64")
            pending.add(name)
            for dependency in table[name].dependencies:
                visit(dependency.name)
            pending.remove(name)
            complete.add(name)

        for name in table:
            visit(name)
        object.__setattr__(self, "definitions", tuple(sorted(definitions, key=lambda item: item.name)))

    def record(self):
        return {"format": "sera-executable-graph-v1", "schema": self.schema.record(),
                "interpreter_sha256": self.interpreter_sha256,
                "definitions": [item.record() for item in self.definitions]}

    @property
    def identity(self):
        return digest(self.record())

    @property
    def dependencies(self):
        return tuple(item.reference for item in self.definitions)

    def definition(self, name):
        for item in self.definitions:
            if item.name == name:
                return item
        raise KeyError(name)

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {
            "format", "schema", "interpreter_sha256", "definitions"
        } or record["format"] != "sera-executable-graph-v1":
            raise ValueError("Invalid executable graph record")
        return cls(EventSchema.from_record(record["schema"]), record["interpreter_sha256"],
                   tuple(ExecutableDefinition.from_record(row) for row in record["definitions"]))


@dataclass(frozen=True, slots=True)
class MechanismRef:
    graph_sha256: str
    parameter_sha256: str
    schema_sha256: str
    interpreter_sha256: str
    dependencies: tuple[DependencyRef, ...]
    owner_name: str = "r1"

    def __post_init__(self):
        for value in (self.graph_sha256, self.parameter_sha256,
                      self.schema_sha256, self.interpreter_sha256):
            _hash(value)
        if self.owner_name != "r1":
            raise ValueError("Executable mechanisms require the registered r1 owner")
        object.__setattr__(self, "dependencies", _dependencies(self.dependencies))

    @classmethod
    def bind(cls, graph, owner):
        return cls(graph.identity, owner.identity, graph.schema.identity,
                   graph.interpreter_sha256, graph.dependencies)

    def validate(self, graph, owner):
        if self != self.bind(graph, owner):
            raise ValueError("Mechanism graph, parameters, schema or interpreter changed")

    def record(self):
        return {"graph_sha256": self.graph_sha256, "parameter_sha256": self.parameter_sha256,
                "schema_sha256": self.schema_sha256,
                "interpreter_sha256": self.interpreter_sha256, "owner_name": self.owner_name,
                "dependencies": [value.record() for value in self.dependencies]}

    @property
    def identity(self):
        return digest(self.record())


@dataclass(frozen=True, slots=True)
class Replacement:
    parent_sha256: str
    graph: ExecutableGraph
    changed: tuple[str, ...]
    invalidated: tuple[str, ...]


class GraphStore:
    """One in-memory atomic pointer; admission is a caller-supplied validator."""

    def __init__(self, graph):
        if type(graph) is not ExecutableGraph:
            raise ValueError("GraphStore requires a validated immutable graph")
        self._current = graph
        self._lock = threading.RLock()

    @property
    def current(self):
        return self._current

    def replace(self, replacements=(), *, remove=(), expected_parent, validator=None):
        with self._lock:
            parent = self.current
            if parent.identity != expected_parent:
                raise ValueError("Executable parent changed before replacement")
            replacements = tuple(replacements)
            if any(type(item) is not ExecutableDefinition for item in replacements):
                raise ValueError("Replacement must supply immutable executable definitions")
            proposed = {item.name: item for item in replacements}
            if len(proposed) != len(replacements):
                raise ValueError("Duplicate replacement name")
            old = {item.name: item for item in parent.definitions}
            remove = set(remove)
            if not remove.issubset(old) or remove & set(proposed):
                raise ValueError("Invalid or contradictory removal request")
            changed = remove | {name for name, item in proposed.items() if old.get(name) != item}
            affected = set(changed)
            while True:
                expanded = affected | {item.name for item in parent.definitions
                                       if any(ref.name in affected for ref in item.dependencies)}
                if expanded == affected:
                    break
                affected = expanded
            # Dependents are removed unless explicitly rebuilt and revalidated.
            retained = {name: item for name, item in old.items() if name not in affected}
            retained.update(proposed)
            candidate = ExecutableGraph(parent.schema, parent.interpreter_sha256,
                                        tuple(retained.values()))
            if validator is not None and validator(candidate) is not True:
                raise ValueError("Executable admission rejected the candidate")
            if self.current.identity != expected_parent:
                raise ValueError("Executable parent changed during validation")
            result = Replacement(expected_parent, candidate, tuple(sorted(changed)),
                                 tuple(sorted(affected - set(proposed))))
            self._current = candidate
            return result


class ExecutionBudget:
    """Branches share a nonnegative operation budget; bytes are a separate metric."""

    def __init__(self, limit=100_000):
        if type(limit) is not int or limit < 0:
            raise ValueError("Execution limit must be a nonnegative integer")
        self.limit = limit
        self._counts = {}

    @property
    def counts(self):
        return dict(self._counts)

    def add(self, kind, count=1):
        if type(kind) is not str or not kind or type(count) is not int or count < 0:
            raise ValueError("Execution accounting requires named nonnegative integer operations")
        self._counts[kind] = self._counts.get(kind, 0) + count
        if sum(self._counts.values()) > self.limit:
            raise InterruptedError("Shared execution budget exhausted")

    def record(self):
        return {"limit": self.limit, "counts": self.counts}

    @classmethod
    def restore(cls, record):
        if type(record) is not dict or set(record) != {"limit", "counts"}:
            raise ValueError("Invalid execution budget record")
        result = cls(record["limit"])
        if type(record["counts"]) is not dict or any(
            type(key) is not str or not key or type(value) is not int or value < 0
            for key, value in record["counts"].items()
        ):
            raise ValueError("Invalid persisted execution counts")
        result._counts = dict(record["counts"])
        return result


class LiveSituation:
    """A bounded factual event record bound to a complete executable snapshot.

    It stores admitted observations for explicit replay, not a learned latent
    representation. Persisting a record does not independently verify its truth.
    """

    def __init__(self, situation_id, graph, owner, *, budget=None, max_events=4096):
        if type(situation_id) is not str or not situation_id:
            raise ValueError("Situation identity is required")
        if type(max_events) is not int or not 1 <= max_events <= 4096:
            raise ValueError("Invalid factual evidence capacity")
        if type(graph) is not ExecutableGraph or not isinstance(owner, SharedOwnerRef):
            raise ValueError("Situation needs an immutable graph and shared owner reference")
        self.situation_id, self.graph, self.owner = situation_id, graph, owner
        self.mechanism = MechanismRef.bind(graph, owner)
        self.budget = ExecutionBudget() if budget is None else budget
        if not isinstance(self.budget, ExecutionBudget):
            raise ValueError("Situation needs a shared execution budget")
        self.max_events = max_events
        self._observations, self._labels = (), ()
        self.migration = None

    @property
    def observations(self):
        return self._observations

    @property
    def labels(self):
        return self._labels

    def _guard(self):
        self.mechanism.validate(self.graph, self.owner)

    def _check_event(self, event):
        self._guard()
        if type(event) is not Event or event.schema != self.graph.schema:
            raise ValueError("Event schema, units or encoding differ from the situation")
        if len(self.observations) + len(self.labels) >= self.max_events:
            raise ValueError("Factual evidence capacity exhausted")
        if event.evidence_key in {row.evidence_key for row in self.observations + self.labels}:
            raise ValueError("Evidence record was already admitted")

    def observe(self, event):
        self._admit(event, label=False, replay=False)

    def admit_label(self, event):
        self._admit(event, label=True, replay=False)

    def _admit(self, event, *, label, replay):
        """Replay pays processing costs without acquiring the same fact again."""
        self._check_event(event)
        if label:
            if not event.eligible_label or event.sequence >= len(self.observations):
                raise ValueError("Labels need independent eligible evidence and an observed position")
            if not all(event.available):
                raise ValueError("An admitted target must be fully available")
            self.budget.add("replayed_labels" if replay else "verified_labels")
            self._labels += (event,)
        else:
            if not event.eligible_observation or event.sequence != len(self.observations):
                raise ValueError("Only factual observations in sequence may update the live situation")
            self.budget.add("replayed_observations" if replay else "factual_observations")
            self._observations += (event,)

    def fork(self):
        self._guard()
        self.budget.add("hypothetical_branches")
        return HypotheticalSituation(self.situation_id, self.graph, self.owner, self.mechanism,
                                     self.observations, self.budget, self.max_events)

    def snapshot(self):
        self._guard()
        payload = {
            "format": "sera-factual-situation-v1", "situation_id": self.situation_id,
            "mechanism": self.mechanism.record(), "max_events": self.max_events,
            "observations": [event.record() for event in self.observations],
            "labels": [event.record() for event in self.labels], "migration": self.migration,
        }
        # Round-trip detaches every exposed container, including migration metadata.
        payload = json.loads(canonical(payload))
        budget = self.budget.record()
        return {"payload": payload, "sha256": digest(payload),
                "budget": budget, "budget_sha256": digest(budget)}

    @classmethod
    def restore(cls, record, graph, owner, *, situation_id, budget=None):
        if type(record) is not dict or set(record) != {
            "payload", "sha256", "budget", "budget_sha256"
        }:
            raise ValueError("Invalid factual snapshot envelope")
        payload = record["payload"]
        if type(payload) is not dict or set(payload) != {
            "format", "situation_id", "mechanism", "max_events", "observations", "labels", "migration"
        } or digest(payload) != record["sha256"]:
            raise ValueError("Factual snapshot integrity or schema failure")
        if (payload["format"] != "sera-factual-situation-v1"
                or payload["situation_id"] != situation_id
                or payload["mechanism"] != MechanismRef.bind(graph, owner).record()):
            raise ValueError("Snapshot situation or complete mechanism identity differs")
        if digest(record["budget"]) != record["budget_sha256"]:
            raise ValueError("Snapshot budget integrity failure")
        restored_budget = ExecutionBudget.restore(record["budget"])
        if budget is not None and budget.record() != restored_budget.record():
            raise ValueError("Resume budget differs from the recorded ledger")
        budget = restored_budget if budget is None else budget
        result = cls(situation_id, graph, owner, budget=budget, max_events=payload["max_events"])
        for row in payload["observations"]:
            result._admit(Event.from_record(row), label=False, replay=True)
        for row in payload["labels"]:
            result._admit(Event.from_record(row), label=True, replay=True)
        result.migration = json.loads(canonical(payload["migration"]))
        return result

    def migrate(self, graph, owner, *, expected_mechanism):
        self._guard()
        if expected_mechanism != self.mechanism.identity:
            raise ValueError("Situation migration parent differs")
        if graph.schema != self.graph.schema:
            raise ValueError("Incompatible schema requires an independently validated migration")
        result = LiveSituation(self.situation_id, graph, owner, budget=self.budget,
                               max_events=self.max_events)
        for event in self.observations:
            result._admit(event, label=False, replay=True)
        for event in self.labels:
            result._admit(event, label=True, replay=True)
        result.migration = {
            "method": "same-schema-event-replay-v1", "parent": self.mechanism.identity,
            "successor": result.mechanism.identity,
            "observations_sha256": digest([event.record() for event in self.observations]),
            "observation_count": len(self.observations), "label_count": len(self.labels),
        }
        return result


class HypotheticalSituation:
    """Immutable factual prefix plus separate conditional assumptions; no admission."""

    def __init__(self, situation_id, graph, owner, mechanism, observations, budget, max_events):
        self.situation_id, self.graph, self.owner = situation_id, graph, owner
        self.mechanism, self.observations = mechanism, tuple(observations)
        self.budget, self.max_events = budget, max_events
        self._assumptions = ()

    @property
    def assumptions(self):
        return self._assumptions

    def assume(self, event):
        self.mechanism.validate(self.graph, self.owner)
        if (type(event) is not Event or event.schema != self.graph.schema
                or event.role != EventRole.HYPOTHETICAL
                or event.origin != EvidenceKind.PREDICTION):
            raise ValueError("Hypothetical events require conditional prediction provenance")
        if event.sequence != len(self.observations) + len(self.assumptions):
            raise ValueError("Hypothetical events must preserve sequence order")
        if len(self.observations) + len(self.assumptions) >= self.max_events:
            raise ValueError("Hypothetical evidence capacity exhausted")
        self.budget.add("hypothetical_assumptions")
        self._assumptions += (event,)

    def observe(self, event):
        raise ValueError("Hypothetical branches cannot admit factual observations")

    def admit_label(self, event):
        raise ValueError("Hypothetical branches cannot admit learning labels")

    def snapshot(self):
        self.mechanism.validate(self.graph, self.owner)
        payload = {"format": "sera-hypothetical-situation-v1", "situation_id": self.situation_id,
                   "mechanism": self.mechanism.record(),
                   "observations": [event.record() for event in self.observations],
                   "assumptions": [event.record() for event in self.assumptions]}
        return {"payload": payload, "sha256": digest(payload)}
