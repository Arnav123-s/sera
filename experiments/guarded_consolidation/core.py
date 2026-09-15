"""Parameterize verified reasoning traces; retain inverse preconditions and dependencies.

Finite-field semantics, equation representation and algebraic rewrite rules are
supplied. The acquired object is an executable abstraction, not a discovered
field theory or a learned learning algorithm. No assessment answer cache exists.
"""

import hashlib
import itertools
import json
import platform
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from sera.event_ir import EventSchema
from sera.storage import canonical, digest
from sera.world_graph import ExecutableDefinition, ExecutableGraph, GraphStore

P = 11
SCHEMA = EventSchema("finite-equation", 1, "symbolic", 3, "F11", encoding="formal-roles-v1")


def interpreter_identity():
    root = Path(__file__).resolve().parents[2]
    names = [p.relative_to(root).as_posix() for p in (root/"src/sera").glob("*.py")]
    names += ["experiments/guarded_consolidation/"+name for name in ("core.py", "indexed_proof.py", "migration.py")]
    return digest({"sources": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)},
                   "runtime": {"python": platform.python_version(), "torch": torch.__version__}})


class Work:
    def __init__(self):
        self.counts = Counter()

    def add(self, kind, n=1):
        self.counts[kind] += n

    @property
    def operations(self):
        return sum(v for k, v in self.counts.items() if not k.endswith("_bytes"))


def evaluate(tree, inputs, work):
    work.add("expression_nodes")
    if tree[0] == "input":
        return inputs[tree[1]]
    if tree[0] == "value":
        return tree[1]
    args = [evaluate(t, inputs, work) for t in tree[1:]]
    if tree[0] == "add":
        return (args[0]+args[1]) % P
    if tree[0] == "sub":
        return (args[0]-args[1]) % P
    if tree[0] == "mul":
        return args[0]*args[1] % P
    if tree[0] == "inv":
        # Totalized primitive: zero maps to zero. Its mathematical inverse
        # interpretation requires the separate nonzero guard below.
        return pow(args[0], P-2, P)
    raise ValueError("Unknown finite primitive")


def isolate(left, right, inputs, work):
    """Supplied local reasoning rules produce a concrete provenance-bearing trace."""
    work.add("rewrite_visits")
    if left == "x":
        return right
    if not isinstance(left, (list, tuple)) or len(left) != 3:
        raise ValueError("Unsupported equation representation")
    operation, child, known = left
    if known not in inputs:
        raise ValueError("Missing formal role")
    leaf = ["value", inputs[known], known]
    work.add("trace_nodes_created", 1)
    if operation in ("add", "sub"):
        result = ["sub" if operation == "add" else "add", right, leaf]
        work.add("trace_nodes_created")
    elif operation == "mul":
        result = ["mul", right, ["inv", leaf]]
        work.add("trace_nodes_created", 2)
    else:
        raise ValueError("No inverse rewrite rule")
    return isolate(child, result, inputs, work)


def inverse_arguments(tree):
    if tree[0] in ("input", "value"):
        return []
    return ([tree[1]] if tree[0] == "inv" else []) + [g for t in tree[1:] for g in inverse_arguments(t)]


def guard(tree, inputs, work):
    work.add("domain_checks", 3)
    if set(inputs) != {"a", "b", "c"} or any(type(x) is not int or not 0 <= x < P for x in inputs.values()):
        return False
    for expression in inverse_arguments(tree):
        work.add("nonzero_checks")
        if evaluate(expression, inputs, work) == 0:
            return False
    return True


def forward(left, inputs, x, work):
    work.add("forward_nodes")
    if left == "x":
        return x
    operation, child, parameter = left
    value = forward(child, inputs, x, work)
    if operation == "add":
        return (value+inputs[parameter]) % P
    if operation == "sub":
        return (value-inputs[parameter]) % P
    if operation == "mul":
        return value*inputs[parameter] % P
    raise ValueError("Invalid observation mechanism")


def observe_solutions(left, inputs, work):
    # Reference sensor exhaustively evaluates the forward relation, independently
    # of inverse reasoning or its compiled program.
    return tuple(x for x in range(P) if forward(left, inputs, x, work) == inputs["c"])


def local_reason(left, inputs, work):
    work.add("trace_nodes_created")
    concrete = isolate(left, ["value", inputs["c"], "c"], inputs, work)
    return evaluate(concrete, inputs, work) if guard(concrete, inputs, work) else None


def teaching_trace(left, inputs, work, record_id):
    concrete = isolate(left, ["value", inputs["c"], "c"], inputs, work)
    if not guard(concrete, inputs, work):
        raise ValueError("A successful trace requires its assumptions")
    prediction = evaluate(concrete, inputs, work)
    answers = observe_solutions(left, inputs, work)
    if answers != (prediction,):
        raise ValueError("Reasoning trace failed independent forward verification")
    return {"inputs": dict(inputs), "trace": concrete, "observed_solutions": list(answers),
            "record_id": record_id, "evidence": "finite_simulator_observation"}


def abstract(traces, work):
    if len(traces) < 3 or len({r["record_id"] for r in traces}) != len(traces):
        raise ValueError("Distinct successful traces are required")
    if any(r["evidence"] != "finite_simulator_observation" for r in traces):
        raise ValueError("Imagined answers cannot become verified teaching")

    def walk(nodes):
        work.add("abstraction_nodes", len(nodes))
        if any(n[0] != nodes[0][0] or len(n) != len(nodes[0]) for n in nodes):
            raise ValueError("Traces have incompatible structures")
        if nodes[0][0] == "value":
            origins = {n[2] for n in nodes}
            if len(origins) != 1:
                raise ValueError("Value provenance differs")
            origin = next(iter(origins))
            if len({n[1] for n in nodes}) < 2 or any(n[1] != r["inputs"][origin] for n, r in zip(nodes, traces)):
                raise ValueError("Incidental values not independently varied or correctly bound")
            return ["input", origin]
        return [nodes[0][0]]+[walk([n[i] for n in nodes]) for i in range(1, len(nodes[0]))]

    return walk([r["trace"] for r in traces])


def prove(tree, left, work):
    accepted = 0
    for a, b, c in itertools.product(range(P), repeat=3):
        inputs = dict(a=a, b=b, c=c)
        if guard(tree, inputs, work):
            answer = evaluate(tree, inputs, work)
            if observe_solutions(left, inputs, work) != (answer,):
                raise ValueError("Finite proof found an incorrect or ambiguous answer")
            accepted += 1
    if not accepted:
        raise ValueError("Vacuous all-abstain program")
    return {"domain": "F11^3", "checked_inputs": P**3, "accepted_inputs": accepted,
            "scope": "Exhaustive correctness inside supplied finite semantics and preconditions only"}


def proof_for_backend(tree, left, work, backend):
    if backend == "exhaustive":
        return prove(tree, left, work)
    if backend == "indexed":
        from .indexed_proof import prove_indexed
        return prove_indexed(tree, left, work)
    raise ValueError("Unknown finite proof backend")


def compile_definition(base, traces, schema, interpreter, work, *, proof_backend="exhaustive"):
    tree = abstract(traces, work)
    for r in traces:
        if observe_solutions(base.body["left"], r["inputs"], work) != tuple(r["observed_solutions"]):
            raise ValueError("Forged teaching outcome")
    proof = proof_for_backend(tree, base.body["left"], work, proof_backend)
    body = {"kind": "guarded_program", "input_types": {x: "F11" for x in ("a", "b", "c")},
            "output_type": "unique F11 solution or unsupported", "program": tree,
            "nonzero_conditions": inverse_arguments(tree), "proof": proof,
            "teaching_sha256": digest(traces), "semantics": base.identity,
            "evidence_status": "exhaustively_verified_finite_domain"}
    return ExecutableDefinition.build("compiled/"+base.name, body, schema, interpreter,
                                      dependencies=(base.reference,))


class Library:
    """One owner's graph with transactional dependency invalidation and paid lookup.

    Batch callers check the owner before and after a synchronous read-only batch.
    No simultaneous owner writer is supported by this experimental facade.
    """
    def __init__(self, store, owner, work):
        self.store, self.owner = store, owner
        if store.current.interpreter_sha256 != interpreter_identity():
            raise ValueError("Stale executable interpreter; explicit verified migration required")
        self.owner_id = owner.identity
        self._in_batch = False
        self._graph = None
        self._graph_object = None
        self._refresh(work)

    def require_owner(self, work):
        work.add("owner_identity_checks")
        work.add("interpreter_identity_checks")
        if self.store.current.interpreter_sha256 != interpreter_identity():
            raise ValueError("Executable interpreter changed")
        if self.owner.identity != self.owner_id:
            raise ValueError("Owner changed; reacquisition/revalidation required")

    @contextmanager
    def batch(self, work):
        if self._in_batch:
            raise ValueError("Nested execution batches are unsupported")
        self.require_owner(work)
        self._in_batch = True
        try:
            yield self
        finally:
            self._in_batch = False
            self.require_owner(work)

    def _refresh(self, work):
        if self.store.current is self._graph_object:
            return
        identity = self.store.current.identity
        work.add("graph_validation_bytes", len(canonical(self.store.current.record()).encode()))
        self._items, self._index = [], {}
        for definition in self.store.current.definitions:
            work.add("library_index_entries")
            body = definition.body
            if body.get("kind") == "guarded_program":
                self._items.append(body)
                if body["semantics"] in self._index:
                    raise ValueError("Ambiguous compiled semantic key")
                self._index[body["semantics"]] = body
        self._graph = identity
        self._graph_object = self.store.current

    def answer(self, semantic_id, inputs, work, *, indexed=True, omit_guard=False):
        if not self._in_batch:
            self.require_owner(work)
        self._refresh(work)
        if indexed:
            work.add("indexed_lookup")
            work.add("lookup_key_bytes", len(semantic_id))
            body = self._index.get(semantic_id)
        else:
            body = None
            for candidate in self._items:
                work.add("linear_key_comparisons")
                work.add("lookup_key_bytes", len(semantic_id))
                if candidate["semantics"] == semantic_id:
                    body = candidate
                    break
        if body is None:
            return None
        tree = body["program"]
        return evaluate(tree, inputs, work) if omit_guard or guard(tree, inputs, work) else None

    def record(self):
        return {"owner_sha256": self.owner_id, "graph": self.store.current.record(),
                "graph_sha256": self.store.current.identity}

    @classmethod
    def restore(cls, record, owner, work, *, proof_backend="exhaustive"):
        graph = ExecutableGraph.from_record(record["graph"])
        if graph.identity != record["graph_sha256"] or owner.identity != record["owner_sha256"]:
            raise ValueError("Stale persisted library")
        result = cls(GraphStore(graph), owner, work)
        # Re-prove every admitted executable from its declared current semantics.
        for item in graph.definitions:
            body = item.body
            if body.get("kind") == "guarded_program":
                if len(item.dependencies) != 1:
                    raise ValueError("Missing executable dependency")
                base = graph.definition(item.dependencies[0].name)
                if body["semantics"] != base.identity or body["nonzero_conditions"] != inverse_arguments(body["program"]):
                    raise ValueError("Forged executable guard/semantic binding")
                if proof_for_backend(body["program"], base.body["left"], work, proof_backend) != body["proof"]:
                    raise ValueError("Invalid executable proof")
        return result


def archive_bytes(library):
    return len(canonical(library.record()).encode()), len(json.dumps(library._index).encode())
