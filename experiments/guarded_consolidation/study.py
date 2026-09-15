"""GC-001: finite trace abstraction, prospective compositions and online correction."""

import argparse
import gzip
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from sera.solver import SolverStore
from sera.storage import canonical, digest
from sera.world_graph import ExecutableDefinition, ExecutableGraph, GraphStore, SharedOwnerRef

from .core import (
    SCHEMA,
    Library,
    P,
    Work,
    archive_bytes,
    compile_definition,
    interpreter_identity,
    local_reason,
    observe_solutions,
    teaching_trace,
)

ROOT = Path(__file__).resolve().parents[2]
RELEASE_BASE = ROOT/"research-continuation/16_v3"
PARENT = ROOT/"runs/SHARED-GG-001/circle/corrected"
METHODS = ("repeated_search", "local_reasoning", "compiled_linear", "compiled_indexed", "omit_guard")
TRAIN_INPUTS = [dict(a=a, b=b, c=c) for a, b, c in ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (2, 1, 3), (3, 2, 1))]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sources():
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/"src/sera").glob("*.py")]
    names += ["experiments/guarded_consolidation/"+n for n in ("__init__.py", "core.py", "indexed_proof.py", "study.py")]
    names += ["tests/test_guarded_consolidation.py"]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}


def release(backend):
    return RELEASE_BASE/("GC-001" if backend == "exhaustive" else "GC-002")


def freeze(backend):
    directory = release(backend)
    directory.mkdir(parents=True, exist_ok=False)
    pointer = json.loads((PARENT/"current.json").read_text())
    p = {"experiment": directory.name, "sources": sources(), "field": P, "proof_backend": backend,
         "parent": PARENT.relative_to(ROOT).as_posix(), "parent_manifest": pointer,
         "training_inputs": TRAIN_INPUTS, "assessment_seed": 90139001 if backend == "exhaustive" else 90239001,
         "counts": {"renamed_new_values": 1024, "withheld_compositions": 8192, "violated_assumptions": 512,
                    "after_correction": 1024, "library_growth": 512},
         "library_sizes": [0, 32, 128], "methods": METHODS,
         "hypotheses": ["Trace abstraction saves repeated reasoning on fresh bindings/compositions after full cost",
                        "Lookup/proof/correction overhead can erase nominal shortcut savings"],
         "guards": "Finite F11 values, inverse operands nonzero, complete interpreter and transitive semantic dependencies",
         "gate": "Zero guarded false answers; all unique in-domain evaluations answered; correct transitive invalidation with unrelated retention; indexed full-cost operation proxy < local reasoning and repeated search over the predefined stream. Wall times reported separately; a proxy win is not a hardware or neural-transfer claim.",
         "scope": "Supplied formal language, finite field and algebra rules. Acquired generalized trace program. Exact proof includes all finite atomic inputs, so this is not unseen-world empirical generalization; unseen compositions/bindings were absent from teaching.",
         "budget": {"job_seconds": 600, "memory_bytes": 2147483648, "threads": 1},
         "knowledge_vs_eta": "K changes through trace consolidation and correction. eta is fixed externally engineered code."}
    write(directory/"protocol.json", {"payload": p, "sha256": digest(p)})
    with zipfile.ZipFile(directory/"frozen-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in p["sources"]:
            z.write(ROOT/name, name)
    print(digest(p))


class ReadOnlyStore(SolverStore):
    def __init__(self, root):
        # Inherited load/current_record/record only read manifests/checkpoints.
        # Avoid opening or modifying the original store's journal.
        self.root = root


def base(name, interpreter, corrected=False):
    left = ["sub" if corrected else "add", ["mul", "x", "a"], "b"] if name == "affine" else ["add", "x", "b"]
    return ExecutableDefinition.build(name, {"kind": "supplied_finite_relation", "left": left,
                                            "field": P, "state": "current equation; supplied semantics"}, SCHEMA, interpreter)


def acquire(definition, interpreter, work, phase, backend):
    traces = [teaching_trace(definition.body["left"], inputs, work, f"{phase}/{definition.name}/{i}")
              for i, inputs in enumerate(TRAIN_INPUTS)]
    return compile_definition(definition, traces, SCHEMA, interpreter, work, proof_backend=backend), traces


def tasks(rng, panel, n):
    result = []
    for i in range(n):
        length = 3 if panel == "withheld_compositions" else 1
        chain = []
        for j in range(length):
            a = 0 if panel == "violated_assumptions" else int(rng.integers(4, P))
            b, c = (int(x) for x in rng.integers(4, P, 2))
            # Role names are renamed, but the formal binding map is supplied.
            names = [f"role_{i}_{j}_{k}" for k in range(3)]
            chain.append({"values": dict(zip(names, (a, b, c))), "bindings": dict(zip(("a", "b", "c"), names))})
        result.append({"id": f"{panel}/{i}", "panel": panel, "chain": chain})
    return result


def bindings(item, previous, work):
    work.add("formal_role_bindings", 3)
    inputs = {k: item["values"][name] for k, name in item["bindings"].items()}
    if previous is not None:
        inputs["c"] = previous
    return inputs


def expected_chain(task, definition, work):
    previous = None
    for item in task["chain"]:
        values = bindings(item, previous, work)
        solutions = observe_solutions(definition.body["left"], values, work)
        if len(solutions) != 1:
            return None
        previous = solutions[0]
    return previous


def evaluate(cohort, definition, library):
    truth_work, output = Work(), {"tasks": cohort, "methods": {}}
    truth = [expected_chain(t, definition, truth_work) for t in cohort]
    for method in METHODS:
        work, start = Work(), time.perf_counter()
        answers = []
        semantic = definition.identity
        with library.batch(work):
            for task in cohort:
                previous = None
                for item in task["chain"]:
                    inputs = bindings(item, previous, work)
                    if method == "repeated_search":
                        candidates = observe_solutions(definition.body["left"], inputs, work)
                        answer = candidates[0] if len(candidates) == 1 else None
                    elif method == "local_reasoning":
                        answer = local_reason(definition.body["left"], inputs, work)
                    else:
                        answer = library.answer(semantic, inputs, work, indexed=method != "compiled_linear", omit_guard=method == "omit_guard")
                    if answer is None:
                        previous = None
                        break
                    previous = answer
                answers.append(previous)
        elapsed = time.perf_counter()-start
        output["methods"][method] = {"answers": answers, "counts": dict(work.counts), "operations": work.operations,
                                     "wall_seconds": elapsed,
                                     "answered": sum(a is not None for a in answers),
                                     "wrong_answers": sum(a is not None and a != b for a, b in zip(answers, truth)),
                                     "missed_unique_solutions": sum(a is None and b is not None for a, b in zip(answers, truth))}
    output["assessor_truth"], output["assessor_work"] = truth, dict(truth_work.counts)
    return output


def run(output, backend):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    protocol = json.loads((release(backend)/"protocol.json").read_text())
    cfg = protocol["payload"]
    if digest(cfg) != protocol["sha256"] or cfg["sources"] != sources():
        raise ValueError("Frozen source/protocol changed")
    write(output/"protocol.json", protocol)
    solver = ReadOnlyStore(PARENT).load()
    if solver.identity() != cfg["parent_manifest"]["solver_sha256"]:
        raise ValueError("Unexpected trained shared parent")
    owner = SharedOwnerRef.from_solver(solver)
    parent_id = solver.identity()
    interpreter = interpreter_identity()
    learning, restore_work = Work(), Work()
    affine, offset = base("affine", interpreter), base("offset", interpreter)
    shortcut, traces = acquire(affine, interpreter, learning, "initial", backend)
    independent, extra_traces = acquire(offset, interpreter, learning, "initial", backend)
    wrapper = ExecutableDefinition.build("compiled/affine_chain", {"kind": "composition", "callee": shortcut.name,
                                                                  "state": "previous conditional solution feeds the next c"},
                                          SCHEMA, interpreter, dependencies=(shortcut.reference,))
    store = GraphStore(ExecutableGraph(SCHEMA, interpreter, (affine, offset, shortcut, independent, wrapper)))
    library = Library(store, owner, learning)
    initial = library.record()
    write(output/"initial-library.json", initial)
    write(output/"teaching.json", traces+extra_traces)
    resumed = Library.restore(initial, owner, restore_work, proof_backend=backend)
    if resumed.record() != initial:
        raise ValueError("Library did not resume exactly")
    rng, panels = np.random.default_rng(cfg["assessment_seed"]), {}
    for panel in ("renamed_new_values", "withheld_compositions", "violated_assumptions"):
        panels[panel] = evaluate(tasks(rng, panel, cfg["counts"][panel]), affine, library)
        print(f"Completed {panel}", flush=True)
    correction_work = Work()
    revised = base("affine", interpreter, corrected=True)
    previous_graph = store.current
    change = store.replace((revised,), expected_parent=previous_graph.identity)
    stale = library.answer(revised.identity, {"a": 4, "b": 5, "c": 6}, correction_work)
    retained = store.current.definition(independent.name).identity == independent.identity
    if stale is not None or not retained or set(change.invalidated) != {shortcut.name, wrapper.name}:
        raise ValueError("Correction did not invalidate exactly the transitive dependent programs")
    repaired, repair_traces = acquire(revised, interpreter, correction_work, "corrected", backend)
    repaired_wrapper = ExecutableDefinition.build(wrapper.name, wrapper.body, SCHEMA, interpreter, dependencies=(repaired.reference,))
    store.replace((repaired, repaired_wrapper), expected_parent=store.current.identity)
    panels["after_correction"] = evaluate(tasks(rng, "after_correction", cfg["counts"]["after_correction"]), revised, library)
    write(output/"corrected-library.json", library.record())
    write(output/"correction-teaching.json", repair_traces)
    # Restore also replays finite proof on the corrected state; preserve its costs.
    restored = Library.restore(library.record(), owner, restore_work, proof_backend=backend)
    if restored.record() != library.record():
        raise ValueError("Corrected state did not resume")
    growth, growth_work = {}, Work()
    growth_tasks = tasks(rng, "library_growth", cfg["counts"]["library_growth"])
    for n in cfg["library_sizes"]:
        definitions = list(store.current.definitions)
        for i in range(n):
            distractor = ExecutableDefinition.build(f"00-context-{i}", offset.body, SCHEMA, interpreter)
            body = independent.body
            body["semantics"] = distractor.identity
            clone = ExecutableDefinition.build(f"00-compiled-{i}", body, SCHEMA, interpreter, dependencies=(distractor.reference,))
            definitions.extend((distractor, clone))
            growth_work.add("context_definition_constructions", 2)
        large = Library(GraphStore(ExecutableGraph(SCHEMA, interpreter, tuple(definitions))), owner, growth_work)
        growth[str(n)] = evaluate(growth_tasks, revised, large)
    learned_bytes, index_bytes = archive_bytes(library)
    totals = {m: sum(p["methods"][m]["operations"] for p in panels.values()) for m in METHODS}
    acquisition_overhead = learning.operations+correction_work.operations+restore_work.operations
    full = {m: value+(acquisition_overhead if m.startswith("compiled") or m == "omit_guard" else 0) for m, value in totals.items()}
    valid = all(p["methods"][m]["wrong_answers"] == 0 and p["methods"][m]["missed_unique_solutions"] == 0
                for p in panels.values() for m in ("compiled_linear", "compiled_indexed"))
    unchanged = solver.identity() == parent_id
    summary = {"protocol_sha256": protocol["sha256"], "panels": {k: {m: {x: y for x, y in v.items() if x != "answers"}
               for m, v in p["methods"].items()} for k, p in panels.items()},
               "full_stream_operations": full, "online_stream_operations": totals,
               "acquisition_verification_restore_overhead": acquisition_overhead,
               "learning_work": dict(learning.counts), "correction_work": dict(correction_work.counts),
               "restore_work": dict(restore_work.counts), "growth_construction_work": dict(growth_work.counts),
               "library_growth": {k: {m: {x: y for x, y in v.items() if x != "answers"} for m, v in p["methods"].items()} for k, p in growth.items()},
               "correction": {"invalidated": list(change.invalidated), "unrelated_retained": retained, "stale_answer": stale},
               "parent_solver_unchanged": unchanged, "shared_owner_sha256": owner.identity,
               "library_record_bytes": learned_bytes, "derived_index_bytes": index_bytes,
               "teaching_observations": len(traces)+len(extra_traces)+len(repair_traces), "new_trainable_parameters": 0,
               "gate": {"correct_guarded_answers": valid, "dependency_correction": retained and stale is None,
                        "full_cost_proxy_gain": full["compiled_indexed"] < min(full["local_reasoning"], full["repeated_search"]),
                        "shared_parent_preserved": unchanged},
               "wall_seconds": time.perf_counter()-started,
               "cost_scope": "Heterogeneous counted operations are a declared work proxy, not FLOPs. Bytes and actual wall times separate. Full stream includes trace observations, abstraction, exhaustive proof, correction and two proof-validating restores. Synthetic library-growth stress and independent final assessor costs are additional separately reported study costs.",
               "limits": "Finite supplied semantics and algebra rules; renamed-role mapping supplied. No neural positive transfer, hidden-state learning, learned latent migration or eta improvement."}
    evidence = {"panels": panels, "library_growth": growth}
    (output/"evidence.json.gz").write_bytes(gzip.compress(canonical(evidence).encode(), mtime=0))
    summary["evidence_sha256"] = hashlib.sha256((output/"evidence.json.gz").read_bytes()).hexdigest()
    write(output/"summary.json", summary)
    print(json.dumps({"gate": summary["gate"], "full_stream_operations": full, "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--variant", choices=("exhaustive", "indexed"), default="exhaustive")
    args = p.parse_args()
    if args.freeze:
        freeze(args.variant)
    else:
        run(args.output, args.variant)
