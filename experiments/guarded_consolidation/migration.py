"""Explicit immutable graph migration, with fresh exhaustive proof under current code."""

from sera.world_graph import ExecutableDefinition, ExecutableGraph, GraphStore

from .core import SCHEMA, Library, interpreter_identity, inverse_arguments, proof_for_backend


def migrate_verified_library(record, owner, work, *, expected_parent):
    old = ExecutableGraph.from_record(record["graph"])
    if old.identity != record["graph_sha256"] or old.identity != expected_parent:
        raise ValueError("Migration parent identity differs")
    if old.schema != SCHEMA or record["owner_sha256"] != owner.identity:
        raise ValueError("Migration requires the same schema and shared parameter owner")
    interpreter = interpreter_identity()
    pending = {d.name: d for d in old.definitions}
    replacements = {}
    while pending:
        progressed = False
        for name, definition in list(pending.items()):
            if not all(d.name in replacements for d in definition.dependencies):
                continue
            body = definition.body
            kind = body.get("kind")
            if kind == "guarded_program":
                if len(definition.dependencies) != 1:
                    raise ValueError("Executable needs one complete semantic dependency")
                predecessor = old.definition(definition.dependencies[0].name)
                successor = replacements[predecessor.name]
                if body["semantics"] != predecessor.identity or body["nonzero_conditions"] != inverse_arguments(body["program"]):
                    raise ValueError("Source semantic or guard binding differs")
                proof = proof_for_backend(body["program"], successor.body["left"], work, "indexed")
                if proof != body["proof"]:
                    raise ValueError("Source proof does not survive current interpreter")
                body["semantics"] = successor.identity
            elif kind == "supplied_finite_relation":
                if body.get("field") != 11:
                    raise ValueError("Field migration is not supported")
            elif kind == "composition":
                if body.get("callee") not in {d.name for d in definition.dependencies}:
                    raise ValueError("Composition lacks its callable dependency")
            else:
                raise ValueError("Unknown executable kind cannot migrate")
            replacements[name] = ExecutableDefinition.build(name, body, SCHEMA, interpreter,
                                                             dependencies=tuple(replacements[d.name].reference for d in definition.dependencies))
            work.add("migrated_definitions")
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError("Incomplete dependency order")
    result = Library(GraphStore(ExecutableGraph(SCHEMA, interpreter, tuple(replacements.values()))), owner, work)
    return result, {"parent_graph": old.identity, "successor_graph": result.store.current.identity,
                    "old_interpreter": old.interpreter_sha256, "new_interpreter": interpreter,
                    "parameter_owner_unchanged": result.owner_id == record["owner_sha256"],
                    "method": "same-schema dependency rebinding plus exhaustive finite reproof; original graph retained"}
