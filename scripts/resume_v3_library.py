"""Read-only use of the repaired library and its exact existing trained parent."""

import argparse
import json
from pathlib import Path

from experiments.guarded_consolidation.core import Library, Work
from experiments.guarded_consolidation.study import PARENT, ReadOnlyStore
from sera.world_graph import SharedOwnerRef

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    for name in ("a", "b", "c"):
        p.add_argument("--"+name, type=int, required=True)
    args = p.parse_args()
    solver = ReadOnlyStore(PARENT).load(version="v0")
    parent = solver.identity()
    record = json.loads((ROOT/"research-continuation/16_v3/GC-002/repair/corrected-library.json").read_text())
    work = Work()
    library = Library.restore(record, SharedOwnerRef.from_solver(solver), work, proof_backend="indexed")
    relation = library.store.current.definition("affine")
    answer = library.answer(relation.identity, vars(args), work)
    if solver.identity() != parent:
        raise ValueError("Read-only use changed the original owner")
    print(json.dumps({"answer": answer, "status": "CONDITIONAL_FINITE_PROOF" if answer is not None else "UNSUPPORTED",
                      "relation": "a*x-b=c modulo 11", "assumptions": "integer field values in [0,10], a nonzero",
                      "owner_sha256": library.owner_id, "library_sha256": library.store.current.identity,
                      "restore_and_execution_work": work.counts}, indent=2))


if __name__ == "__main__":
    main()
