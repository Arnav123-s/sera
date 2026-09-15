"""Read-only use of the qualified experimental owner, factual state and library."""

import argparse
import json
import math
from pathlib import Path

import torch

from experiments.guarded_consolidation.core import Library, Work
from experiments.guarded_consolidation.study import ReadOnlyStore
from sera.contracts import EvidenceKind, Observation, Provenance
from sera.generative import SharedGenerativeSession
from sera.world_graph import SharedOwnerRef

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/17_transfer/integration"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("status", "motion", "finite"))
    parser.add_argument("--coordinates", type=float, nargs=6)
    parser.add_argument("--a", type=int)
    parser.add_argument("--b", type=int)
    parser.add_argument("--c", type=int)
    args = parser.parse_args()
    record = read(RELEASE/"result.json")
    solver = ReadOnlyStore(ROOT/record["store"]).load()
    identity = solver.identity()
    if identity != record["solver_record"]["solver_sha256"]:
        raise ValueError("Experimental solver identity changed")
    session = SharedGenerativeSession.restore(solver, read(RELEASE/"factual-situation.json"))
    work = Work()
    library = Library.restore(read(RELEASE/"finite-library.json"), SharedOwnerRef.from_solver(solver), work, proof_backend="indexed")
    response = {"solver_sha256": identity, "status": "EXPERIMENTAL_QUALIFIED_COMPONENT",
                "scope": "One simulated source instance and a supplied circle family; no general answer certificate",
                "factual_observations": len(session.situation.observations),
                "factual_corrections": len(session.situation.labels)}
    if args.mode == "motion":
        if args.coordinates is None or not all(map(math.isfinite, args.coordinates)):
            parser.error("motion requires six finite coordinates for three equally spaced observations")
        provenance = Provenance("user-supplied-motion", "read-only-query", EvidenceKind.OBSERVATION)
        observations = tuple(Observation("numeric", tuple(args.coordinates[2*i:2*i+2]), i, provenance, units="m") for i in range(3))
        response["prediction"] = solver.components["typed"].predict(observations, "motion")
        response["answer_status"] = "CONDITIONAL_UNCALIBRATED_NEURAL_PREDICTION"
    elif args.mode == "finite":
        if any(value is None for value in (args.a, args.b, args.c)):
            parser.error("finite requires --a, --b and --c")
        relation = library.store.current.definition("affine")
        value = library.answer(relation.identity, {"a": args.a, "b": args.b, "c": args.c}, work)
        response.update(answer=value, relation="a*x-b=c modulo 11", assumptions="a nonzero; all inputs in F11",
                        answer_status="CONDITIONAL_FINITE_PROOF" if value is not None else "UNSUPPORTED")
    response["read_only_restore_and_proof_work"] = work.counts
    if solver.identity() != identity:
        raise ValueError("Read-only use changed the owner")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
