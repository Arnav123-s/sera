"""Integrate only the predefined readout descendant after its prospective gate."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.cross_route_transfer.study import (
    RELEASE,
    ROOT,
    check_contract,
    parent,
    read,
    sha,
    write,
)
from experiments.guarded_consolidation.core import Library, Work
from experiments.guarded_consolidation.study import ReadOnlyStore
from experiments.instance_transfer.study import contract as instance_contract
from sera.generative import SharedGenerativeSession
from sera.shared_archive import load_shared_checkpoint
from sera.solver import SolverStore
from sera.world_graph import SharedOwnerRef

from .migration import migrate


def run():
    normalized = RELEASE/"A06-TRANSFER-002"
    instance = ROOT/"research-continuation/18_instance_transfer"
    normalized_contract = check_contract(normalized)
    normalized_audit = read(normalized/"audit-summary.json")
    raw_contract = instance_contract()["payload"]
    raw_audit = read(instance/"audit-summary.json")
    # This preference/stream choice is declared before instance final data.
    if raw_audit["gate"]["causal_component_pass"]:
        chosen, seed, causal = instance/"seed-307/corrected_teacher", 307, True
        config = raw_contract["config"]
    elif normalized_audit["gate"]["conditional_component_pass"]:
        chosen, seed, causal = normalized/"seed-181/integrated", 181, False
        config = normalized_contract["config"]
    else:
        raise ValueError("Neither prospective component gate passed; no integration authorized")
    if config["scope"] != "readout":
        raise ValueError("Only a restricted readout successor is qualified for this migration")
    cell = read(chosen/"result.json")
    selected = ROOT/cell["selected_checkpoint"]["path"]
    if sha(selected) != cell["selected_checkpoint"]["sha256"]:
        raise ValueError("Chosen descendant bytes changed")
    evidence = read(RELEASE/"acquisition-intervention/result.json")
    if evidence["status"] != "PASS":
        raise ValueError("Acquired-knowledge attribution audit is missing")
    original = parent()
    successor_owner = load_shared_checkpoint(selected)
    factual = read(ROOT/"runs/SHARED-GG-001/circle/situation.json")
    finite = read(ROOT/"research-continuation/16_v3/GC-002/repair/corrected-library.json")
    successor, session, library, report = migrate(original, successor_owner, factual, finite)
    output = RELEASE/"integration"
    output.mkdir(parents=True, exist_ok=False)
    store_path = ROOT/"runs/A06-integrated-001/solver"
    if store_path.exists():
        raise FileExistsError("Integration store already exists; do not restart it")
    record = SolverStore(store_path).initialize(successor)
    write(output/"factual-situation.json", session.snapshot())
    write(output/"finite-library.json", library.record())
    loaded = ReadOnlyStore(store_path).load()
    reloaded_session = SharedGenerativeSession.restore(loaded, read(output/"factual-situation.json"))
    reload_work = Work()
    reloaded_library = Library.restore(read(output/"finite-library.json"), SharedOwnerRef.from_solver(loaded),
                                       reload_work, proof_backend="indexed")
    if loaded.identity() != successor.identity():
        raise ValueError("Integrated solver did not restore exactly")
    if reloaded_library.record() != library.record():
        raise ValueError("Integrated finite library did not restore exactly")
    # Predicting after restoration may add budget counters, but not factual events.
    if len(reloaded_session.situation.observations) != len(session.situation.observations):
        raise ValueError("Live factual history was lost")
    write(output/"result.json", {"status": "PASS", "report": report, "solver_record": record,
                                 "store": store_path.relative_to(ROOT).as_posix(), "reload_proof_work": reload_work.counts,
                                 "lineage": {"predecessor_solver_sha256": original.identity(),
                                             "trained_checkpoint": cell["selected_checkpoint"]},
                                 "chosen_stream": seed, "chosen_cell": chosen.relative_to(ROOT).as_posix(),
                                 "choice": "Raw-coordinate corrected K, stream 307, if it passes; otherwise normalized conditional teaching, stream 181, if it passes. Declared before raw-instance final outcomes.",
                                 "conditional_readout_component": True,
                                 "causal_acquired_coefficient_transfer": causal,
                                 "operational_solver_replaced": False,
                                 "statistical_answer_guarantee": None,
                                 "scope": "A resumable experimental shared descendant preserves both original executable K and its factual state; the old operational solver remains the predecessor.",
                                 "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in Path(__file__).parent.glob("*.py")}})
    print(json.dumps({"status": "PASS", "solver": record["solver_sha256"], "report": report}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    argparse.ArgumentParser().parse_args()
    run()
