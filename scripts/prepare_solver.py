"""Copy a completed solver and gather actual support evidence for continued learning."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from sera.experience import EvidenceReplay
from sera.solver import SolverStore
from sera.storage import write_json


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Choose a fresh output outside the completed seed directory")
    current = SolverStore(source / "solver").load()
    if (source / "solver/.writer.lock").exists():
        raise RuntimeError("Study solver has an active or interrupted writer")
    evidence, sources = EvidenceReplay(), []
    paths = [source / "initial-evidence.json", source / "planned-evidence.json",
             *sorted(source.glob("generation-*-evidence.json"))]
    for path in paths:
        for row in EvidenceReplay.load(path).records:
            evidence.admit(row)
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    shutil.copytree(source / "solver", output)
    evidence.save(output / "experience.json")
    assert SolverStore(output).load().identity() == current.identity()
    result = {"version": current.version, "solver_identity": current.identity(),
              "admitted_trajectories": len(evidence.records),
              "worlds_with_evidence": sorted({row.world_id for row in evidence.records}),
              "evidence_sources": sources,
              "scope": "Continued-learning copy; adds actual support archives, changes no model or skill; excludes meta/query/test records"}
    write_json(output / "preparation.json", result)
    print(f"Prepared {output}: {current.version}, {len(evidence.records)} admitted trajectories")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed study seed directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh solver directory")
    args = parser.parse_args()
    prepare(args.input, args.output)
