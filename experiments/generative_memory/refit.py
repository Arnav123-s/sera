"""Refit frozen GG controls for deterministic training parity and resource auditing.

Only compact comparisons/costs are retained: original model and query archives
remain authoritative, with no duplicate checkpoint bank or independent-seed claim.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit import load_frozen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads((args.run / "state.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    if state["status"] != "COMPLETED":
        raise ValueError("Only a completed primary cohort can be refitted")
    for name, expected in state["source_files"].items():
        if sha(args.run / "source" / name) != expected:
            raise ValueError("Frozen source bytes changed")
    core = load_frozen(args.run / "source/core.py", "gg_frozen_refit")
    torch.set_num_threads(1)
    start, cpu = time.perf_counter(), time.process_time()
    count = mismatches = 0
    with (args.output / "comparisons.jsonl").open("w", encoding="utf-8") as output:
        for completed in state["completed"]:
            path = args.run / completed["path"]
            if sha(path) != completed["sha256"]:
                raise ValueError("Completed primary record changed")
            case = json.loads(path.read_text())
            with np.load(path.parent / "observations.npz", allow_pickle=False) as data:
                banks = [core.Observations(data[f"{role}_t"], data[f"{role}_y"], role)
                         for role in ("support", "selection", "calibration")]
            for record in case["records"]:
                model, work = core.fit(record["method"], *banks, case["case"]["noise"],
                                       initialization_seed=case["case"]["seed"] + 50000,
                                       steps=protocol["neural_steps"])
                model["decoder_sha256"] = state["source_files"]["core.py"]
                fresh = hashlib.sha256((core.canonical(model) + "\n").encode()).hexdigest()
                if sha(path.parent / record["artifact"]) != record["artifact_sha256"]:
                    raise ValueError("Original learned artifact changed")
                matched = fresh == record["artifact_sha256"]
                mismatches += not matched
                output.write(core.canonical({"case": completed["case_id"], "method": record["method"],
                                              "expected_sha256": record["artifact_sha256"], "refit_sha256": fresh,
                                              "exact_match": matched, "fit_cpu_seconds": work["cpu_seconds"],
                                              "fit_wall_seconds": work["wall_seconds"]}) + "\n")
                count += 1
            output.flush()
            if count % 256 == 0:
                print(core.canonical({"models_refitted": count, "mismatches": mismatches}), flush=True)
    report = {"status": "PASS" if mismatches == 0 else "FAILED", "refitted_models": count,
              "exact_artifact_mismatches": mismatches, "cases": len(state["completed"]),
              "source_sha256": state["contract"]["source_sha256"], "protocol_sha256": state["contract"]["protocol_sha256"],
              "refit_script_sha256": sha(Path(__file__)),
              "comparisons_sha256": sha(args.output / "comparisons.jsonl"),
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter()-start,
              "boundary": "Same-seed, same-source training reproduction; does not add independent instances. Native memory and hard deadline are recorded by the separate supervisor."}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
