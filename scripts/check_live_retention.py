"""Continue a preserved solver under the new gates without resetting its history."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sera.solver import SolverStore
from sera.storage import write_json
from sera.training import source_hash


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/sera-0.3-current"))
    parser.add_argument("--output", type=Path, default=Path("runs/sera-0.4-current"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() or args.report.exists() or source in output.parents:
        raise ValueError("Choose fresh continuation and report paths")
    if (source / ".writer.lock").exists():
        raise RuntimeError("The parent has an active or interrupted writer")
    previous = SolverStore(source).current_record()
    before = hashes(source)
    shutil.copytree(source, output)
    lineage = {"parent_directory": source.as_posix(), "parent_current": previous,
               "parent_file_hashes": before, "source_sha256": source_hash(),
               "admission_contract": "separate-retention-v2",
               "purpose": "One continuation fork preserves the entire existing alpha ledger, rejected versions and replay. The parent remains a historical workspace."}
    write_json(output / "continuation.json", lineage)
    start = time.perf_counter()
    process = subprocess.run([sys.executable, "-m", "sera", "learn", str(output), "--world-seed", "880003",
                              "--family", "permutation", "--seed", "810003", "--samples", "1024", "--steps", "32"],
                             capture_output=True, text=True)
    if process.returncode:
        write_json(args.report, {"status": "failed", "stdout": process.stdout, "stderr": process.stderr})
        raise RuntimeError("Live continuation failed; inspect its preserved report")
    result = json.loads(process.stdout)
    after = SolverStore(output).current_record()
    if hashes(source) != before:
        raise ValueError("The preserved source workspace changed")
    if result["decision"]["capability_contract"] != "separate-retention-v2":
        raise ValueError("Ordinary learning did not use the new contract")
    report = {"source_sha256": source_hash(), "parent_unchanged": True, "parent": previous,
              "current": after, "result": result, "wall_seconds": time.perf_counter() - start,
              "continuation_path": output.as_posix(), "scope": "One fresh autonomous world attempt, separate from the three-seed comparison. Typed v2 remains an unpromoted research variant."}
    write_json(args.report, report)
    print(json.dumps({"status": result["status"], "round": result["round_index"], "method": result["proposal"]["method"],
                      "capabilities": len(result["decision"]["retention_loss_by_capability"]),
                      "parent_unchanged": True, "current_version": after["version"]}))


if __name__ == "__main__":
    main()
