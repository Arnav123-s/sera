"""Replay GG-GUARD-001 in a fresh workspace using its exact frozen source bytes.

The current independent auditor has a documented provenance-label repair. This
entry point keeps that revision separate from the original experiment decoder.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/15_applicability"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=RELEASE)
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Use a fresh replay directory inside the repository runs folder")
    protocol = json.loads((run/"protocol.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    workspace = output/"workspace"
    workspace.mkdir()
    expected = protocol["payload"]["sources"]
    with zipfile.ZipFile(RELEASE/"frozen-sources.zip") as archive:
        if set(archive.namelist()) != set(expected) or len(archive.namelist()) != len(expected):
            raise ValueError("Frozen source inventory mismatch")
        for name, record in expected.items():
            target = (workspace/name).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError("Unsafe frozen source path")
            raw = archive.read(name)
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Frozen decoder bytes changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    # Empty package markers affect discovery only; all executable decoder files
    # are byte-checked above. They are declared here rather than copied from HEAD.
    for package in ("experiments", "experiments/generative_memory"):
        (workspace/package/"__init__.py").write_bytes(b"")
    source_run = workspace/"runs/source"
    source_run.mkdir(parents=True)
    for name in ("protocol.json", "guard-model.json", "thresholds.json", "before-final.json", "records.json.gz", "summary.json"):
        shutil.copyfile(run/name, source_run/name)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(workspace), str(workspace/"src")]),
                   "SERA_GUARD_MODE": "replay", "SERA_GUARD_RUN": str(source_run),
                   "SERA_GUARD_OUTPUT": str(workspace/"runs/replay")}
    result = subprocess.run([sys.executable, str(workspace/"scripts/run_bounded.py"),
                             "--seconds", "300", "--output", str(output/"supervisor"),
                             "--module", "pytest", "--", "-q", "-s",
                             "experiments/generative_memory/applicability_study.py"],
                            cwd=workspace, env=environment, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    shutil.copyfile(workspace/"runs/replay/replay.json", output/"replay.json")
    print(f"Exact frozen-source replay complete: {output/'replay.json'}")


if __name__ == "__main__":
    main()
