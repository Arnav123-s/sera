"""Run the release checks once, retaining each command and complete output."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.gap_inquiry import ROOT, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned check directory")
    output.mkdir(parents=True)
    checks = [
        ("lint", ["-m", "ruff", "check", "src", "tests", "scripts", "experiments"]),
        ("regression", ["-m", "pytest", "--junitxml", str(output / "junit.xml")]),
        *[(name, [f"scripts/{name}.py"]) for name in (
            "verify_release", "verify_continuation", "verify_applicability", "verify_v3_release",
            "verify_transfer_release", "verify_continuing_release", "verify_concept_release")],
        ("cli", ["-m", "sera", "--help"]),
    ]
    results = []
    for name, command in checks:
        started = time.perf_counter()
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"name": name, "command": command, "returncode": result.returncode,
               "seconds": time.perf_counter() - started}
        results.append(row)
        write(output / "checks.json", results)
        print(json.dumps(row), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
