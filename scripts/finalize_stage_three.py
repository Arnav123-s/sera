"""Collect completed evidence, audit executable behavior, then build the public report."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sera.storage import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    while not (args.study / "summary.json").exists():
        if not args.wait or (args.study / "cohort-failure.json").exists() or time.monotonic() - started > 7200:
            raise RuntimeError("The declared study has not completed successfully")
        time.sleep(5)
    reports = root / "reports"
    shutil.copyfile(args.study / "summary.json", reports / "stage-three-data.json")
    shutil.copyfile(root / "runs/original-reproduction-stage-three/reproduction.json",
                    reports / "stage-three-source-reproduction.json")
    summary = json.loads((args.study / "summary.json").read_text())
    episodes = []
    for seed in summary["manifest"]["seeds"]:
        episodes.extend(json.loads(path.read_text()) for path in sorted((args.study / str(seed) / "policy-episodes").glob("*.json")))
    write_json(reports / "stage-three-policy-episodes.json", episodes)
    subprocess.run([sys.executable, str(root / "scripts/audit_stage_three.py"), "--study", str(args.study),
                    "--output", str(reports / "stage-three-verification.json")], check=True)
    subprocess.run([sys.executable, str(root / "scripts/build_stage_three_report.py")], check=True)
    paths = ["stage-three-data.json", "stage-three-source-reproduction.json", "stage-three-policy-episodes.json",
             "stage-three-verification.json", "stage-three-development.json", "stage-three-source-index.json",
             "stage-three-empty-target-repair.json", "stage-three-results.png", "stage-three-results.svg"]
    paths.append("stage-three-source-audit.json")
    write_json(reports / "stage-three-evidence-manifest.json",
               [{"file": name, "sha256": hashlib.sha256((reports / name).read_bytes()).hexdigest(),
                 "bytes": (reports / name).stat().st_size} for name in paths])
    subprocess.run([sys.executable, str(root / "scripts/verify_release.py")], check=True)
    print("Completed artifact collection, independent audit, report and evidence verification.", flush=True)


if __name__ == "__main__":
    main()
