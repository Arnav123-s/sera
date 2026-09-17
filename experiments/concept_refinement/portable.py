"""Run affected checks in a separate fresh checkout with authenticated artifacts."""

import json
import os
import shutil
import subprocess
import sys
import time

from .common import OUT, ROOT, sha, write


def main():
    checkout = ROOT / "runs/CR-ci-clean"
    folder = OUT / "publication-hardening"
    if not checkout.is_dir():
        raise ValueError("Create the explicitly named fresh detached worktree first")
    for name in ["scripts/restore_test_artifacts.py"] + [
        "research-continuation/28_concept_refinement/publication-hardening/" + name
        for name in ("fixtures.json", "additional-fixtures.zip")
    ]:
        destination = checkout / name
        if destination.exists():
            raise FileExistsError("Fresh-checkout overlay must not overwrite existing work")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    commands = [
        ("restore", ["scripts/restore_test_artifacts.py"]),
        ("affected-tests", ["-m", "pytest", "-q", *read_targets(folder)]),
    ]
    receipts = []
    for name, arguments in commands:
        start = time.perf_counter()
        path = folder / ("fresh-" + name + ".log")
        with path.open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", *arguments],
                cwd=checkout,
                env={
                    **os.environ,
                    "PYTHONPATH": str(checkout / "src") + os.pathsep + str(checkout),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=200,
            )
        receipts.append(
            {
                "job": name,
                "returncode": result.returncode,
                "seconds": time.perf_counter() - start,
                "log_sha256": sha(path),
            }
        )
        write(folder / "fresh-checkout.json", receipts)
        print(json.dumps(receipts[-1]), flush=True)
        if result.returncode:
            raise RuntimeError("Preserve the failed fresh-checkout verification")


def read_targets(folder):
    return json.loads((folder / "fixture-reads.json").read_text())["targets"]


if __name__ == "__main__":
    main()
