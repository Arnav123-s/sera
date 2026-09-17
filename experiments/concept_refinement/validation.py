"""One bounded pass over CI integrity checks and the actual public CLI paths."""

import argparse
import json
import re
import subprocess
import sys
import time

from .common import OUT, ROOT, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="validation")
    parser.add_argument("--start-at", type=int, choices=range(12), default=0)
    args = parser.parse_args()
    if not re.fullmatch(r"validation(?:-[a-z0-9]+)*", args.folder):
        raise ValueError("Use a fresh validation subdirectory name")
    folder = OUT / args.folder
    folder.mkdir(exist_ok=False)
    commands = [("lint", ["-m", "ruff", "check", "src", "tests", "scripts", "experiments"])]
    commands += [
        (name, ["scripts/" + name + ".py"])
        for name in (
            "verify_release",
            "verify_continuation",
            "verify_applicability",
            "verify_v3_release",
            "verify_transfer_release",
            "verify_continuing_release",
        )
    ]
    commands += [("sera-help", ["-m", "sera", "--help"])]
    commands += [
        ("cli-" + action, ["-m", "experiments.concept_refinement.runtime", action, *arguments])
        for action, arguments in (
            ("predict", ["--commands", "0.8,0.8,0.8,0.8"]),
            ("motion", []),
            ("interpret", ["--text", "enciende las luces"]),
        )
    ]
    receipts = []
    for name, arguments in commands[args.start_at:]:
        path = folder / (name + ".log")
        start = time.perf_counter()
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", *arguments],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
        receipt = {
            "job": name,
            "arguments": arguments,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - start,
            "log_sha256": sha(path),
        }
        receipts.append(receipt)
        write(folder / "receipts.json", receipts)
        print(json.dumps(receipt), flush=True)
        if result.returncode:
            raise RuntimeError("Preserved validation failure; inspect before continuing")
    for action, status in (("predict", "EMPIRICAL_PREDICTION"), ("motion", "CERTIFIED_ALGEBRA")):
        value = json.loads((folder / ("cli-" + action + ".log")).read_text())
        assert value["status"] == status
    exact = json.loads((folder / "cli-motion.log").read_text())
    assert exact["result"]["position"] == "125/4" and exact["result"]["velocity"] == "55/2"


if __name__ == "__main__":
    main()
