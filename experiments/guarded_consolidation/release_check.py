"""Final release checks in a resource-bounded owned worker; no experiment reruns."""

import json
import subprocess
import sys
from pathlib import Path


def test_release_and_read_only_resume():
    root = Path(__file__).resolve().parents[2]
    scripts = ("verify_release.py", "verify_continuation.py", "verify_applicability.py", "verify_v3_release.py")
    results = {}
    for script in scripts:
        result = subprocess.run([sys.executable, "-X", "utf8", "scripts/"+script], cwd=root,
                                capture_output=True, text=True, check=True)
        results[script] = result.stdout.strip()
        print(result.stdout)
    command = [sys.executable, "-X", "utf8", "-m", "scripts.resume_v3_library", "--a", "4", "--b", "5", "--c", "6"]
    resumed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    output = json.loads(resumed.stdout)
    assert output["answer"] == 0 and output["status"] == "CONDITIONAL_FINITE_PROOF"
    target = root/"research-continuation/16_v3/release-verification"
    target.mkdir(exist_ok=True)
    with (target/"result.json").open("x", encoding="utf-8") as f:
        json.dump({"status": "PASS", "checks": results, "resumed_example": output,
                   "scope": "Historical/current manifests, frozen sources and read-only repaired checkpoint; no training or cohort restart"}, f, indent=2)
