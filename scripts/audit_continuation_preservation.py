"""Verify the pre-continuation inventory against bytes or its preserved Git commit."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/probability-cloud-intake/preservation-baseline.json.gz"


def main():
    baseline = json.loads(gzip.decompress(BASELINE.read_bytes()))
    changed, failures, unchanged = [], [], 0
    for record in baseline["files"]:
        path = ROOT / record["path"]
        assert path.resolve().is_relative_to(ROOT)
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if current == record["sha256"]:
            unchanged += 1
            continue
        original = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show",
                                   f"{baseline['git_commit']}:{record['path']}"], cwd=ROOT,
                                  capture_output=True, check=False)
        if original.returncode == 0 and hashlib.sha256(original.stdout).hexdigest() == record["sha256"]:
            changed.append({"path": record["path"], "original_sha256": record["sha256"],
                            "current_sha256": current, "preserved_git_commit": baseline["git_commit"]})
        else:
            failures.append({"path": record["path"], "original_sha256": record["sha256"],
                             "current_sha256": current})
    report = {"status": "PASS" if not failures else "FAIL", "baseline_files": len(baseline["files"]),
              "baseline_bytes": sum(row["bytes"] for row in baseline["files"]),
              "unchanged_files": unchanged, "edited_files_preserved_in_git": changed,
              "unresolved_preservation_failures": failures,
              "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
              "scope": "All files in the pre-continuation inventory. Every changed file has its original bytes in "
                       "the pinned Git commit; untracked raw evidence must remain byte-identical. "
                       "This does not classify every transient editing buffer as a published research artifact."}
    output = ROOT / "research-continuation/01_audit/preservation.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
