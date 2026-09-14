"""Record existing work without copying it; verify changes against preserved Git blobs."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def capture(path):
    if path.exists():
        raise FileExistsError("Use a new baseline; existing preservation records are immutable")
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and not SKIP.intersection(p.relative_to(ROOT).parts))
    rows = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha(p.read_bytes())} for p in paths]
    record = {"git_commit": git("rev-parse", "HEAD").decode().strip(), "files": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0))
    return {"files": len(rows), "bytes": sum(r["bytes"] for r in rows), "git_commit": record["git_commit"]}


def verify(path):
    record = json.loads(gzip.decompress(path.read_bytes()))
    changed, missing, lost = [], [], []
    for row in record["files"]:
        current = ROOT / row["path"]
        if not current.is_file():
            missing.append(row["path"])
        elif sha(current.read_bytes()) != row["sha256"]:
            try:
                original = git("show", f"{record['git_commit']}:{row['path']}")
            except subprocess.CalledProcessError:
                original = b""
            if sha(original) != row["sha256"]:
                lost.append(row["path"])
            else:
                changed.append(row["path"])
    result = {"passed": not missing and not lost, "baseline_files": len(record["files"]),
              "unchanged_files": len(record["files"]) - len(changed) - len(missing) - len(lost),
              "revised_with_exact_original_in_git": changed, "missing": missing,
              "unpreserved_changes": lost, "original_commit": record["git_commit"]}
    if not result["passed"]:
        raise ValueError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.baseline) if args.action == "capture" else verify(args.baseline), indent=2))


if __name__ == "__main__":
    main()
