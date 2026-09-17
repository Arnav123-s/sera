"""Preserve the five standard addenda in a new, path-checked intake directory."""

import hashlib
import json
import stat
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "research/intake/v6-v10-20260916"
OUT = ROOT / "research-continuation/25_constraint_inquiry"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
        text=True,
    ).strip()


def main():
    if DEST.exists() or OUT.exists():
        raise FileExistsError("Preserve existing intake and reconciliation")
    OUT.mkdir(parents=True)
    budget = ROOT / "runs/v3-batch-001/budget.json"
    lock = ROOT / "runs/v3-batch-001/active.lock"
    kavi = ROOT / "research-continuation/00_sources/kavi-pinned"
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "sera_head": git(ROOT, "rev-parse", "HEAD"),
        "sera_dirty": git(ROOT, "status", "--porcelain"),
        "kavi_head": git(kavi, "rev-parse", "HEAD"),
        "kavi_dirty": git(kavi, "status", "--porcelain"),
        "budget_sha256": sha(budget),
        "budget": json.loads(budget.read_text()),
        "active_lock": lock.read_text() if lock.exists() else None,
        "pointers": {},
        "archives": [],
    }
    for store in ["sera-workbench", "sera-task-transfer", "sera-inquiry"]:
        path = ROOT / "runs" / store / "current.json"
        record["pointers"][store] = {"sha256": sha(path), "pointer": json.loads(path.read_text())}
    for version in ["v6", "v7_Light", "v8", "v9", "v10"]:
        path = Path("D:/") / f"SERA_{version}.zip"
        root = f"SERA_{version}"
        identities = {}
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 2000 or sum(x.file_size for x in members) > 100_000_000:
                raise ValueError("Unexpected archive dimensions")
            names = set()
            for member in members:
                name = PurePosixPath(member.filename)
                target = DEST.joinpath(*name.parts).resolve()
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or ":" in member.filename or not name.parts or name.parts[0] != root
                        or any(p.endswith((".", " ")) for p in name.parts)
                        or not target.is_relative_to(DEST)
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unsafe archive member")
                if str(target).casefold() in names:
                    raise ValueError("Case insensitive collision")
                names.add(str(target).casefold())
            for member in members:
                target = DEST / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(member)
                with target.open("xb") as stream:
                    stream.write(data)
                identities[member.filename] = hashlib.sha256(data).hexdigest()
        record["archives"].append({
            "path": str(path), "sha256": sha(path), "files": identities,
            "bytes": sum(x.file_size for x in members),
        })
    record["scope"] = "Static extraction and identity capture; no imported program executed."
    (OUT / "reconciliation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(DEST), "file_counts": [len(a["files"]) for a in record["archives"]],
                      "remaining_seconds": record["budget"]["remaining_seconds"]}))


if __name__ == "__main__":
    main()
