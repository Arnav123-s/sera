"""Compact public research endpoints; all working checkpoints remain preserved."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/34_learning_progress"
RUN = ROOT / "runs/LP-study-001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if k.startswith("runs/LP-") and v["status"] != "RESERVED"}
    destination = OUT / "publication" if (OUT / "release-manifest.json").exists() else OUT
    for key in jobs:
        for name in ("state.json", "process.log"):
            source = ROOT / key / name
            if source.exists():
                target = destination / "checks" / (Path(key).name+"-"+name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    write(destination / "costs.json", {"jobs": jobs, "charged_seconds": sum(r["charged_seconds"] for r in jobs.values()),
                                     "live_remaining_seconds": ledger["remaining_seconds"], "new_grants": 0,
                                     "worker_threads": 1, "memory_limit_bytes": 2147483648,
                                     "pending": [k for k, v in ledger["jobs"].items() if v["status"] == "RESERVED"]})


def seal():
    if (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the release seal")
    costs()
    # Full checkpoints at both transitions, both initial policies, exact-resume
    # boundary, prepared features and independent prediction witnesses suffice
    # to reproduce the study. Redundant in-between saves remain locally intact.
    files = set(RUN.glob("*-step-048.pt")) | set(RUN.glob("*-step-096.pt"))
    files |= set(RUN.glob("initial-*.pt")) | {RUN / "learned-3401-step-040.pt"}
    files |= {RUN / n for n in ("parent.json", "data.pt", "data-manifest.json", "final-opened.json")}
    files |= set((RUN / "predictions").glob("*.npz"))
    inventory = []
    for path in sorted(p for p in RUN.rglob("*") if p.is_file()):
        inventory.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                          "public_archive": path in files,
                          "retention": "public endpoint" if path in files else "preserved local intermediate; reproducible from endpoint and source"})
    write(OUT / "checkpoint-inventory.json", inventory)
    groups, group, size = [], [], 0
    for path in sorted(files):
        if group and size+path.stat().st_size > 32*1024*1024:
            groups.append(group)
            group, size = [], 0
        group.append(path)
        size += path.stat().st_size
    if group:
        groups.append(group)
    archives = []
    for i, group in enumerate(groups):
        target = OUT / f"research-{i+1:02d}.zip"
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in group:
                archive.write(path, path.relative_to(ROOT).as_posix())
        archives.append({"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target),
                         "members": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in group]})
    sources = list((ROOT / "experiments/learning_progress").glob("*.py"))
    sources += [ROOT / n for n in ("scripts/run_learning_bounded.py", "scripts/learning_artifacts.py", "tests/test_learning_progress.py")]
    sources += [p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip" and p.name not in {"release-manifest.json", "checklist.md"}]
    write(OUT / "release-manifest.json", {"schema": "sera.learning-progress.release.1", "archives": archives,
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(sources)],
        "live_authority": "private assessment key remains local; public witnesses contain no signing key",
        "local_intermediates": "No working checkpoints were deleted; see checkpoint-inventory.json"})


def verify(restore=False):
    manifest = read(OUT / "release-manifest.json")
    count = 0
    for record in manifest["files"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Changed learning research source: "+record["path"])
    for record in manifest["archives"]:
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("Changed research archive")
        with zipfile.ZipFile(path) as archive:
            expected = {r["path"]: r["sha256"] for r in record["members"]}
            if sorted(archive.namelist()) != sorted(expected):
                raise ValueError("Duplicate or unowned archive member")
            for name, checksum in expected.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/LP-study-001/"):
                    raise ValueError("Unsafe artifact path")
                raw = archive.read(name)
                if hashlib.sha256(raw).hexdigest() != checksum:
                    raise ValueError("Changed checkpoint bytes")
                target = (ROOT / name).resolve()
                if not target.is_relative_to(ROOT / "runs/LP-study-001"):
                    raise ValueError("Escaped artifact root")
                if restore:
                    if target.exists() and sha(target) != checksum:
                        raise ValueError("Preserve divergent local checkpoint: "+name)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as stream:
                            stream.write(raw)
                count += 1
    print(json.dumps({"source_files": len(manifest["files"]), "archive_members": count, "restore": restore}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--costs", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.costs:
        costs()
    else:
        verify(args.restore)


if __name__ == "__main__":
    main()
