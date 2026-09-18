"""Publish/restore the exact gap inquiry without duplicating its preserved ancestry."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from experiments.gap_inquiry import OUT, ROOT, RUN, read, sha, write
from workbench.storage import Store

TAG = "research-2026-09-18-knowledge-gaps"
ASSET = "sera-43-knowledge-gaps.zip"
ARCHIVE = OUT / "research.zip.part001"


def sources():
    files = [ROOT / p for p in ("experiments/gap_inquiry.py", "experiments/gap_assessor.py",
             "experiments/gap_consequences.py", "scripts/gap_study.py", "scripts/gap_use.py",
             "scripts/gap_artifacts.py", "scripts/run_gap_bounded.py", "tests/test_gap_inquiry.py")]
    files += [p for p in OUT.rglob("*") if p.is_file() and "publication" not in p.relative_to(OUT).parts
              and p.name not in {"release-manifest.json", "research.zip.part001", "checklist.md"}]
    return sorted(set(files))


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    target = OUT / "publication"
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith("KG-") and v["status"] != "RESERVED"}
    for key in jobs:
        for name in ("state.json", "process.log"):
            destination = target / "checks" / (Path(key).name + "-" + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / key / name, destination)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
                                 "remaining_seconds_snapshot": ledger["remaining_seconds"],
                                 "one_cpu_thread": True, "memory_cap_bytes": 2147483648,
                                 "standing_authority": "../RESOURCE_POLICY.md"})
    write(target / "resource-ledger.json", ledger)


def seal():
    if (OUT / "release-manifest.json").exists() or ARCHIVE.exists():
        raise FileExistsError("Preserve the existing exact archive")
    files = sorted([p for p in RUN.rglob("*") if p.is_file()] +
                   [p for p in (ROOT / "runs/sera-gap-inquiry-live").rglob("*") if p.is_file()])
    members = []
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            archive.writestr(name, raw)
            members.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    write(OUT / "release-manifest.json", {
        "schema": "sera.knowledge-gap-archive.1", "archive": sha(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "asset": ASSET, "url": f"https://github.com/Arnav123-s/sera/releases/download/{TAG}/{ASSET}",
        "members": members, "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources()],
        "parent": {"store": "runs/sera-observed-discovery-live",
                   "owner": "91f6c21ffcf80e8aab05ca8eea8e34499a27803cd823aa5a087839ac44957230",
                   "release": "https://github.com/Arnav123-s/sera/releases/tag/research-2026-09-18-discovery"}})
    verify(False)


def verify(restore=False):
    manifest = read(OUT / "release-manifest.json")
    for item in manifest["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Changed published source or evidence: " + item["path"])
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".download")
        with urllib.request.urlopen(manifest["url"], timeout=60) as response, temporary.open("xb") as dest:
            shutil.copyfileobj(response, dest, length=1048576)
        if sha(temporary) != manifest["archive"]:
            raise ValueError("Downloaded archive hash differs")
        temporary.replace(ARCHIVE)
    if sha(ARCHIVE) != manifest["archive"] or ARCHIVE.stat().st_size != manifest["bytes"]:
        raise ValueError("Changed gap archive")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(m["path"] for m in manifest["members"]):
            raise ValueError("Archive member inventory differs")
        for item in manifest["members"]:
            path = (ROOT / item["path"]).resolve()
            if not any(path.is_relative_to(root.resolve()) for root in (RUN, ROOT / "runs/sera-gap-inquiry-live")):
                raise ValueError("Archive member outside owned continuation")
            raw = archive.read(item["path"])
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Archive member differs")
            if restore:
                if path.exists():
                    if sha(path) != item["sha256"]:
                        raise ValueError("Preserve newer local work: " + item["path"])
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as dest:
                        dest.write(raw)
    print(f"Verified {len(manifest['members'])} exact gap records and {len(manifest['files'])} source/evidence files.")
    if restore:
        restore_parent_dependencies()


def restore_parent_dependencies():
    from scripts.refinement_artifacts import archive_file

    required = {
        "41_self_chosen_discovery": ["runs/SD-study-001/language-public.json", "runs/SD-study-001/observation.csv",
                                     "runs/SD-study-001/sera-observed-discovery-live.json"],
        "42_composed_discovery": ["runs/CF-study-001/language-public.json"],
    }
    for folder, paths in required.items():
        directory = ROOT / "research-continuation" / folder
        manifest = read(directory / "release-manifest.json")
        indexed = {r["path"]: r for r in manifest["members"]}
        with zipfile.ZipFile(archive_file(directory, manifest)) as archive:
            for name in paths:
                path = ROOT / name
                if path.exists():
                    if sha(path) != indexed[name]["sha256"]:
                        raise ValueError("Preserve a changed parent dependency: " + name)
                else:
                    raw = archive.read(name)
                    if hashlib.sha256(raw).hexdigest() != indexed[name]["sha256"]:
                        raise ValueError("Changed parent archive dependency")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as dest:
                        dest.write(raw)
                    del raw
                if name.endswith("sera-observed-discovery-live.json"):
                    parent_store = Store(ROOT / "runs/sera-observed-discovery-live")
                    if not (parent_store.directory / "current.json").exists():
                        with path.open(encoding="utf-8") as source:
                            value = json.load(source)
                        if value["owner"] != read(OUT / "release-manifest.json")["parent"]["owner"]:
                            raise ValueError("Parent archive owner mismatch")
                        parent_store.commit(value, None)
                        del value


def trace_dependencies():
    """Record numerical dependencies needed by a genuinely restored owner."""
    from scripts.gap_use import restore

    original, seen = io.open, set()
    def tracked(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(ROOT / "runs") and resolved.is_file():
                seen.add(resolved)
        return original(path, *args, **kwargs)
    io.open = tracked
    try:
        session, _ = restore()
    finally:
        io.open = original
    write(OUT / "dependencies.json", {"owner": session.identity(),
          "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(seen)]})
    print(f"Recorded {len(seen)} actual restore dependencies.")


def clean_check():
    """Use a fresh source tree and restore new ancestry from published-format parts."""
    scratch = ROOT / "runs/KG-clean-checkout-001"
    scratch.mkdir(exist_ok=False)
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    paths = subprocess.run(command, cwd=ROOT, check=True, capture_output=True).stdout.decode().split("\0")
    for name in sorted(set(paths) - {""}):
        source = ROOT / name
        if source.is_file():
            destination = scratch / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for item in read(OUT / "dependencies.json")["files"]:
        name = item["path"]
        if any(name.startswith(p) for p in ("runs/SD-study-001/", "runs/CF-study-001/", "runs/sera-observed-discovery-live/", "runs/sera-gap-inquiry-live/")):
            continue
        destination = scratch / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for folder in ("41_self_chosen_discovery", "42_composed_discovery"):
        directory = ROOT / "research-continuation" / folder
        for part in read(directory / "release-manifest.json")["parts"]:
            shutil.copyfile(directory / part["name"], scratch / "research-continuation" / folder / part["name"])
    shutil.copyfile(ARCHIVE, scratch / ARCHIVE.relative_to(ROOT))
    environment = {**os.environ, "PYTHONPATH": str(scratch / "src")}
    steps = [["-m", "scripts.gap_artifacts", "--restore"],
             ["-m", "scripts.gap_use", "--input", "research-continuation/43_knowledge_gaps/example-tasks.json",
              "--output", "runs/clean-use.json"]]
    for index, args in enumerate(steps):
        with (scratch / f"check-{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, *args], cwd=scratch, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    if read(scratch / "runs/clean-use.json") != read(RUN / "example-results.json"):
        raise ValueError("Clean restored owner task results differ")
    write(OUT / "publication/clean-restore.json", {"passed": True, "tasks": 7,
          "source_tree": scratch.relative_to(ROOT).as_posix(), "owner": read(OUT / "integration.json")["owner"],
          "parent_restored_from_exact_prior_archive": True, "results_identical": True})
    print("Fresh-tree restore reproduced all seven actual-owner task results exactly.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seal", action="store_true")
    p.add_argument("--restore", action="store_true")
    p.add_argument("--costs", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--clean-check", action="store_true")
    args = p.parse_args()
    if args.clean_check:
        clean_check()
    elif args.trace:
        trace_dependencies()
    elif args.costs:
        costs()
    elif args.seal:
        seal()
    else:
        verify(args.restore)


if __name__ == "__main__":
    main()
