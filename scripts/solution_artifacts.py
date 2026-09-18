"""Preserve complete solution studies and restore exact qualified checkpoints."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/44_solution_portfolios"
RUN = ROOT / "runs/PS-study-001"
ARCHIVE = OUT / "research.zip.part001"
TAG = "research-2026-09-18-solutions"
ASSET = "sera-44-solution-portfolios.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    target = OUT / "publication"
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith("PS-") and v["status"] not in {"RESERVED", "RUNNING"}}
    for key in jobs:
        for name in ("state.json", "process.log", "junit.xml"):
            source = ROOT / key / name
            if source.exists():
                dest = target / "checks" / (Path(key).name + "-" + name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
              "unlimited_local_time": ledger["unlimited_local_time"], "one_cpu_thread": True, "memory_cap_bytes": 2147483648,
              "original_main_run_seconds": read(ROOT / "runs/PS-full-cycle-001/state.json")["charged_seconds"]})
    write(target / "resource-ledger.json", ledger)


def sources():
    files = list((ROOT / "experiments").glob("solution_*.py"))
    files += list((ROOT / "scripts").glob("solution_*.py"))
    files += [ROOT / "scripts/run_solution_persistent.py", ROOT / "scripts/run_solutions_bounded.py"]
    files += list((ROOT / "tests").glob("test_solution*.py"))
    files += [p for p in OUT.rglob("*") if p.is_file() and "publication" not in p.parts
              and p.name not in {"research.zip.part001", "release-manifest.json", "checklist.md"}]
    return sorted(set(files))


def seal():
    if ARCHIVE.exists() or (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the sealed solution release")
    files = [p for folder in (RUN, ROOT / "runs/sera-solutions-live", ROOT / "runs/sera-solution-progress-live")
             for p in folder.rglob("*") if p.is_file()]
    members = []
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            members.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    write(OUT / "release-manifest.json", {"schema": "sera.solution-portfolio-release.1", "archive": sha(ARCHIVE),
          "bytes": ARCHIVE.stat().st_size, "asset": ASSET,
          "url": f"https://github.com/Arnav123-s/sera/releases/download/{TAG}/{ASSET}", "members": members,
          "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources()],
          "parent": "research-continuation/43_knowledge_gaps/release-manifest.json",
          "preservation": "Original study, corrected credits, failed fits, full proposal queue and exact resumable checkpoints"})
    verify()


def verify(restore=False):
    manifest = read(OUT / "release-manifest.json")
    for item in manifest["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Changed solution source or evidence: " + item["path"])
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".download")
        with urllib.request.urlopen(manifest["url"], timeout=60) as response, temporary.open("xb") as dest:
            shutil.copyfileobj(response, dest, length=1048576)
        if sha(temporary) != manifest["archive"]:
            raise ValueError("Downloaded solution archive differs")
        temporary.replace(ARCHIVE)
    if sha(ARCHIVE) != manifest["archive"] or ARCHIVE.stat().st_size != manifest["bytes"]:
        raise ValueError("Changed solution archive")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(m["path"] for m in manifest["members"]):
            raise ValueError("Unexpected or duplicate archive members")
        for item in manifest["members"]:
            name = item["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archived path")
            path = (ROOT / name).resolve()
            if not any(path.is_relative_to(p) for p in (RUN, ROOT / "runs/sera-solutions-live", ROOT / "runs/sera-solution-progress-live")):
                raise ValueError("Archive escaped its owned work")
            raw = archive.read(name)
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Archived checkpoint changed")
            if restore:
                if path.exists():
                    if sha(path) != item["sha256"]:
                        raise ValueError("Preserve newer local work: " + name)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as dest:
                        dest.write(raw)
    print(f"Verified {len(manifest['members'])} archived records and {len(manifest['files'])} source/evidence files.")


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
