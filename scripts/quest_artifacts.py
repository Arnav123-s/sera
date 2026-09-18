"""Preserve and verify the finite quest release without overwriting live work."""

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/33_capability_portfolio"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if k.startswith("runs/QP-") and v["status"] != "RESERVED"}
    destination = OUT / "publication" if (OUT / "release-manifest.json").exists() else OUT
    for key in jobs:
        for name in ("state.json", "process.log"):
            source = ROOT / key / name
            if source.exists():
                target = destination / "checks" / (Path(key).name+"-"+name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    write(destination / "costs.json", {"as_of": datetime.now(timezone.utc).isoformat(), "jobs": jobs,
                              "charged_seconds": sum(r["charged_seconds"] for r in jobs.values()),
                              "live_remaining_seconds": ledger["remaining_seconds"],
                              "ledger_sha256": sha(ROOT / "runs/v3-batch-001/budget.json"),
                              "worker_threads": 1, "memory_limit_bytes": 2147483648, "new_grants": 0,
                              "pending_reservations": [k for k, v in ledger["jobs"].items() if v["status"] == "RESERVED"]})


def seal():
    path = OUT / "release-manifest.json"
    if path.exists():
        raise FileExistsError("Preserve this release seal")
    costs()
    selected = ROOT / read(OUT / "selection.json")["checkpoint"]
    runtime = [selected, ROOT / "runs/QP-study-001/parent.json"]
    research = []
    for prefix in ("QP-study-001", "QP-owner-audit-001", "QP-packet-check-001", "QP-packet-portable-001"):
        research.extend(p for p in (ROOT / "runs" / prefix).rglob("*")
                        if p.is_file() and p not in runtime and p.name != "issuer.key")
    manifest = {"schema": "sera.verified-quests.release.1", "archives": {}, "files": [],
                "private_assessor_keys": "retained locally; never published",
                "mutable_navigation": ["research-continuation/33_capability_portfolio/checklist.md"]}
    for name, files in (("runtime", runtime), ("research", research)):
        destination = OUT / (name+".zip")
        if destination.exists():
            raise FileExistsError("Preserve existing release archives")
        records = []
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for file in sorted(files):
                relative = file.relative_to(ROOT).as_posix()
                archive.write(file, relative)
                records.append({"path": relative, "sha256": sha(file)})
        manifest[name] = records
        manifest["archives"][name] = sha(destination)
    files = list((ROOT / "experiments/quest_portfolio").glob("*.py"))
    files += [ROOT / "scripts/run_quests_bounded.py", ROOT / "scripts/quest_artifacts.py", ROOT / "tests/test_quest_portfolio.py"]
    files += [p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip"
              and p.name not in {"release-manifest.json", "checklist.md"}]
    manifest["files"] = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(files)]
    write(path, manifest)


def verify(restore=False, research=False):
    manifest = read(OUT / "release-manifest.json")
    for entry in manifest["files"]:
        file = (ROOT / entry["path"]).resolve()
        if not file.is_relative_to(ROOT) or sha(file) != entry["sha256"]:
            raise ValueError("Changed published quest evidence: "+entry["path"])
    count = 0
    for kind in ("runtime", "research"):
        archive_path = OUT / (kind+".zip")
        if sha(archive_path) != manifest["archives"][kind]:
            raise ValueError("Archive checksum changed")
        with zipfile.ZipFile(archive_path) as archive:
            if sorted(archive.namelist()) != sorted(e["path"] for e in manifest[kind]):
                raise ValueError("Unexpected, missing or duplicated archive entries")
            for entry in manifest[kind]:
                name = entry["path"]
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/"):
                    raise ValueError("Unsafe archive path")
                raw = archive.read(name)
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("Changed archived artifact")
                if kind == "research" and not research:
                    continue
                destination = (ROOT / name).resolve()
                if not destination.is_relative_to(ROOT / "runs"):
                    raise ValueError("Artifact escaped runs")
                if destination.exists():
                    if sha(destination) != entry["sha256"]:
                        raise ValueError("Preserve divergent local work: "+name)
                elif restore:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as file:
                        file.write(raw)
                    count += 1
    print(json.dumps({"files_verified": len(manifest["files"]), "restored": count,
                      "runtime_members": len(manifest["runtime"]), "research_members": len(manifest["research"]),
                      "experiments_restarted": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--costs", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--research", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.costs:
        costs()
    else:
        verify(args.restore, args.research)


if __name__ == "__main__":
    main()
