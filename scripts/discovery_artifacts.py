"""Deduplicate repeated parent states while preserving exact discovery checkpoints."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/35_operator_discovery"
RUN = ROOT / "runs/OD-study-001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if k.startswith("runs/OD-") and v["status"] != "RESERVED"}
    target = OUT / "publication" if (OUT / "release-manifest.json").exists() else OUT
    for key in jobs:
        for name in ("state.json", "process.log"):
            dest = target / "checks" / (Path(key).name+"-"+name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / key / name, dest)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
                                 "live_remaining_seconds": ledger["remaining_seconds"], "new_grants": 0,
                                 "pending": [k for k, v in ledger["jobs"].items() if v["status"] == "RESERVED"]})


def seal():
    if (OUT / "release-manifest.json").exists():
        raise FileExistsError("Completed discovery release is immutable")
    costs()
    records = []
    files = [RUN / "parent.json", RUN / "selection.json"]
    for path in sorted(RUN.glob("step-*.json")):
        state = read(path)
        parent = state.pop("parent")
        if parent != read(RUN / "parent.json"):
            raise ValueError("A discovery checkpoint changed its inherited learner")
        compact = RUN / "portable" / path.name
        write(compact, state)
        files.append(compact)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                        "compact": compact.relative_to(ROOT).as_posix(), "compact_sha256": sha(compact)})
    write(OUT / "checkpoint-index.json", {"parent": sha(RUN / "parent.json"), "records": records,
                                         "reconstruction_order": ["schema", "contracts", "parent", "owner", "discovery"]})
    target = OUT / "research.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    sources = list((ROOT / "experiments/operator_discovery").glob("*.py"))
    sources += [ROOT / n for n in ("scripts/run_discovery_bounded.py", "scripts/discovery_artifacts.py", "tests/test_operator_discovery.py")]
    sources += [p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip" and p.name not in {"release-manifest.json", "checklist.md"}]
    write(OUT / "release-manifest.json", {"archive": sha(target),
        "members": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in files],
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(sources)]})


def verify():
    from experiments.operator_discovery.core import certify, rational_rank

    manifest = read(OUT / "release-manifest.json")
    for record in manifest["files"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Changed discovery source")
    if sha(OUT / "research.zip") != manifest["archive"]:
        raise ValueError("Changed discovery archive")
    with zipfile.ZipFile(OUT / "research.zip") as archive:
        if sorted(archive.namelist()) != sorted(r["path"] for r in manifest["members"]):
            raise ValueError("Changed archive membership")
        for record in manifest["members"]:
            if hashlib.sha256(archive.read(record["path"])).hexdigest() != record["sha256"]:
                raise ValueError("Changed checkpoint")
        parent = json.loads(archive.read("runs/OD-study-001/parent.json"))
        for record in read(OUT / "checkpoint-index.json")["records"]:
            small = json.loads(archive.read(record["compact"]))
            full = {"schema": small["schema"], "contracts": small["contracts"], "parent": parent,
                    "owner": small["owner"], "discovery": small["discovery"]}
            raw = (json.dumps(full, indent=2, allow_nan=False)+"\n").encode()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Deduplicated checkpoint failed exact reconstruction")
    records = read(OUT / "results.json")["records"]
    if not all(certify(r["coefficients"])["accepted"] for r in records):
        raise ValueError("A discovered relationship failed independent re-verification")
    if rational_rank([r["coefficients"] for r in records]) != len(records):
        raise ValueError("Duplicate discovery credit")
    print(json.dumps({"source_files": len(manifest["files"]), "checkpoint_reconstructions": 9,
                      "independently_verified_relations": len(records)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--costs", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.costs:
        costs()
    else:
        verify()


if __name__ == "__main__":
    main()
