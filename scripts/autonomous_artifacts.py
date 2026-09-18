"""Seal, reconstruct and independently check the autonomous discovery release."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/36_autonomous_discovery"
RUN = ROOT / "runs/AD-study-002"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if k.startswith("runs/AD-") and v["status"] != "RESERVED"}
    target = OUT / "publication" if (OUT / "release-manifest.json").exists() else OUT
    for key in jobs:
        for name in ("state.json", "process.log"):
            dest = target / "checks" / (Path(key).name+"-"+name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / key / name, dest)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
                                 "remaining_seconds_snapshot": ledger["remaining_seconds"], "new_grants": 0,
                                 "cpu_threads": 1, "memory_limit_bytes": 2147483648,
                                 "pending": [k for k, v in ledger["jobs"].items() if v["status"] == "RESERVED"]})


def seal():
    if (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the completed release")
    costs()
    records, files = [], [RUN / "parent.json", RUN / "selection.json"]
    for path in sorted(RUN.glob("step-*.json")):
        state = read(path)
        parent = state.pop("parent")
        if parent != read(RUN / "parent.json"):
            raise ValueError("Changed inherited owner")
        compact = RUN / "portable" / path.name
        write(compact, state)
        files.append(compact)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                        "compact": compact.relative_to(ROOT).as_posix()})
    write(OUT / "checkpoint-index.json", {"records": records, "parent_sha256": sha(RUN / "parent.json"),
        "failed_first_parent": {"path": "runs/AD-study-001/parent.json",
                                "sha256": sha(ROOT / "runs/AD-study-001/parent.json"),
                                "retention": "preserved locally; byte-identical inherited parent"}})
    target = OUT / "research.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    sources = list((ROOT / "experiments/autonomous_discovery").glob("*.py"))
    sources += [ROOT / p for p in ("scripts/run_autonomous_bounded.py", "scripts/autonomous_artifacts.py", "tests/test_autonomous_discovery.py")]
    sources += [p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip" and p.name not in {"release-manifest.json", "checklist.md"}]
    write(OUT / "release-manifest.json", {"archive": sha(target),
        "members": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in files],
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(sources)]})


def verify():
    from fractions import Fraction as Q

    from experiments.autonomous_discovery.core import WORDS, certify, consequence_span
    from experiments.operator_discovery.core import WORDS as OLD_WORDS
    from experiments.self_study.algebra import check

    manifest = read(OUT / "release-manifest.json")
    for record in manifest["files"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Changed autonomous research source: "+record["path"])
    if sha(OUT / "research.zip") != manifest["archive"]:
        raise ValueError("Changed autonomous archive")
    with zipfile.ZipFile(OUT / "research.zip") as archive:
        if sorted(archive.namelist()) != sorted(r["path"] for r in manifest["members"]):
            raise ValueError("Changed archive membership")
        for record in manifest["members"]:
            if hashlib.sha256(archive.read(record["path"])).hexdigest() != record["sha256"]:
                raise ValueError("Changed checkpoint")
        parent = json.loads(archive.read("runs/AD-study-002/parent.json"))
        for record in read(OUT / "checkpoint-index.json")["records"]:
            small = json.loads(archive.read(record["compact"]))
            full = {"schema": small["schema"], "contracts": small["contracts"], "parent": parent,
                    "owner": small["owner"], "state": small["state"], "rng": small["rng"],
                    "policy": small["policy"], "denominators": small["denominators"]}
            raw = (json.dumps(full, indent=2, allow_nan=False)+"\n").encode()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Failed exact checkpoint reconstruction")
    known = []
    for record in parent["discovery"]["records"]:
        row = [Q(0)]*len(WORDS)
        for word, coefficient in zip(OLD_WORDS, record["coefficients"], strict=True):
            row[WORDS.index(word)] = Q(coefficient)
        known.append({"coefficients": row})
    result = read(OUT / "results.json")
    total, limits, procedure_points = 0, {"integral": 120, "sum": 120}, 0
    for event in result["state"]["events"]:
        new_count = 0
        span = consequence_span(known)
        for receipt in event["proposals"]:
            certificate = certify(receipt["coefficients"])
            if certificate != receipt["certificate"]:
                raise ValueError("Changed independent certificate")
            new = certificate["accepted"] and span.add(receipt["coefficients"])
            if new != receipt["new_discovery"]:
                raise ValueError("Duplicate or withheld discovery credit")
            if new:
                known.append({"coefficients": receipt["coefficients"]})
                span = consequence_span(known)
                new_count += 1
        updates = event["generation"].get("procedure_updates", [])
        for update in updates:
            domain = update["domain"]
            if (update["previous_limit"] != limits[domain] or update["new_limit"] <= limits[domain]
                    or check(domain, update["input"], update["old_candidate"])["accepted"]
                    or not check(domain, update["input"], update["new_candidate"])["accepted"]):
                raise ValueError("Unverified procedure-progress credit")
            limits[domain] = update["new_limit"]
        expected = new_count+len(updates)-.01*event["generation"]["completed_program_columns"]
        if new_count != event["discovery_points"] or abs(expected-event["reward"]) > 1e-12:
            raise ValueError("Reward accounting mismatch")
        total += new_count
        procedure_points += len(updates)
    if total != len(result["state"]["records"]) or procedure_points != result["procedure_improvement_points"]:
        raise ValueError("Discovery inventory mismatch")
    print(json.dumps({"source_files": len(manifest["files"]), "checkpoint_reconstructions": 3,
                      "verified_discoveries": total, "verified_procedure_advances": procedure_points}))


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
