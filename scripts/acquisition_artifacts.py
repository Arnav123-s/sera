"""Preserve and independently verify prerequisite acquisition and its discoveries."""

import argparse
import hashlib
import json
import shutil
import zipfile

from scripts.autonomous_artifacts import ROOT, read, sha, write

OUT = ROOT / "research-continuation/37_constraint_acquisition"
RUN = ROOT / "runs/AC-study-001"


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if k.startswith("runs/AC-") and v["status"] != "RESERVED"}
    target = OUT / "publication" if (OUT / "release-manifest.json").exists() else OUT
    for key in jobs:
        from pathlib import Path
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
        raise FileExistsError("Preserve the completed acquisition release")
    costs()
    files, records = [RUN / "parent.json", RUN / "selection.json"], []
    for path in [RUN / "acquired.json", *sorted(RUN.glob("step-*.json"))]:
        state = read(path)
        if state.pop("parent") != read(RUN / "parent.json"):
            raise ValueError("Changed inherited learner")
        compact = RUN / "portable" / path.name
        write(compact, state)
        files.append(compact)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                        "compact": compact.relative_to(ROOT).as_posix()})
    write(OUT / "checkpoint-index.json", {"records": records, "parent_sha256": sha(RUN / "parent.json")})
    with zipfile.ZipFile(OUT / "research.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    sources = [ROOT / p for p in ("experiments/autonomous_discovery/acquire.py", "scripts/run_acquisition_bounded.py",
                                  "scripts/acquisition_artifacts.py", "tests/test_constraint_acquisition.py")]
    sources += [p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip" and p.name not in {"release-manifest.json", "checklist.md"}]
    write(OUT / "release-manifest.json", {"archive": sha(OUT / "research.zip"),
        "members": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in files],
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(sources)]})


def verify():
    from fractions import Fraction as Q

    from experiments.autonomous_discovery.core import WORDS, certify, consequence_span
    from experiments.operator_discovery.core import WORDS as OLD_WORDS
    from experiments.self_study.algebra import check, independent, vector

    manifest = read(OUT / "release-manifest.json")
    for record in manifest["files"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Changed acquisition research source")
    if sha(OUT / "research.zip") != manifest["archive"]:
        raise ValueError("Changed acquisition archive")
    with zipfile.ZipFile(OUT / "research.zip") as archive:
        if sorted(archive.namelist()) != sorted(r["path"] for r in manifest["members"]):
            raise ValueError("Changed archive membership")
        for record in manifest["members"]:
            if hashlib.sha256(archive.read(record["path"])).hexdigest() != record["sha256"]:
                raise ValueError("Changed checkpoint")
        parent = json.loads(archive.read("runs/AC-study-001/parent.json"))
        for record in read(OUT / "checkpoint-index.json")["records"]:
            small = json.loads(archive.read(record["compact"]))
            full = {"schema": small["schema"], "contracts": small["contracts"], "parent": parent,
                    **{k: small[k] for k in ("owner", "maps", "state", "acquisition", "rng", "policy", "denominators")}}
            raw = (json.dumps(full, indent=2, allow_nan=False)+"\n").encode()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Failed exact acquisition checkpoint reconstruction")
    result = read(OUT / "results.json")
    acquired = 0
    for record in result["acquisition"]:
        if not check(record["domain"], record["input"], record["candidate"])["accepted"]:
            raise ValueError("Unverified learned operator correction")
        if not independent(record["domain"], record["input"], record["candidate"]):
            raise ValueError("Independent operator verification failed")
        if record["acquired"]:
            start = vector(record["original"])
            start[0] = Q(0)
            reconstructed = [a+sum(Q(c)*Q(d[i]) for c, d in zip(record["coefficients"], record["directions"], strict=True))
                             for i, a in enumerate(start)]
            if list(map(str, reconstructed)) != record["candidate"]:
                raise ValueError("Candidate provenance mismatch")
            acquired += 1
    known = []
    for record in parent["parent"]["discovery"]["records"]:
        row = [Q(0)]*len(WORDS)
        for word, c in zip(OLD_WORDS, record["coefficients"], strict=True):
            row[WORDS.index(word)] = Q(c)
        known.append({"coefficients": row})
    for record in result["state"]["records"]:
        if not certify(record["coefficients"])["accepted"] or not consequence_span(known).add(record["coefficients"]):
            raise ValueError("Invalid or already implied discovery")
        known.append(record)
    if acquired != result["state"]["prerequisite_credit"]["points"]:
        raise ValueError("Incorrect prerequisite credit")
    print(json.dumps({"source_files": len(manifest["files"]), "checkpoint_reconstructions": 4,
                      "verified_basis_acquisitions": acquired, "verified_relations": len(result["state"]["records"])}))


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
