"""Preserve complete counterfactual investigations and restore exact qualified checkpoints."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/45_counterfactual_inquiry"
RUN = ROOT / "runs/CI-study-001"
QUALIFIED = ROOT / "runs/CI-qualified-001"
FOLDERS = (RUN, QUALIFIED, ROOT / "runs/sera-counterfactual-live", ROOT / "runs/sera-counterfactual-qualified-live", ROOT / "runs/CI-user-queries")
ARCHIVE = OUT / "research.zip.part001"
TAG = "research-2026-09-18-counterfactual"
ASSET = "sera-45-counterfactual-inquiry.zip"


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
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith("CI-") and v["status"] not in {"RESERVED", "RUNNING"}}
    for key in jobs:
        for name in ("state.json", "process.log", "junit.xml"):
            source = ROOT / key / name
            if source.exists():
                dest = target / "checks" / (Path(key).name + "-" + name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
              "unlimited_local_time": ledger["unlimited_local_time"], "one_cpu_thread": True, "memory_cap_bytes": 2147483648,
              "original_procedure_training_seconds": read(ROOT / "runs/CI-train-001/state.json")["charged_seconds"],
              "qualified_procedure_training_seconds": read(ROOT / "runs/CI-guard-train-001/state.json")["charged_seconds"]})
    write(target / "resource-ledger.json", ledger)


def sources():
    files = list((ROOT / "experiments").glob("counterfactual_*.py"))
    files += list((ROOT / "scripts").glob("counterfactual_*.py"))
    files += [ROOT / "scripts/run_counterfactual_persistent.py"]
    files += list((ROOT / "tests").glob("test_counterfactual*.py"))
    files += [p for p in OUT.rglob("*") if p.is_file() and "publication" not in p.parts
              and p.name not in {"research.zip.part001", "release-manifest.json", "checklist.md"}]
    return sorted(set(files))


def analyze():
    """Summarize saved outcomes without reopening or rerunning an evaluation."""
    audit = read(OUT / "audit.json")
    selection = read(OUT / "qualification/selection.json")
    ref = read(ROOT / "runs/sera-counterfactual-qualified-live/current.json")
    owner = read(ROOT / "runs/sera-counterfactual-qualified-live/revisions" / ref["revision"])["state"]
    proposed, proved, steps, follows, cases = set(), set(), 0, {}, []
    domains = {}
    for goal, record in owner["records"].items():
        result = read(ROOT / record["artifact"]["path"])
        if sha(ROOT / record["artifact"]["path"]) != record["artifact"]["sha256"]:
            raise ValueError("A retained investigation record changed")
        for event in result["events"]:
            proposed.update((goal, c["id"]) for c in event["candidates"])
        steps += len(result["observations"])
        for followup in result["followups"]:
            follows[followup["kind"]] = follows.get(followup["kind"], 0) + 1
        domains[record["domain"]] = domains.get(record["domain"], 0) + 1
        cases.append({"goal": goal, "domain": record["domain"], "axis": record["axis"], "target": record["target"],
                      "survivors": len(record["survivors"]), "status": record["status"], "observations": record["observations"],
                      "followups": len(record["followups"]), "rules_added": len(record["rules_added"]), "points": record["points"],
                      "artifact": record["artifact"]})
    for file in QUALIFIED.glob("*/proofs/*.json"):
        receipt = read(file)
        proved.update((receipt["question"], r["candidate"]) for r in receipt["proofs"] if r["accepted"])
    if proposed != proved:
        raise ValueError("The retained proposal and proof portfolios differ")
    metrics = []
    for path in QUALIFIED.glob("*/done/*.json"):
        metrics.extend(read(path)["metrics"])
    report = {"owner": audit["owner"], "questions": len(cases), "domains": domains,
              "distinct_conditional_worlds": len(proposed), "independently_proved_worlds": len(proved),
              "selected_trajectory_observations": steps, "followups": follows,
              "answered_with_portfolio": sum(r["status"] == "SCOPED_ANSWER" for r in cases),
              "retained_rules": len(owner["rules"]), "new_shape_points": sum(r["points"] for r in owner["credits"].values()),
              "comparison_trajectories": len(metrics), "comparison_observation_attempts": sum(r["observations"] for r in metrics),
              "selected_policy": selection["policy"], "practice": read(OUT / "qualification/train.json"),
              "validation": read(OUT / "qualification/validation.json"), "final": read(OUT / "qualification/final.json"),
              "audit": audit, "question_records": cases}
    write(OUT / "analysis.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"question_records", "practice", "validation", "final", "audit"}}, indent=2))


def seal():
    if ARCHIVE.exists() or (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the sealed solution release")
    completed_jobs = [p for p in (ROOT / "runs").glob("CI-*") if p.is_dir() and (p / "state.json").exists()
                      and read(p / "state.json").get("status") not in {None, "RUNNING", "RESERVED"}]
    query_root = ROOT / "runs/CI-user-queries"
    owner = read(OUT / "audit.json")["owner"]
    query_folders = [p.parent for p in query_root.glob("*/proposal.json") if read(p)["predictor"] == owner]
    folders = [p for p in FOLDERS if p != query_root]
    files = {p for folder in (*folders, *query_folders, *completed_jobs) for p in folder.rglob("*") if p.is_file()}
    members = []
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            members.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    write(OUT / "release-manifest.json", {"schema": "sera.counterfactual-release.1", "archive": sha(ARCHIVE),
          "bytes": ARCHIVE.stat().st_size, "asset": ASSET,
          "url": f"https://github.com/Arnav123-s/sera/releases/download/{TAG}/{ASSET}", "members": members,
          "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources()],
          "parent": "research-continuation/44_solution_portfolios/release-manifest.json",
          "preservation": "Complete proposals, simulations, failures, comparisons, learned rules and exact resumable checkpoints"})
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
            job_path = len(relative.parts) >= 3 and relative.parts[0] == "runs" and relative.parts[1].startswith("CI-")
            if not any(path.is_relative_to(p) for p in FOLDERS) and not job_path:
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
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.analyze:
        analyze()
    elif args.costs:
        costs()
    else:
        verify(args.restore)


if __name__ == "__main__":
    main()
