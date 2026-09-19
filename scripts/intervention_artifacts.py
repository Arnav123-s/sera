"""Preserve and safely restore the qualified intervention successor and evidence."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/47_intervention_understanding"
ARCHIVE = OUT / "research.zip.part001"
TAG = "research-2026-09-18-interventions"
ASSET = "sera-47-intervention-understanding.zip"
FOLDERS = tuple(ROOT / p for p in (
    "runs/IU-study-001", "runs/sera-intervention-live", "runs/IU-v18-verification", "runs/IU-intake", "runs/IU-verification-001",
    "research/intake/v17-understanding", "research/intake/v18-understanding"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith("IU-") and v["status"] not in {"RUNNING", "RESERVED"}}
    for key in jobs:
        for name in ("state.json", "process.log", "junit.xml"):
            source = ROOT / key / name
            if source.exists():
                destination = OUT / "publication/checks" / (Path(key).name+"-"+name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    write(OUT / "publication/costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
          "one_cpu_thread": True, "memory_cap_bytes": 2147483648, "paid_services": False,
          "unlimited_local_time": ledger["unlimited_local_time"], "parent_costs": "../46_structural_refinement/publication/costs.json"})
    write(OUT / "publication/resource-ledger.json", ledger)


def analyze():
    rows = [read(p)["result"] for p in sorted((ROOT / "runs/IU-study-001/final/completed").glob("*.json"))]
    supported = [r for r in rows if r["policy"] == "information" and r["family"] != "omitted"]
    def mean(items):
        items = list(items)
        return sum(items)/len(items)
    before = mean(r["original_initial_squared_error"] for r in supported)
    after = mean(r["original_final_squared_error"] for r in supported)
    known = [r for r in rows if r["policy"] == "information"]
    records = [{"subject": r["subject"], "initial_probability": r["initial_answer"]["plus_probability"],
                "returned_probability": r["answer"]["plus_probability"], "independent_expectation": r["original_truth"],
                "accepted": r["accepted"], "selected": r["selected"],
                "actual_pulse_application_fraction": r["qualification"]["actuation"]["observed_application_fraction"]} for r in known]
    result = {"distinct_worlds": len(rows)//3, "episodes": len(rows), "supported_information_worlds": len(supported),
              "qualified_supported_information_worlds": sum(r["accepted"] for r in supported),
              "original_initial_mse": before, "original_final_mse": after, "original_error_reduction_percent": 100*(1-after/before),
              "supported_final_mse": mean(r["mse"] for r in supported),
              "supported_control_mse": {k: mean(r["controls"][k]["mse"] for r in supported) for k in supported[0]["controls"]},
              "final_unseen_programs_per_world": len(rows[0]["final_programs"]),
              "final_distinct_world_programs": sum(len(r["final_programs"]) for r in known),
              "final_prediction_exposures": sum(len(r["final_programs"]) for r in rows),
              "wide_prediction_exposures": sum(len(r["wide_programs"]) for r in rows),
              "all_false_admissions": [r["subject"] for r in rows if r["family"] == "omitted" and r["accepted"]],
              "supported_rejections": [r["subject"] for r in rows if r["family"] != "omitted" and r["policy"] != "passive" and not r["accepted"]],
              "returned_answers": records}
    write(OUT / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "returned_answers"}, indent=2))


def sources():
    paths = [*ROOT.glob("experiments/intervention_*.py"), *ROOT.glob("scripts/intervention_*.py"),
             ROOT / "scripts/run_intervention_persistent.py", *ROOT.glob("tests/test_intervention*.py")]
    paths += [p for p in OUT.rglob("*") if p.is_file() and "publication" not in p.parts and
              p.name not in {"release-manifest.json", "research.zip.part001", "checklist.md"}]
    return sorted(set(paths))


def seal():
    if ARCHIVE.exists() or (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the sealed intervention release")
    if not read(OUT / "audit.json")["passed"]:
        raise ValueError("Complete independent audit before sealing")
    jobs = [p for p in (ROOT / "runs").glob("IU-*") if p.is_dir() and (p / "state.json").exists()
            and read(p / "state.json").get("status") not in {None, "RUNNING", "RESERVED"}]
    files = {p for folder in (*FOLDERS, *jobs) for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    members = []
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            members.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    write(OUT / "release-manifest.json", {"schema": "sera.intervention-release.1", "archive": sha(ARCHIVE),
          "bytes": ARCHIVE.stat().st_size, "asset": ASSET,
          "url": f"https://github.com/Arnav123-s/sera/releases/download/{TAG}/{ASSET}", "members": members,
          "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources()],
          "parent": "research-continuation/46_structural_refinement/release-manifest.json",
          "preservation": "All new acquired evidence, competing weights, failed attempts, packet source, replay and current successor. Predecessor archives remain separate."})
    verify()


def verify(restore=False):
    manifest = read(OUT / "release-manifest.json")
    for item in manifest["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Changed intervention source or evidence: "+item["path"])
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".download")
        with urllib.request.urlopen(manifest["url"], timeout=60) as response, temporary.open("xb") as output:
            shutil.copyfileobj(response, output, length=1048576)
        if sha(temporary) != manifest["archive"]:
            raise ValueError("Downloaded archive differs")
        temporary.replace(ARCHIVE)
    if sha(ARCHIVE) != manifest["archive"] or ARCHIVE.stat().st_size != manifest["bytes"]:
        raise ValueError("Changed intervention archive")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(m["path"] for m in manifest["members"]):
            raise ValueError("Unexpected or duplicate archived entries")
        for item in manifest["members"]:
            name = item["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path")
            path = (ROOT / name).resolve()
            job = len(relative.parts) >= 3 and relative.parts[0] == "runs" and relative.parts[1].startswith("IU-")
            if not any(path.is_relative_to(folder) for folder in FOLDERS) and not job:
                raise ValueError("Archive escaped its owned directories")
            raw = archive.read(name)
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Archived evidence changed")
            if restore:
                if path.exists():
                    if sha(path) != item["sha256"]:
                        raise ValueError("Preserve newer local work: "+name)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as file:
                        file.write(raw)
    print(f"Verified {len(manifest['members'])} archived records and {len(manifest['files'])} source/evidence files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--costs", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.costs:
        costs()
    elif args.analyze:
        analyze()
    else:
        verify(args.restore)


if __name__ == "__main__":
    main()
