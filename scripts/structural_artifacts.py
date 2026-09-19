"""Preserve and restore the qualified field successor without copying prior archives."""

import argparse
import hashlib
import json
import shutil
import statistics
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/46_structural_refinement"
FOLDERS = tuple(ROOT / p for p in ("runs/SF-study-001", "runs/sera-structural-live", "runs/sera-structural-development",
                                  "runs/SF-v16-verification", "runs/CI-user-queries"))
ARCHIVE = OUT / "research.zip.part001"
TAG = "research-2026-09-18-structural"
ASSET = "sera-46-structural-refinement.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def analyze():
    """Aggregate saved predictions; never fit or reopen an assessment process."""
    audit = read(OUT / "audit.json")
    rows = [read(p) for p in sorted((ROOT / "runs/SF-study-001/final/completed").glob("*.json"))]
    if len(rows) != audit["exact_replay_episodes"]:
        raise ValueError("The audited final cohort changed")
    groups = []
    for family in ("radial", "directional", "omitted"):
        for policy in ("disagreement", "balanced"):
            selected = [r for r in rows if r["family"] == family and r["policy"] == policy]
            plans = []
            for record in selected:
                plan = record["plan"]
                if plan["selected"] is None:
                    continue
                chosen = next(b for b in plan["branches"] if b["control"] == plan["selected"]["control"])
                best_error = min(b["independent_goal_error"] for b in plan["branches"])
                plans.append({"subject": record["subject"], "selected_control": chosen["control"],
                              "regret_m": chosen["independent_goal_error"] - best_error,
                              "best_of_five": chosen["independent_goal_error"] <= best_error + 1e-10})
            groups.append({"family": family, "policy": policy, "worlds": len(selected),
                           "original_goal_initial_mse": statistics.mean(r["original_task"]["initial_mse"] for r in selected),
                           "original_goal_final_mse": statistics.mean(r["original_task"]["final_mse"] for r in selected),
                           "qualified_plans": len(plans), "independently_best_of_five": sum(r["best_of_five"] for r in plans),
                           "mean_plan_regret_m": statistics.mean(r["regret_m"] for r in plans) if plans else None,
                           "plans": plans})
    write(OUT / "analysis.json", {"owner": audit["owner"], "episodes": len(rows), "distinct_worlds": len(rows)//2,
          "paired_seed_triplets": 12, "groups": groups, "observed_pair_exposures": sum(r["observed_pairs"] for r in rows),
          "final_prediction_exposures": sum(len(r["final_x"]) for r in rows),
          "distinct_world_query_pairs": sum(len(r["final_x"]) for r in rows if r["policy"] == "disagreement"),
          "wide_prediction_exposures": sum(len(r["wide_x"]) for r in rows),
          "trajectory_comparisons": sum(len(r["trajectories"]) for r in rows),
          "source_kind": "SIMULATED_OBSERVATION", "paired_controls_share_sources": True})
    print(json.dumps({"groups": [{k: v for k, v in g.items() if k != "plans"} for g in groups]}, indent=2))


def costs():
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith("SF-") and v["status"] not in {"RESERVED", "RUNNING"}}
    for key in jobs:
        for name in ("state.json", "process.log", "junit.xml"):
            source = ROOT / key / name
            if source.exists():
                dest = OUT / "publication/checks" / (Path(key).name + "-" + name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
    write(OUT / "publication/costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
          "one_cpu_thread": True, "memory_cap_bytes": 2147483648, "paid_services": False,
          "unlimited_local_time": ledger["unlimited_local_time"], "counterfactual_parent_costs": "../45_counterfactual_inquiry/publication/costs.json"})
    write(OUT / "publication/resource-ledger.json", ledger)


def sources():
    paths = [*ROOT.glob("experiments/structural_*.py"), *ROOT.glob("scripts/structural_*.py"),
             ROOT / "scripts/run_structural_persistent.py", *ROOT.glob("tests/test_structural*.py")]
    paths += [p for p in OUT.rglob("*") if p.is_file() and "publication" not in p.parts and
              p.name not in {"research.zip.part001", "release-manifest.json", "checklist.md"}]
    return sorted(set(paths))


def seal():
    if ARCHIVE.exists() or (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the sealed structural release")
    prior = read(ROOT / "research-continuation/45_counterfactual_inquiry/release-manifest.json")
    owned_by_parent = {m["path"] for m in prior["members"]}
    completed_jobs = [p for p in (ROOT / "runs").glob("SF-*") if p.is_dir() and (p / "state.json").exists()
                      and read(p / "state.json").get("status") not in {None, "RUNNING", "RESERVED"}]
    files = {p for folder in (*FOLDERS, *completed_jobs) for p in folder.rglob("*")
             if p.is_file() and p.relative_to(ROOT).as_posix() not in owned_by_parent}
    members = []
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            members.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    write(OUT / "release-manifest.json", {"schema": "sera.structural-release.1", "archive": sha(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
          "asset": ASSET, "url": f"https://github.com/Arnav123-s/sera/releases/download/{TAG}/{ASSET}", "members": members,
          "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources()],
          "parent": "research-continuation/45_counterfactual_inquiry/release-manifest.json",
          "preservation": "All acquired observations, competing models, withheld results, interruptions and exact successor state; prior archives stay separate"})
    verify()


def verify(restore=False):
    manifest = read(OUT / "release-manifest.json")
    for item in manifest["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Changed structural source or evidence: " + item["path"])
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".download")
        with urllib.request.urlopen(manifest["url"], timeout=60) as response, temporary.open("xb") as dest:
            shutil.copyfileobj(response, dest, length=1048576)
        if sha(temporary) != manifest["archive"]:
            raise ValueError("Downloaded structural archive differs")
        temporary.replace(ARCHIVE)
    if sha(ARCHIVE) != manifest["archive"] or ARCHIVE.stat().st_size != manifest["bytes"]:
        raise ValueError("Changed structural archive")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(m["path"] for m in manifest["members"]):
            raise ValueError("Unexpected or duplicate archived entries")
        for item in manifest["members"]:
            name = item["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archived path")
            path = (ROOT / name).resolve()
            job_path = len(relative.parts) >= 3 and relative.parts[0] == "runs" and relative.parts[1].startswith("SF-")
            if not any(path.is_relative_to(folder) for folder in FOLDERS) and not job_path:
                raise ValueError("Archive escaped its owned work")
            raw = archive.read(name)
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Archived evidence changed")
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
    p = argparse.ArgumentParser()
    p.add_argument("--seal", action="store_true")
    p.add_argument("--restore", action="store_true")
    p.add_argument("--costs", action="store_true")
    p.add_argument("--analyze", action="store_true")
    a = p.parse_args()
    if a.seal:
        seal()
    elif a.costs:
        costs()
    elif a.analyze:
        analyze()
    else:
        verify(a.restore)


if __name__ == "__main__":
    main()
