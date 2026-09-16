"""Read-only evidence verification; no experiment, optimization or model execution."""

import argparse
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/23_task_transfer"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-state", action="store_true")
    args = parser.parse_args()
    manifest = read(RELEASE/"release-manifest.json")
    expected = set()
    for item in manifest["artifacts"]:
        path = (ROOT/item["path"]).resolve()
        require(path.is_relative_to(RELEASE) and path.is_file(), "Invalid artifact path")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Changed artifact: "+item["path"])
        expected.add(item["path"])
    present = {p.relative_to(ROOT).as_posix() for p in RELEASE.rglob("*") if p.is_file() and p.name not in ("release-manifest.json", "verification.json")}
    require(expected == present, "Missing or unlisted evidence")
    sources = read(RELEASE/"source-manifest.json")["sources"]
    with zipfile.ZipFile(RELEASE/"sources.zip") as z:
        require(set(z.namelist()) == {r["path"] for r in sources}, "Source closure differs")
        for item in sources:
            require(hashlib.sha256(z.read(item["path"])).hexdigest() == item["sha256"], "Frozen source differs")
    cohorts = 0
    for folder in RELEASE.glob("TT-*"):
        if not folder.is_dir():
            continue
        protocol = read(folder/"protocol.json")
        with zipfile.ZipFile(folder/"sources.zip") as z:
            require(set(z.namelist()) == set(protocol["sources"]), "Cohort closure differs")
            for name, digest in protocol["sources"].items():
                require(hashlib.sha256(z.read(name)).hexdigest() == digest, "Cohort source differs")
        cohorts += 1
    costs = read(RELEASE/"costs.json")
    with zipfile.ZipFile(RELEASE/"supervision.zip") as z:
        for row in costs["owned_jobs"]:
            data = z.read(row["job"]+"/state.json")
            require(hashlib.sha256(data).hexdigest() == row["state_sha256"], "Receipt differs")
            state = json.loads(data)
            require(state["status"] == row["status"] == "PASS", "Unresolved numerical failure")
            require(hashlib.sha256(z.read(row["job"]+"/process.log")).hexdigest() == state["log_sha256"], "Worker log differs")
            require(state["worker_threads"] == 1 and state["peak_job_committed_bytes"] <= 2147483648, "Resource ceiling violated")
    charged = sum(row["charged_seconds"] for row in costs["owned_jobs"])
    require(abs(charged-costs["cycle_charged_seconds"]) < 1e-7, "Cost sum differs")
    tests = read(RELEASE/"test-summary.json")
    unique, executions = set(), 0
    for report in tests["reports"]:
        tree = ET.parse(ROOT/report)
        require(not list(tree.iter("failure")) and not list(tree.iter("error")), "Failed targeted test")
        for case in tree.iter("testcase"):
            unique.add((case.get("classname"), case.get("name")))
            executions += 1
    require(len(unique) == tests["distinct_tests"] == 22 and executions == 26, "Test accounting differs")
    final = read(RELEASE/"TT-FINAL-001/independent-audit.json")
    math_audit = read(RELEASE/"TT-FINAL-001/mathematical-audit.json")
    require(final["status"] == math_audit["status"] == "PASS", "Final audit missing")
    require(final["numerical_promotion_condition"] and math_audit["handbook_lower_positive"], "Failed promotion condition")
    with gzip.open(RELEASE/"TT-FINAL-001/records.jsonl.gz", "rt", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    require(count == 3888 == final["records"], "Final record count differs")
    migration = read(RELEASE/"TT-MIGRATION-001/result.json")
    require(migration["status"] == "PASS" and migration["old_revisions_preserved"] == 9, "Migration record differs")
    with zipfile.ZipFile(RELEASE/"TT-OWNER-001/session-records.zip") as z:
        require(sum(p.startswith("revisions/") for p in z.namelist()) == 9, "Original history incomplete")
    with zipfile.ZipFile(RELEASE/"TT-MIGRATION-001/migration-state.zip") as z:
        pointer = json.loads(z.read("current.json"))
        payload = z.read("revisions/"+pointer["revision"])
        require(hashlib.sha256(payload).hexdigest() == pointer["sha256"], "Migration revision differs")
        require(json.loads(payload)["owner"] == migration["owner"], "Migration owner differs")
    if args.local_state:
        local = ROOT/"runs/sera-task-transfer"
        require(sha(local/"current.json") == migration["pointer_sha256"], "Local descendant advanced or changed; reconcile its newer state")
        require(read(ROOT/"runs/sera-workbench/current.json") == read(RELEASE/"application-preservation.json")["live_pointer"], "Live owner advanced or changed; reconcile its newer state")
    print(json.dumps({"status": "PASS", "artifact_files": len(expected), "source_files": len(sources),
                      "cohort_archives": cohorts, "owned_jobs": len(costs["owned_jobs"]),
                      "distinct_tests": len(unique), "test_executions": executions,
                      "final_outcomes": count, "charged_seconds": charged,
                      "remaining_at_release": costs["closing_remaining_seconds"], "new_experiments": 0,
                      "local_state_checked": args.local_state}))


if __name__ == "__main__":
    main()
