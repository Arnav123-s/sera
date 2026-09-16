"""Read-only sparse-release hash/receipt audit; no numerical jobs or refitting."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/22_sparse_mechanisms"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024*1024), b""):
            result.update(piece)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read(RELEASE/"release-manifest.json")
    expected = set()
    for item in manifest["artifact_files"]:
        path = (ROOT/item["path"]).resolve()
        require(path.is_relative_to(RELEASE) and path.is_file(), "Artifact escaped or disappeared")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Artifact changed: "+item["path"])
        require(item["path"] not in expected, "Duplicate artifact")
        expected.add(item["path"])
    present = {p.relative_to(ROOT).as_posix() for p in RELEASE.rglob("*") if p.is_file() and p.name not in {"release-manifest.json", "verification.json"}}
    require(expected == present, "Uninventoried or missing release file")
    sources = read(RELEASE/"source-manifest.json")["sources"]
    with zipfile.ZipFile(RELEASE/"sources.zip") as archive:
        require(set(archive.namelist()) == {r["path"] for r in sources}, "Source closure differs")
        for item in sources:
            payload = archive.read(item["path"])
            require(len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"], "Frozen source changed")
    cohorts = 0
    for folder in sorted(RELEASE.glob("CS-*")):
        if not folder.is_dir():
            continue
        protocol = read(folder/"protocol.json")
        if "sources" not in protocol:
            continue
        with zipfile.ZipFile(folder/"sources.zip") as archive:
            require(set(archive.namelist()) == set(protocol["sources"]), "Cohort sources differ")
            for name, digest in protocol["sources"].items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Cohort source hash differs")
        cohorts += 1
    costs = read(RELEASE/"costs.json")
    with zipfile.ZipFile(RELEASE/"supervision.zip") as archive:
        for job in costs["owned_jobs"]:
            data = archive.read(job["job"]+"/state.json")
            require(hashlib.sha256(data).hexdigest() == job["state_sha256"], "Changed job receipt")
            state = json.loads(data)
            require(state["status"] == job["status"], "Job status differs")
            require(state["status"] == "PASS" or job["expected_rejection"], "Unresolved numerical failure")
            require(hashlib.sha256(archive.read(job["job"]+"/process.log")).hexdigest() == state["log_sha256"], "Changed job log")
            require(state["worker_threads"] == 1 and state["peak_job_committed_bytes"] <= state["memory_limit_bytes"], "Resource condition failed")
    charged = sum(row["charged_seconds"] for row in costs["owned_jobs"])
    require(abs(charged-costs["cycle_charged_seconds"]) < 1e-7, "Cost sum differs")
    tests = read(RELEASE/"test-summary.json")
    distinct = set()
    for report in tests["reports"]:
        root = ET.parse(ROOT/report["path"]).getroot()
        require(not list(root.iter("failure")) and not list(root.iter("error")), "Contract test failed")
        distinct.update((r.get("classname"), r.get("name")) for r in root.iter("testcase"))
    require(len(distinct) == tests["distinct_targeted_tests"], "Test count differs")
    results = read(RELEASE/"results.json")
    require(all(v == "PASS" for v in results["final_audits"].values()), "Missing independent final audit")
    require(results["promotion"]["mechanism_supported_answers"] == "REJECTED_FOR_INTEGRATION", "Failed mechanism gate was promoted")
    require(results["promotion"]["disagreement_investigator"] == "REJECTED_FOR_PROMOTION", "Failed investigator gate was promoted")
    require(read(RELEASE/"CS-REAL-MEMORY-001/result.json")["silent_wrong_restores"] == 0, "Memory utility returned wrong bytes")
    print(json.dumps({"status": "PASS", "artifact_files": len(expected), "frozen_source_files": len(sources),
                      "cohort_source_archives": cohorts, "numerical_jobs": len(costs["owned_jobs"]),
                      "distinct_targeted_tests": len(distinct), "new_final_records": results["new_final_records"],
                      "charged_seconds": charged, "new_experiments": 0,
                      "scope": "Frozen bytes, source/receipt integrity and declared gate consistency. Independent numerical checks are preserved; no repeated fitting."}))


if __name__ == "__main__":
    main()
