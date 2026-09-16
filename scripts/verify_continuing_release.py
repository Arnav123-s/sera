"""Verify saved A08 evidence and frozen versions without resampling any worlds."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/19_continuing"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check(entry):
    path = (ROOT/entry["path"]).resolve()
    require(path.is_relative_to(ROOT), "Path escapes repository")
    raw = path.read_bytes()
    require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], f"Changed artifact: {entry['path']}")


def archive_check(path, sources):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(sources) and set(archive.namelist()) == set(sources), "Incomplete frozen version")
        for name, sha in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha, f"Changed frozen source: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-checkpoints", action="store_true")
    args = parser.parse_args()
    manifest = read(RELEASE/"release-manifest.json")
    entries = manifest["files"]
    require(entries and len({r["path"] for r in entries}) == len(entries), "Incomplete release inventory")
    for entry in entries:
        check(entry)
    archive_check(RELEASE/"current-sources.zip", read(RELEASE/"current-source-manifest.json"))
    for name in ("A08-PILOT-001", "A08-PILOT-002", "A08-FINAL-001"):
        protocol = read(RELEASE/name/"protocol.json")
        archive_check(RELEASE/name/"frozen-sources.zip", protocol["payload"]["sources"])
        require(read(RELEASE/name/"summary.json")["status"] == "COMPLETE", "Incomplete lifetime cohort")
    audit = read(RELEASE/"A08-FINAL-001/independent-audit.json")
    require(audit["status"] == "PASS" and audit["lifetimes"] == 168 and audit["independent_action_checks"] == 12096,
            "Missing independent evidence")
    require(audit["gate"]["component_pass"] and max(audit["maximum_differences"].values()) < 1e-9, "Unexpected qualification result")
    require(len({(r["seed"], r["family"]) for r in audit["records"]}) == 24, "Incorrect independent world count")
    for key in ("adaptive_reactive", "adaptive_one_step"):
        result = audit["paired_nominal_comparisons"][key]
        require(result["relative_cost_reduction"] >= .1 and result["interval_97_5_percent"][0] > 0, "Incomplete primary comparison")
    retention = read(RELEASE/"retention/result.json")
    require(retention["status"] == "PASS" and retention["groups"] == 61
            and retention["group_scores_and_all_saved_case_scores_exact"] and retention["added_trainable_parameters"] == 0,
            "Missing retained-behavior result")
    owner = read(RELEASE/"integration/owner.json")
    for name, sha in owner["files"].items():
        require(hashlib.sha256((RELEASE/"integration"/name).read_bytes()).hexdigest() == sha, "Integrated dependency changed")
    require(owner["new_observation_count"] == 79 and owner["new_action_count"] == 78
            and owner["report"]["old_factual_observations"] == 16 and owner["report"]["old_corrections"] == 1
            and len(owner["report"]["stale_bindings_rejected"]) == 2, "Incomplete continuing persistence")
    require(read(RELEASE/"integration/reload.json")["status"] == "PASS", "Missing complete reload")
    tests = ET.parse(RELEASE/"tests.xml").getroot().findall("testsuite")
    require(sum(int(t.attrib["tests"]) for t in tests) == 219, "Unexpected regression count")
    require(sum(int(t.attrib.get(k, 0)) for t in tests for k in ("errors", "failures", "skipped")) == 0, "Regression failure or skip")
    costs, budget = read(RELEASE/"costs.json"), read(RELEASE/"worker-budget.json")
    require(math.isclose(sum(j["charged_seconds"] for j in costs["jobs"].values()), costs["charged_worker_seconds"]), "Incomplete full costs")
    require(math.isclose(sum(j["charged_seconds"] for j in budget["jobs"].values())+budget["remaining_seconds"], budget["initial_seconds"]),
            "Cumulative budget differs")
    require(costs["peak_job_committed_bytes"] <= budget["memory_limit_bytes"], "Memory limit exceeded")
    old = read(RELEASE/"preservation.json")
    require(old["status"] == "PASS" and old["unchanged_previous_release_files"] == 451
            and old["unchanged_original_predecessors"] == 108 and len(old["unchanged_shared_sources"]) == 45,
            "Missing original preservation evidence")
    local = read(RELEASE/"local-artifacts.json")
    if args.local_checkpoints:
        for entry in local["files"]+[local["immutable_base"]]:
            check(entry)
    print(json.dumps({"status": "PASS", "release_files": len(entries), "prospective_lifetimes": 168,
                      "development_lifetimes": 42, "independent_worlds": 24, "audited_decisions": 12096,
                      "tests": 219, "retention_groups": 61, "resumed_observations": 79,
                      "local_checkpoints_checked": len(local["files"]) if args.local_checkpoints else 0,
                      "new_experiments_run": 0, "uncertainty_certificate": False}))


if __name__ == "__main__":
    main()
