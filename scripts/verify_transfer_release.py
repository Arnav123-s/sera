"""Read-only integrity and saved-evidence checks; never refit or resample."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/18_instance_transfer"
NORMALIZED = ROOT/"research-continuation/17_transfer"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(row):
    path = (ROOT/row["path"]).resolve()
    require(path.is_relative_to(ROOT), "Path outside release")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    require(path.stat().st_size == row["bytes"] and digest.hexdigest() == row["sha256"],
            f"Changed file: {row['path']}")


def check_archive(path, sources, current=False):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(sources) and set(archive.namelist()) == set(sources),
                f"Incomplete source archive: {path}")
        for name, sha in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha, f"Changed archived source: {name}")
            if current:
                require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, f"Changed current source: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-checkpoints", action="store_true")
    args = parser.parse_args()
    manifest = read(RELEASE/"release-manifest.json")
    entries = manifest["files"]
    require(entries and len({r["path"] for r in entries}) == len(entries), "Release inventory must be unique")
    for entry in entries:
        check(entry)
    check_archive(RELEASE/"current-sources.zip", read(RELEASE/"source-manifest.json"), current=True)
    shared, readout = (read(NORMALIZED/name/"audit-summary.json")
                       for name in ("A06-TRANSFER-001", "A06-TRANSFER-002"))
    for name, summary in zip(("A06-TRANSFER-001", "A06-TRANSFER-002"), (shared, readout), strict=True):
        payload = read(NORMALIZED/name/"protocol.json")["payload"]
        check_archive(NORMALIZED/name/"frozen-sources.zip", payload["sources"], current=True)
        require(summary["status"] == "PASS" and summary["checkpoint_prediction_cases_checked"] == 9600,
                "Missing normalized independent evidence")
        require(len(payload["seeds"]) == 5 and len(payload["arms"]) == 5, "Unexpected normalized study size")
    require(not shared["gate"]["conditional_component_pass"], "Rejected shared-path candidate promoted")
    require(readout["gate"]["conditional_component_pass"], "Unexpected readout component result")
    payload = read(RELEASE/"protocol.json")["payload"]
    check_archive(RELEASE/"frozen-sources.zip", payload["source_sha256"], current=True)
    require(len(payload["seeds"]) == 5 and len(payload["arms"]) == 4, "Unexpected instance study size")
    instance = read(RELEASE/"audit-summary.json")
    require(instance["status"] == "PASS" and instance["gate"]["causal_component_pass"], "Missing qualified result")
    require(instance["checkpoint_predictions_checked"] == 7680, "Missing instance predictions")
    for contrast in instance["contrasts"].values():
        require(contrast["passes"] and contrast["relative_mse_gain"] >= .1
                and min(contrast["per_stream_mse_gain"]) > 0
                and contrast["paired_bootstrap_97_5_percent"][0] > 0, "Incomplete transfer gate")
    require(instance["aggregate"]["corrected_teacher"]["maximum_retention_drop"] == 0, "Unexpected retention loss")
    tensors = read(RELEASE/"parameter-preservation-audit.json")
    require(tensors["status"] == "PASS" and tensors["checked_models"] == len(tensors["records"]) == 20,
            "Missing tensor preservation audit")
    for record in tensors["records"]:
        require(record["generator_buffers_bitwise_preserved"] and record["all_other_numerical_parameters_preserved"]
                and set(record["changed_tensors"]) == {"typed_numeric.weight", "typed_numeric.bias"},
                "Unexpected parameter scope")
    integration = read(NORMALIZED/"integration/result.json")
    require(integration["status"] == "PASS" and integration["chosen_stream"] == 307
            and integration["causal_acquired_coefficient_transfer"] and not integration["operational_solver_replaced"],
            "Unexpected experimental integration")
    require(integration["report"]["factual_observations_replayed"] == 16
            and integration["report"]["factual_corrections_replayed"] == 1
            and len(integration["report"]["stale_records_rejected"]) == 2
            and integration["report"]["parent_solver_unchanged"], "Incomplete persistence evidence")
    tests = ET.parse(NORMALIZED/"tests.xml").getroot().findall("testsuite")
    require(sum(int(t.attrib["tests"]) for t in tests) == 210, "Unexpected test count")
    require(sum(int(t.attrib.get(k, 0)) for t in tests for k in ("errors", "failures", "skipped")) == 0,
            "Local regression failure or skip")
    costs = read(RELEASE/"final-costs.json")
    budget = read(RELEASE/"worker-budget.json")
    require(math.isclose(sum(v["charged_seconds"] for v in costs["jobs"].values()), costs["charged_worker_seconds"]),
            "Incomplete worker accounting")
    require(math.isclose(sum(v["charged_seconds"] for v in budget["jobs"].values())+budget["remaining_seconds"],
                         budget["initial_seconds"]), "Cumulative ledger imbalance")
    require(len(costs["failed_execution_jobs"]) == 3, "Failed executions not preserved")
    require(costs["peak_job_committed_bytes"] <= budget["memory_limit_bytes"], "Memory cap exceeded")
    preservation = read(RELEASE/"preservation.json")
    require(preservation["status"] == "PASS" and len(preservation["unchanged_shared_sources"]) == 45
            and len(preservation["predecessor_files"]) == 108, "Missing original preservation")
    local = read(RELEASE/"local-artifact-inventory.json")
    if args.local_checkpoints:
        for entry in local["files"]+preservation["predecessor_files"]:
            check(entry)
        check(integration["lineage"]["trained_checkpoint"])
    print(json.dumps({"status": "PASS", "release_files": len(entries), "prospective_models": 70,
                      "development_models": 3, "independently_audited_predictions": 26880, "tests": 210,
                      "local_artifacts_checked": len(local["files"]) if args.local_checkpoints else 0,
                      "shared_path_gate": "FAIL", "instance_readout_gate": "PASS",
                      "new_training_or_evaluation_runs": 0}))


if __name__ == "__main__":
    main()
