"""Read-only verification of the usable application release and rejected candidate."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/20_live_workbench"


def read(name):
    return json.loads((RELEASE/name).read_text())


def main():
    manifest = read("release-manifest.json")
    for row in manifest["artifacts"]:
        raw = (ROOT/row["path"]).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Changed released artifact: "+row["path"])
    with zipfile.ZipFile(RELEASE/"sources.zip") as archive:
        if set(archive.namelist()) != set(manifest["sources"]):
            raise ValueError("Source archive is incomplete")
        for name, sha in manifest["sources"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("Changed frozen source")
    qualification = read("ONDEMAND-001/summary.json")
    assert qualification["status"] == "PASS" and qualification["accepted"] == 60 and qualification["withheld"] == 30
    assert qualification["independently_checked_new_queries"] == 540 and qualification["maximum_accepted_query_error"] < 1e-6
    eta = read("A09-FINAL-001/summary.json")
    assert eta["status"] == "REJECTED" and not eta["gate"]["strong_controls"]
    assert read("A09-FINAL-001/audit.json")["checked_predictions"] == 4320
    first = (RELEASE/"interrupted-regression.log").read_text()
    completed = sum(len(re.match(r"^\.+", line).group()) for line in first.splitlines() if re.match(r"^\.+", line))
    assert completed == 202
    suites = ET.parse(RELEASE/"remainder-tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 31
    assert sum(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")) == 0
    retained = read("retention/result.json")
    assert retained["groups"] == 61 and retained["exact_group_and_case_scores"]
    snapshot = read("application-snapshot.json")
    assert len(snapshot["interaction"]["events"]) == 85 and len(snapshot["contexts"]) == 3
    assert read("preservation.json")["previous_release_files_unchanged"] == 356
    assert read("resource-grants/grant.json")["added_seconds"] == 3600
    print(json.dumps({"status": "PASS", "artifact_files": len(manifest["artifacts"]),
                      "frozen_source_files": len(manifest["sources"]), "tests_completed_across_runs": 233,
                      "full_suite_timeout_preserved": True, "retention_groups": 61,
                      "task_qualification": qualification["status"], "eta_promotion": False,
                      "new_experiments": 0}))


if __name__ == "__main__":
    main()
