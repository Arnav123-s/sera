"""Verify the compact continuation release and its frozen-source witnesses."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research-continuation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    release = WORK / "14_release"
    manifest = read(release / "release-manifest.json")
    entries = manifest["files"]
    if not entries or len({row["path"] for row in entries}) != len(entries):
        raise ValueError("Empty or duplicate continuation evidence inventory")
    for row in entries:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or invalid evidence path: {row['path']}")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Continuation evidence changed: {row['path']}")
    members = read(release / "frozen-source-members.json")
    with zipfile.ZipFile(release / "frozen-protocol-sources.zip") as archive:
        if len(archive.namelist()) != len(members) or set(archive.namelist()) != set(members):
            raise ValueError("Frozen source inventory differs")
        for name, row in members.items():
            raw = archive.read(name)
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"Frozen source member differs: {name}")
    act = read(release / "GG-ACT-001-summary.json")
    assert act["complete_policy_records"] == len(act["raw_final_rows"]) == 300
    assert len({(row["case"], row["policy"]) for row in act["raw_final_rows"]}) == 300
    replay = read(release / "GG-ACT-001-replay.json")
    assert replay["status"] == "passed" and replay["records_replayed"] == 300
    assert replay["observation_steps_replayed"] == 3000
    independent = read(WORK / "01_audit/GG-ACT-001-independent.json")
    assert independent["status"] == "PASS" and independent["independent_batch_gaussians"] == 12000
    integrated = read(release / "SHARED-GG-001-result.json")
    assert integrated["status"] == "PASS" and len(integrated["families"]) == 4
    assert integrated["retention_exact"] and integrated["retention_capabilities"] == 40
    assert integrated["retention_scores_compared"] == 944
    xml = ET.parse(release / "current-tests.xml").getroot()
    assert sum(int(s.attrib["tests"]) for s in xml.findall("testsuite")) == 156
    assert sum(int(s.attrib["failures"])+int(s.attrib["errors"]) for s in xml.findall("testsuite")) == 0
    preservation = read(WORK / "01_audit/preservation.json")
    assert preservation["status"] == "PASS" and preservation["baseline_files"] == 2404
    print(f"Verified {len(entries)} continuation files, {len(members)} frozen source members, "
          "300 inquiry runs and trained-parent retention evidence.")


if __name__ == "__main__":
    main()
