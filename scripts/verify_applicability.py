"""Verify the published applicability evidence, source versions and gate outcome."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/15_applicability"


def read(name):
    return json.loads((RELEASE/name).read_text(encoding="utf-8"))


def main():
    manifest = read("release-manifest.json")
    entries = manifest["files"]
    if not entries or len(entries) != len({r["path"] for r in entries}):
        raise ValueError("Empty or duplicate applicability inventory")
    for row in entries:
        path = (ROOT/row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Invalid evidence path")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Applicability evidence changed: {row['path']}")
    protocol = read("protocol.json")
    canonical = json.dumps(protocol["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != protocol["sha256"]:
        raise ValueError("Frozen protocol integrity failure")
    sources = protocol["payload"]["sources"]
    with zipfile.ZipFile(RELEASE/"frozen-sources.zip") as archive:
        if len(archive.namelist()) != len(sources) or set(archive.namelist()) != set(sources):
            raise ValueError("Frozen source inventory differs")
        for name, record in sources.items():
            raw = archive.read(name)
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Frozen experiment source bytes differ")
    repair = read("audit-repair.json")
    old = sources[repair["path"]]["sha256"]
    current = hashlib.sha256((ROOT/repair["path"]).read_bytes()).hexdigest()
    if old != repair["original_sha256"] or current != repair["corrected_sha256"]:
        raise ValueError("Independent auditor revision is not the declared repair")
    summary, replay, audit = read("summary.json"), read("replay.json"), read("independent.json")
    if summary["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("Summary belongs to another protocol")
    if (replay["status"] != "PASS" or replay["worlds_replayed"] != 1344
            or not replay["exact_model_refit"] or not replay["exact_thresholds"]
            or not replay["exact_summary_and_curves"]):
        raise ValueError("Incomplete same-interpreter replay")
    if audit["status"] != "PASS" or audit["worlds_audited"] != 1344 or audit["auditor_sha256"] != current:
        raise ValueError("Independent corrected audit incomplete")
    if hashlib.sha256((RELEASE/"records.json.gz").read_bytes()).hexdigest() != summary["records_sha256"]:
        raise ValueError("Raw evidence archive differs")
    records = json.loads(gzip.decompress((RELEASE/"records.json.gz").read_bytes()))["records"]
    if len(records) != 1344 or len({r["episode_id"] for r in records}) != 1344:
        raise ValueError("World inventory mismatch")
    final = [r for r in records if r["split"] == "final"]
    if len(final) != 384 or any(len(r["observed_valid"]) != 33 for r in records):
        raise ValueError("Final or per-world query counts differ")
    if summary["gate"]["component_promotion"] or summary["gate"]["risk_at_most_10_percent_each_family_with_5_percent_coverage"]:
        raise ValueError("Failed guard was incorrectly promoted")
    preservation = read("preservation.json")
    if preservation["status"] != "PASS" or preservation["unchanged_files"] != 16556:
        raise ValueError("Prior model/result preservation incomplete")
    suites = ET.parse(RELEASE/"current-tests.xml").getroot().findall("testsuite")
    if sum(int(s.attrib["failures"])+int(s.attrib["errors"]) for s in suites):
        raise ValueError("Current release tests failed")
    print(f"Verified {len(entries)} applicability files, 1344 worlds, 44352 query predictions, "
          f"{len(sources)} frozen sources, corrected independent audit and withheld promotion.")


if __name__ == "__main__":
    main()
