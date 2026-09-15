"""Verify that GG-GUARD-001 leaves every inventoried predecessor byte intact."""

import gzip
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/15_applicability"


def main():
    destination = RELEASE/"preservation.json"
    if destination.exists():
        raise ValueError("Preservation audit already recorded; use its immutable evidence")
    started = time.perf_counter()
    raw = (RELEASE/"predecessor-inventory.json.gz").read_bytes()
    baseline = json.loads(gzip.decompress(raw))
    if len(baseline["files"]) != baseline["file_count"]:
        raise ValueError("Predecessor inventory count mismatch")
    checked = 0
    for row in baseline["files"]:
        path = (ROOT/row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Invalid predecessor path")
        with path.open("rb") as stream:
            current = hashlib.file_digest(stream, "sha256").hexdigest()
        if current != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Predecessor changed: {row['path']}")
        checked += 1
    protocol = json.loads((RELEASE/"protocol.json").read_text())
    owner_sources = {n: v for n, v in protocol["payload"]["sources"].items() if n.startswith("src/sera/")}
    for name, record in owner_sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Existing shared owner/interpreter changed")
    result = {"status": "PASS", "unchanged_files": checked, "unchanged_bytes": baseline["bytes"],
              "unchanged_owner_source_files": len(owner_sources), "parent_commit": baseline["parent_commit"],
              "inventory_sha256": hashlib.sha256(raw).hexdigest(), "wall_seconds": time.perf_counter()-started,
              "boundary": "Byte preservation of the listed local predecessors and shared source. "
                          "The 40-capability score comparison remains the previous SHARED-GG-001 result; "
                          "this component made no shared-owner update or new transfer claim."}
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
