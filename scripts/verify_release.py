"""Check published evidence hashes and source concept coverage after a checkout."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "reports/evidence_manifest.json").read_text(encoding="utf-8"))
connected = root / "reports/connected-evidence-manifest.json"
if connected.exists():
    manifest += json.loads(connected.read_text(encoding="utf-8"))
for entry in manifest:
    path = root / "reports" / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Evidence changed: {entry['file']}")
concepts = json.loads((root / "research/physics_component_map.json").read_text(encoding="utf-8"))
if len(concepts) != 154 or len({c["id"] for c in concepts}) != 154:
    raise ValueError("Source concept coverage mismatch")
if connected.exists():
    data = json.loads((root / "reports/connected-study-data.json").read_text(encoding="utf-8"))
    audited = json.loads((root / "reports/connected-verification.json").read_text(encoding="utf-8"))
    if audited["source_sha256"] != data["manifest"]["environment"]["source_sha256"]:
        raise ValueError("Reproduction audit used a different source")
    if [r["seed"] for r in data["runs"]] != data["manifest"]["seeds"]:
        raise ValueError("Connected study seed coverage mismatch")
    episodes = json.loads((root / "reports/connected-policy-episodes.json").read_text(encoding="utf-8"))
    seen, queries = set(), set()
    for row in episodes:
        ids = set(row["support_record_ids"])
        if seen.intersection(ids) or row["query_dataset_id"] in queries:
            raise ValueError("Meta-episode evidence reused")
        seen.update(ids)
        queries.add(row["query_dataset_id"])
    if len(seen) != audited["unique_meta_support_records"] or len(queries) != audited["unique_meta_query_datasets"]:
        raise ValueError("Published provenance counts differ from the audit")
print(f"Verified {len(manifest)} evidence artifacts and 154 unique source concepts.")
