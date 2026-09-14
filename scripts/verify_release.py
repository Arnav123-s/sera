"""Check published evidence hashes and source concept coverage after a checkout."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "reports/evidence_manifest.json").read_text(encoding="utf-8"))
for entry in manifest:
    path = root / "reports" / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Evidence changed: {entry['file']}")
concepts = json.loads((root / "research/physics_component_map.json").read_text(encoding="utf-8"))
if len(concepts) != 154 or len({c["id"] for c in concepts}) != 154:
    raise ValueError("Source concept coverage mismatch")
print(f"Verified {len(manifest)} evidence artifacts and 154 unique source concepts.")
