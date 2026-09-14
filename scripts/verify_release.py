"""Check published evidence hashes and source concept coverage after a checkout."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "reports/evidence_manifest.json").read_text(encoding="utf-8"))
connected = root / "reports/connected-evidence-manifest.json"
if connected.exists():
    manifest += json.loads(connected.read_text(encoding="utf-8"))
source_audit = root / "reports/original-source-audit-manifest.json"
if source_audit.exists():
    manifest += json.loads(source_audit.read_text(encoding="utf-8"))
stage_three = root / "reports/stage-three-evidence-manifest.json"
if stage_three.exists():
    manifest += json.loads(stage_three.read_text(encoding="utf-8"))
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
if source_audit.exists():
    audit = json.loads((root / "reports/original-source-audit-evidence.json").read_text(encoding="utf-8"))
    originals = json.loads((root / "research/source_manifest.json").read_text(encoding="utf-8"))
    if [{key: row[key] for key in ("filename", "bytes", "sha256")}
            for row in audit["original_sources"]] != originals:
        raise ValueError("Original-source audit used different reference files")
    if sorted(c["id"] for c in concepts) != audit["original_physics_concept_ids"]:
        raise ValueError("Source concept identities differ from the original-source audit")
    if audit["before"]["valid_behavior_reference"] != audit["after"]["valid_behavior_reference"]:
        raise ValueError("Valid-input baseline comparison differs")
    if connected.exists() and audit["before"]["source_sha256"] != data["manifest"]["environment"]["source_sha256"]:
        raise ValueError("Original-source audit baseline differs from the published study")
    credit = audit["after"]["program_credit_contract"]
    budget = audit["after"]["proposal_budget_contract"]
    if credit["sealed_program_credit_accepted"] or credit["parameters_changed"]:
        raise ValueError("Program-credit admission repair did not hold")
    if not budget["overlength_fixed_search_rejected"] or budget["work"]["operation_sum"] != 0:
        raise ValueError("Search-length guard did not hold")
    if budget["valid_length_five_budget_one_work"]["operations"]["program_candidates_constructed"] != 1:
        raise ValueError("Fixed candidate construction exceeded its execution budget")
if stage_three.exists():
    data = json.loads((root / "reports/stage-three-data.json").read_text(encoding="utf-8"))
    verified = json.loads((root / "reports/stage-three-verification.json").read_text(encoding="utf-8"))
    from sera.training import source_hash
    if source_hash() != data["manifest"]["environment"]["source_sha256"] or source_hash() != verified["source_sha256"]:
        raise ValueError("Stage-three source, study and verification identities differ")
    if not verified["passed"] or [row["seed"] for row in data["runs"]] != data["manifest"]["seeds"]:
        raise ValueError("Stage-three verification or seed coverage failed")
    episodes = json.loads((root / "reports/stage-three-policy-episodes.json").read_text(encoding="utf-8"))
    ids = [row["episode_id"] for row in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Policy episodes repeated")
    by_id = {row["episode_id"]: row for row in episodes}
    for run in data["runs"]:
        model_ids = set()
        for generation in run["outer"]["generations"]:
            if generation["before"] == generation["after"] or generation["after"] in model_ids:
                raise ValueError("An outer generation did not change the policy")
            model_ids.add(generation["after"])
            for episode in generation["training_episode_ids"]:
                row = by_id[episode]
                if row["split"] != "meta-train" or row["query_dataset_id"] == row["support_dataset_id"]:
                    raise ValueError("Outer learning used sealed or overlapping evidence")
        for model in run["predictors"]["models"].values():
            if model["training"]["steps"] < 1 or model["training"]["training_seconds"] < data["manifest"]["config"]["predictor_seconds"]:
                raise ValueError("A predictor did not receive its declared training budget")
    source = json.loads((root / "reports/stage-three-source-reproduction.json").read_text(encoding="utf-8"))
    if len(source["checkpoint_checks"]) != 36 or len(source["retraining"]) != 36 or not all(row["passed"] for row in source["mathematics"] + source["checkpoint_checks"]):
        raise ValueError("Original source reproduction is incomplete")
print(f"Verified {len(manifest)} evidence artifacts and 154 unique source concepts.")
