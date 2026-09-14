"""Check published evidence hashes and source concept coverage after a checkout."""

import hashlib
import json
import subprocess
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
evaluation_v2 = root / "reports/evaluation-v2-evidence-manifest.json"
historical_registries = {}
if evaluation_v2.exists():
    entries = json.loads(evaluation_v2.read_text(encoding="utf-8"))
    manifest += entries
    historical_registries = {(entry["file"], entry["sha256"]): "f9529cd8cb1351df121936917f0e9ee13bfcd8ea"
                             for entry in entries if entry["file"] in {"../research/variants.json", "../research/variants.md"}}
shared = root / "reports/shared-learner-evidence-manifest.json"
if shared.exists():
    manifest += json.loads(shared.read_text(encoding="utf-8"))
for entry in manifest:
    path = root / "reports" / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        historical = historical_registries.get((entry["file"], entry["sha256"]))
        if historical is None:
            raise ValueError(f"Evidence changed: {entry['file']}")
        name = path.resolve().relative_to(root).as_posix()
        preserved = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "show", f"{historical}:{name}"], cwd=root)
        if hashlib.sha256(preserved).hexdigest() != entry["sha256"]:
            raise ValueError(f"Historical registry changed: {entry['file']}")
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
    release = json.loads((root / "research/release-sources.json").read_text(encoding="utf-8"))["0.3.0"]
    def historical_git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root)
    paths = historical_git("ls-tree", "-r", "--name-only", release["git_commit"], "src/sera").decode().splitlines()
    historical = hashlib.sha256()
    for filename in sorted(p for p in paths if p.endswith(".py") and p.count("/") == 2):
        historical.update(Path(filename).name.encode())
        historical.update(historical_git("show", f"{release['git_commit']}:{filename}").replace(b"\r\n", b"\n"))
    if historical.hexdigest() != release["source_sha256"]:
        raise ValueError("Historical executable source differs from its pinned identity")
    if release["source_sha256"] != data["manifest"]["environment"]["source_sha256"] or release["source_sha256"] != verified["source_sha256"]:
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
if evaluation_v2.exists():
    data = json.loads((root / "reports/evaluation-v2-data.json").read_text(encoding="utf-8"))
    verified = json.loads((root / "reports/evaluation-v2-verification.json").read_text(encoding="utf-8"))
    release = json.loads((root / "research/release-sources.json").read_text(encoding="utf-8"))["0.4.0"]
    paths = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "ls-tree", "-r", "--name-only",
                                     release["git_commit"], "src/sera"], cwd=root).decode().splitlines()
    historical = hashlib.sha256()
    for filename in sorted(p for p in paths if p.endswith(".py") and p.count("/") == 2):
        historical.update(Path(filename).name.encode())
        historical.update(subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "show",
                          f"{release['git_commit']}:{filename}"], cwd=root).replace(b"\r\n", b"\n"))
    if (historical.hexdigest() != release["source_sha256"] or release["source_sha256"] != data["environment"]["source_sha256"]
            or release["source_sha256"] != verified["source_sha256"]):
        raise ValueError("Historical evaluation-v2 and executable source identities differ")
    if data["status"] != "completed" or not verified["passed"]:
        raise ValueError("Current evaluation was not completed and verified")
    if [r["seed"] for r in data["typed"]] != data["configuration"]["seeds"]:
        raise ValueError("Current evaluation seed coverage differs")
    if not all(r["manifest"]["cross_partition_overlap"] == 0 and r["manifest"]["within_partition_duplicates"] == 0
               for r in data["typed"]):
        raise ValueError("Current semantic partitions contain overlap")
    if len(verified["typed_checkpoint_checks"]) != 36 or max(r["maximum_score_difference"] for r in verified["typed_checkpoint_checks"]) > 1e-7:
        raise ValueError("Typed checkpoint replay is incomplete")
    if verified["decision_arithmetic_checks"] != 15 or not all(r["old_contract_reproduced"] for r in data["retention"]):
        raise ValueError("Historical proposal comparison is incomplete")
    if not data["preservation"]["passed"] or not data["live_continuation"]["parent_unchanged"]:
        raise ValueError("Preservation check failed")
    if data["live_continuation"]["result"]["decision"]["capability_contract"] != "separate-retention-v2":
        raise ValueError("Ordinary continuation did not use the current retention contract")
if shared.exists():
    from sera.training import source_hash
    data = json.loads((root / "reports/shared-learner-data.json").read_text(encoding="utf-8"))
    verified = json.loads((root / "reports/shared-learner-verification.json").read_text(encoding="utf-8"))
    if data["source_sha256"] != source_hash() or verified["source_sha256"] != source_hash():
        raise ValueError("Shared study source identity differs from the executable package")
    if data["status"] != "completed" or not verified["passed"]:
        raise ValueError("Shared study is incomplete")
    if {(r["seed"], r["kind"]) for r in data["trials"]} != {(s, k) for s in range(3) for k in ("delta", "reference")}:
        raise ValueError("Shared study seed/core coverage differs")
    if verified["checkpoint_replays"] != 114 or verified["decision_arithmetic_checks"] != 108:
        raise ValueError("Shared candidate replay or decision coverage is incomplete")
    for audit in verified["audits"]:
        for trial in audit["trials"]:
            if trial["partitions"]["cross_partition_overlap"] or not all(r["passed"] and r["maximum_score_difference"] <= 1e-7 for r in trial["replay_checks"]):
                raise ValueError("Shared evidence overlap or changed replay outputs")
    if not verified["live"]["decision_reproduced"] or not verified["live"]["parameter_owner_shared_after_reload"]:
        raise ValueError("Persistent shared learning did not reproduce")
    if not data["preservation"]["passed"]:
        raise ValueError("Earlier work was not preserved")
print(f"Verified {len(manifest)} evidence artifacts and 154 unique source concepts.")
