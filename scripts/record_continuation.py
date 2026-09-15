"""Refresh the persistent continuation ledger from completed local evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research-continuation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def append_unique(filename, rows):
    path = WORK / filename
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    ids = {row["id"] for row in existing}
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for row in rows:
            if row["id"] not in ids:
                output.write(json.dumps(row, sort_keys=True) + "\n")
                ids.add(row["id"])


def main():
    state = read(WORK / "RESEARCH_STATE.json")
    stages = {stage["id"]: stage for stage in state["stages"]}
    state["date"] = "2026-09-15"
    stages["02-source-audit"].update(status="complete_with_declared_scope",
        detail="Pinned SERA 71 tests and reduced shared study pass; pinned Kavi 321 tests pass. Full historical SERA training remains separate evidence.",
        evidence="01_audit/pinned-sera-execution.md")
    stages["03-reproduction"].update(status="completed_with_disclosed_strict_portability_failures",
        detail="First bundle57checkpoints/603conditions: exact correctness,2NLL failures. Second bundle55tests,1050eval/240teaching/50snapshot portable replay; fresh7job reproduction passes. Archived/fresh strict P2 hashes and101numeric leaves across2records differ.",
        evidence=["01_audit/bundle1-evidence.json", "01_audit/bundle2-evidence.json"])
    stages["04-generative-geometry"].update(status="GG-P0_subset_complete_and_audited",
        detail="640paired cases,5120models,20480query partitions;8controls/4of10task families. Exact replay and full refit pass. Six task families and unordered geometry remain open.",
        evidence="08_analysis/GG-P0-001/audit-and-interpretation.md")
    stages["05-competing-hypotheses"].update(status="bounded_shared_owner_integration_complete",
        detail="Four supplied generator explanations in registered R1 buffers; four trained-parent fixtures, corrections, predecessors and reload pass. No positive neural transfer.",
        evidence="05_experiments/shared-generative-integration.md")
    stages["06-active-investigation"].update(status="fixed_policy_component_complete_full_integration_open",
        detail="300 policy runs, 3000 observations and 12000 independently audited class posteriors. No broad advantage over space filling; no new eta trained.",
        evidence="08_analysis/GG-ACT-001/report.md")
    stages["07-imagination"].update(status="conditional_completion_verified_causal_decision_value_open",
        detail="Same-owner hypothetical predictions are bitwise equal and evidence-isolated. Causal rollout and matched-cost decision benefit remain untested.")
    stages["09-learned-applicability"].update(status="dependency_contracts_complete_learned_guard_pending",
        detail="Transitive stale-definition invalidation passes; observed overconfidence blocks general applicability claims.")
    stages["10-continual-revision"].update(status="same_schema_correction_verified_task_stream_and_latent_migration_open",
        detail="Actual trained-parent correction and 40-group conditional retention verified; no long unfamiliar-task stream or learned latent mapping.")
    state["next_action"] = "Implement and freeze GG-GUARD-001 learned applicability with strong fixed controls and protected future families; then consolidate validated generators and integrate inquiry through the shared owner."
    state["completed_local_evidence"] = {
        "geometry_replay": "../runs/generative-memory-GG-P0-001-verification/verification.json",
        "geometry_refit": "../runs/generative-memory-GG-P0-001-refit-resource-v2/verification.json",
        "geometry_source": "b83aab04331d39eadb59db4d6bff183ad69a84324adecadc09d4eee444dcc590",
        "geometry_protocol": "dbbd8ef1d7d1bec79dd8727d89331377c08042779e08ef9d595b77e959006ffd",
        "resource_supervisor": "../runs/GG-P0-resource-supervisor-v2/state.json",
        "peak_memory_boundary": "Corrected refit job peak committed memory308879360bytes; original peakRSS unavailable."
    }
    write(WORK / "RESEARCH_STATE.json", state)
    append_unique("EVIDENCE_LEDGER.jsonl", [
        {"id": "pinned-SERA-executed", "class": "newly_executed", "claim": "71 pinned tests and reduced shared-study plumbing pass", "evidence": "01_audit/pinned-sera-execution.md"},
        {"id": "GG-ACT-001-completed", "class": "newly_executed_and_replayed", "claim": "300 fixed-policy runs and 3000 observations replay exactly; independent batch algebra checks 12000 class posteriors", "evidence": "08_analysis/GG-ACT-001/report.md", "boundary": "Supplied phase, grammar and noise; no learned investigator or applicability."},
        {"id": "SHARED-GG-001-completed", "class": "newly_executed_integration", "claim": "Actual trained parent acquires and corrects four generator fixtures; 944 retained score elements across 40 groups match exactly", "evidence": "05_experiments/shared-generative-integration.md", "boundary": "Registered analytic buffers; old routes unchanged; no positive neural transfer."}
    ])
    append_unique("EXPERIMENT_REGISTRY.jsonl", [
        {"id": "GG-ACT-001-plumbing", "status": "completed_plumbing_only", "raw": "../runs/GG-ACT-001-plumbing"},
        {"id": "GG-ACT-001-final", "status": "completed_exploratory_not_promoted", "policy_runs": 300, "observations": 3000, "raw": "../runs/GG-ACT-001-final", "report": "08_analysis/GG-ACT-001/report.md"},
        {"id": "SHARED-GG-001", "status": "completed_integration_control", "fixtures": 4, "observations": 64, "corrections": 4, "raw": "../runs/SHARED-GG-001", "report": "05_experiments/shared-generative-integration.md"}
    ])
    append_unique("FAILURE_LEDGER.jsonl", [
        {"id": "GG-ACT-001-omitted-class", "status": "open_research_failure", "effect": "Information gain: 99.738% mean best-class confidence, 51.3% nominal95 interval coverage; alarm misses 3 of 12 omitted-pattern runs.", "evidence": "08_analysis/GG-ACT-001/report.md"},
        {"id": "shared-generator-precision-and-reconciliation", "status": "fixed_with_unchanged_tolerances", "effect": "Removed float32 hypothetical/likelihood intermediates; restore rebuilds recorded statistics; all 14 new regressions pass.", "evidence": "09_failures/shared-generative/"}
    ])
    append_unique("ARCHITECTURE_REGISTRY.jsonl", [
        {"id": "shared-generator-fixture-v1", "status": "implemented_and_verified", "numeric_buffer_bytes": 2880, "new_trainable_parameters": 0, "boundary": "Four alternatives for one trajectory, registered in R1; conditional retention, not positive transfer."}
    ])
    append_unique("EVIDENCE_LEDGER.jsonl", [
        {"id": "B1-exhaustive-replay", "class": "newly_replayed", "status": "strict_numeric_partial_failure", "claim": "All411648 correctness elements match;601of603NLL checks pass1e-6", "evidence": "01_audit/bundle1-evidence.json"},
        {"id": "B2-reproduction", "class": "newly_reproduced", "claim": "Seven-job fresh reproduction/self-replay completed with full expected record coverage", "evidence": "01_audit/bundle2-evidence.json", "boundary": "Cross-platform archive parity has disclosed exact-hash and numeric failures; discrete decisions agree."},
        {"id": "GG-P0-001", "class": "newly_executed_and_reproduced", "claim": "5120models,20480partitions replay exactly; all5120refitted artifact hashes match", "evidence": "08_analysis/GG-P0-001/audit-and-interpretation.md", "boundary": "Supplied-phase four-family component comparison, not shared neural transfer or geometry ontology discovery."},
        {"id": "GG-P0-independent-uncertainty", "class": "independently_checked_algebra", "claim": "4480linear components,640class-weight vectors and20480marginal NLL partitions verified", "evidence": "01_audit/generative-memory-code-review.md"},
        {"id": "W01-W02-contracts", "class": "newly_executed", "claim": "19event/ownership/invalidation/hypothetical/replay lifecycle cases pass", "evidence": "01_audit/world-graph-contracts.md", "boundary": "Identity-schema event replay; latent migration and learned semantics remain open."}
    ])
    append_unique("EXPERIMENT_REGISTRY.jsonl", [
        {"id": "B2-fresh-reproduction", "status": "completed", "raw": "12_reproductions/bundle2/reproduction", "independent_seeds_added": 0, "purpose": "Same frozen-seed reproduction of supplied packet"},
        {"id": "GG-P0-smoke-excluded", "status": "completed_plumbing_only", "models": 32, "raw": "../runs/generative-memory-smoke"},
        {"id": "GG-P0-001", "status": "completed_exploratory", "cases": 640, "models": 5120, "raw": "../runs/generative-memory-GG-P0-001", "source_sha256": state["completed_local_evidence"]["geometry_source"], "report": "08_analysis/GG-P0-001/report.md"},
        {"id": "GG-P0-001-refit", "status": "completed_exact_hash_parity", "models": 5120, "independent_seeds_added": 0, "raw": "../runs/generative-memory-GG-P0-001-refit"},
        {"id": "GG-P0-001-resource-refit-v2", "status": "completed_exact_hash_parity", "models": 5120, "independent_seeds_added": 0, "raw": "../runs/generative-memory-GG-P0-001-refit-resource-v2", "supervisor": "../runs/GG-P0-resource-supervisor-v2"}
    ])
    append_unique("FAILURE_LEDGER.jsonl", [
        {"id": "B1-NLL-portability", "status": "preserved", "evidence": "01_audit/bundle1-evidence.json", "effect": "Two original1e-6NLL checks fail; accuracy vectors unchanged."},
        {"id": "B2-strict-parity", "status": "preserved_and_diagnosed", "evidence": "01_audit/bundle2-evidence.json", "effect": "Archived repair exact hashes and101numeric leaves across2records fail; fresh self-replay succeeds."},
        {"id": "GG-P0-exception-confidence", "status": "open_research_failure", "evidence": "08_analysis/GG-P0-001/summary.json", "effect": "Local residual improvement leaves unseen-arc error unchanged and coverage drops34.1%to23.2% in n64/noise.02 stratum."},
        {"id": "GG-P0-codec-seed-budget-claims", "status": "errata_recorded_future_source_corrected", "evidence": "01_audit/generative-memory-code-review.md", "effect": "Mixed precision, affine initializer coupling, soft cap and missing original memory are explicit; original artifacts preserved."},
        {"id": "GG-P0-dense-reference-conditioning", "status": "diagnosed_reference_limitation", "evidence": "01_audit/generative-memory-svd-evidence.json", "effect": "197dense-reference evidence differences; stableSVD passes all4480 at unchanged1e-7 tolerance. Failed diagnostic retained."},
        {"id": "Windows-launcher-telemetry", "status": "repaired_and_verified", "evidence": "09_failures/launcher-telemetry-v1/README.md", "effect": "First supervisor observed launcher only; new job-object supervision includes descendants. Three-second forced timeout left no recorded live process."}
    ])
    append_unique("ARCHITECTURE_REGISTRY.jsonl", [
        {"id": "GG-P0-reference-generators", "status": "executed_component_not_promoted", "boundary": "Supplied phase/grammar and learned coefficients/class choice/residuals; local improvement does not imply applicability."},
        {"id": "W01-W02-common-contracts", "status": "implemented_and_focused_tests_pass", "boundary": "Parameter-free SharedR1 references and immutable graph/event ownership; event replay is not learned latent migration."}
    ])
    literature = WORK / "02_literature/LITERATURE_LEDGER.jsonl"
    if literature.exists():
        checksum = hashlib.sha256(literature.read_bytes()).hexdigest()
        indexed = []
        for index, line in enumerate(literature.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                entry = json.loads(line)
                indexed.append({"id": entry.get("id", f"L{index:02d}"), "class": "literature_supported_not_reproduced",
                                "ledger": "02_literature/LITERATURE_LEDGER.jsonl", "line": index,
                                "ledger_sha256": checksum})
        (WORK / "LITERATURE_LEDGER.jsonl").write_text("".join(json.dumps(row)+"\n" for row in indexed), encoding="utf-8", newline="\n")
    evidence_map = read(WORK / "EVIDENCE_MAP.json")
    for entry in evidence_map["entries"]:
        if entry["claim"] == "Generative geometry and integrated investigation":
            entry.update({"class": "components_and_bounded_owner_integration_executed", "evidence": "RESULTS_2026-09-15.md",
                          "limit": "GG-P0 eight controls/four families; GG-ACT fixed policies; trained-parent generator integration verified. Full shared inquiry, learned applicability and M2-M4 remain open."})
    write(WORK / "EVIDENCE_MAP.json", evidence_map)
    print(json.dumps({"status": "updated", "geometry_models": 5120, "stages": len(state["stages"])}))


if __name__ == "__main__":
    main()
