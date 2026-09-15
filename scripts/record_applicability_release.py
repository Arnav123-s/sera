"""Record the completed applicability component without changing past evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/"research-continuation"
RELEASE = WORK/"15_applicability"


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")


def append(path, entry):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("id") == entry["id"] for r in rows):
        return
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True)+"\n")


def main():
    state_path = WORK/"RESEARCH_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stage = next(r for r in state["stages"] if r["id"] == "09-learned-applicability")
    stage.update(status="component_trained_and_audited_risk_gate_failed",
                 detail="GG-GUARD-001: 1,344 distinct worlds, 291 learned scalars, exact refit and independent audit. "
                        "Utility improves over fixed guards; local-exception and changing-mechanism risk remains unacceptable. "
                        "Candidate preserved outside the operational shared owner.",
                 evidence="15_applicability/report.md")
    state["next_action"] = ("Implement and freeze GG-GUARD-002: observed-feature calibration strata, a minimum validity probability, "
                           "and matched-cost additional evidence. Reserve new mechanisms; do not reuse GG-GUARD-001 final families as unseen. "
                            "Integrate only after prospective reliability and useful-acceptance gates, then test shared positive transfer.")
    state["completed_local_evidence"]["applicability"] = {
        "protocol": "15_applicability/protocol.json", "report": "15_applicability/report.md",
        "raw_complete": "15_applicability/records.json.gz", "model": "15_applicability/guard-model.json",
        "replay": "15_applicability/replay.json", "independent": "15_applicability/independent.json",
        "preservation": "15_applicability/preservation.json", "candidate_promoted": False}
    write(state_path, state)
    append(WORK/"EVIDENCE_LEDGER.jsonl", {
        "id": "GG-GUARD-001-completed", "class": "newly_trained_replayed_and_independently_audited",
        "claim": "Learned guard trained on 576 worlds plus disjoint calibration banks; 384 final worlds; "
                 "all 1,344 histories and exact model refit verified. Paired utility improves over fixed guards.",
        "boundary": "Conditional-risk gate fails. Supplied phase/grammar and fixed investigator; no owner promotion or M2.",
        "evidence": "15_applicability/report.md"})
    append(WORK/"EXPERIMENT_REGISTRY.jsonl", {
        "id": "GG-GUARD-001-final", "status": "completed_risk_gate_failed_candidate_preserved",
        "worlds": 1344, "final_worlds": 384, "support_observations": 13440, "query_outcomes": 44352,
        "raw": "15_applicability/records.json.gz", "report": "15_applicability/report.md"})
    append(WORK/"FAILURE_LEDGER.jsonl", {
        "id": "GG-GUARD-001-pooled-risk", "status": "open_research_failure",
        "effect": "Pooled calibration selects probability threshold 0.084088; local exceptions have 46.34% accepted risk, "
                  "piecewise drift 65.80%. Aggregate utility gains do not establish applicability.",
        "evidence": "15_applicability/summary.json"})
    append(WORK/"FAILURE_LEDGER.jsonl", {
        "id": "GG-GUARD-001-auditor-repairs", "status": "fixed_with_original_tolerance",
        "effect": "Corrected serialized provenance enum and independent coefficient ordering. "
                  "Both failed attempts, original/intermediate source and seven regression cases preserved. No training/result change.",
        "evidence": "15_applicability/audit-repair.json"})
    evidence_path = WORK/"EVIDENCE_MAP.json"
    evidence = json.loads(evidence_path.read_text())
    if not any(e.get("id") == "GG-GUARD-001" for e in evidence["entries"]):
        evidence["entries"].append({"id": "GG-GUARD-001", "claim": "Learned applicability component and its failure boundary",
                                    "class": "newly_executed_rejected_candidate",
                                    "evidence": "15_applicability/report.md",
                                    "limit": "Better selection utility; unreliable accepted predictions on omitted/nonstationary mechanisms. "
                                             "All old owner/model bytes unchanged; no positive transfer."})
    write(evidence_path, evidence)
    checklist_path = WORK/"NEXT_ACTIONS.md"
    checklist = checklist_path.read_text(encoding="utf-8")
    checklist = checklist.replace("- [ ] Train a learned applicability guard under GG-GUARD-001 and evaluate unseen mechanism families.",
                                  "- [x] Train GG-GUARD-001 with episode-disjoint teaching/calibration and two protected final mechanism families.\n"
                                  "- [x] Replay all 1,344 worlds, refit the model exactly, independently audit its mathematics and preserve failed checkers.\n"
                                  "- [x] Verify 174 tests and byte preservation of 16,556 older model/result files.\n"
                                  "- [ ] Pass prospective applicability and useful-acceptance gates under GG-GUARD-002; the first candidate failed conditional risk.")
    checklist = checklist.replace("Learned applicability must be tested before treating those predictions as supported knowledge.",
                                  "GG-GUARD-001 now tests learned applicability and finds that pooled calibration still permits unacceptable conditional risk. "
                                  "The candidate remains experimental; the next study must correct that boundary with new prospective evidence.")
    checklist_path.write_text(checklist, encoding="utf-8", newline="\n")
    decision = WORK/"DECISION_LOG.md"
    marker = "## GG-GUARD-001: retain the candidate; do not promote it"
    if marker not in decision.read_text(encoding="utf-8"):
        with decision.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n"+marker+"\n\nI trained a validity guard and verified exact replay, independent algebra and predecessor preservation. "
                         "The frozen conditional-risk gate failed despite improved utility. I retain its full weights, witnesses, raw cohort "
                         "and failed checker attempts in `15_applicability/`. No shared-owner source or checkpoint was changed. "
                         "The next protocol tests observed-feature calibration and additional paid evidence on new prospective mechanisms.\n")
    names = [p for p in RELEASE.rglob("*") if p.is_file() and p.name != "release-manifest.json"]
    names += [ROOT/name for name in (
        "README.md", ".github/workflows/ci.yml", "scripts/verify_continuation.py", "scripts/verify_applicability.py",
        "scripts/audit_guard_preservation.py", "scripts/freeze_guard_protocol.py", "scripts/build_applicability_report.py",
        "scripts/replay_guard_release.py", "scripts/record_applicability_release.py",
        "experiments/generative_memory/applicability.py", "experiments/generative_memory/applicability_study.py",
        "experiments/generative_memory/applicability_audit.py", "experiments/generative_memory/applicability_check.py",
        "tests/test_applicability.py", "tests/test_applicability_audit_contract.py",
        "research-continuation/RESEARCH_STATE.json", "research-continuation/NEXT_ACTIONS.md",
        "research-continuation/NEXT_EXPERIMENT_PROTOCOL.md", "research-continuation/EVIDENCE_LEDGER.jsonl",
        "research-continuation/EXPERIMENT_REGISTRY.jsonl", "research-continuation/FAILURE_LEDGER.jsonl",
        "research-continuation/DECISION_LOG.md", "research-continuation/EVIDENCE_MAP.json")]
    entries = []
    for path in sorted(set(names)):
        raw = path.read_bytes()
        entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    write(RELEASE/"release-manifest.json", {"schema": "sera.gg-guard.release.v1", "date": "2026-09-15",
                                           "parent_commit": "71512a612961f3bd6d3e58a779e65dc8d2dec352",
                                           "candidate_promoted": False, "files": entries})
    print(f"Recorded applicability milestone and {len(entries)} release files.")


if __name__ == "__main__":
    main()
