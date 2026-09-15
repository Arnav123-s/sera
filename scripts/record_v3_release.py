"""Copy newly completed evidence, update navigation, preserve historical bytes."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/16_v3"
BATCH = ROOT/"runs/v3-batch-001"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def preserve(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError(f"Refusing to replace existing evidence: {target}")
        return
    shutil.copyfile(source, target)


def append(path, value):
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(json.loads(line).get("id") == value["id"] for line in lines):
        raise ValueError("Evidence identifier already recorded")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, sort_keys=True)+"\n")


for experiment, run in (("GG-GUARD-002", "GG-GUARD-002-final-v2"), ("GC-001", "GC-001-final"), ("GC-002", "GC-002-final")):
    for source in (BATCH/run/"data").iterdir():
        preserve(source, RELEASE/experiment/source.name)
    audit = BATCH/(experiment+"-audit")/"independent.json"
    preserve(audit, RELEASE/experiment/"independent.json")
for source in (BATCH/"GC-002-migration/data").iterdir():
    preserve(source, RELEASE/"GC-002/repair"/source.name)
for directory in BATCH.iterdir():
    if directory.is_dir() and (directory/"state.json").is_file():
        for name in ("state.json", "process.log"):
            preserve(directory/name, RELEASE/"supervision"/directory.name/name)
preserve(BATCH/"full-regression/tests.xml", RELEASE/"tests.xml")
preserve(BATCH/"budget.json", RELEASE/"worker-budget.json")
preserve(ROOT.parent.parent/"sera-local-reconciliation-v3-20260915/v3-tests.log", RELEASE/"intake-tests.log")
collector_path = ROOT.parent.parent/"sera-local-reconciliation-v3-20260915/collector.json"
collector = json.loads(collector_path.read_text())
checked = []
for inventory in collector["additional_artifacts"]:
    for row in inventory["files"]:
        if row["status"] != "hashed":
            raise ValueError("Incomplete explicit predecessor inventory")
        path = Path(inventory["root"])/row["path"]
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Predecessor changed: {path}")
        checked.append({"path": path.relative_to(ROOT).as_posix(), "sha256": row["sha256"], "bytes": len(raw)})
src = {}
for p in (ROOT/"src/sera").glob("*.py"):
    name = p.relative_to(ROOT).as_posix()
    old = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"9104839400d5053152ca423be6da1efe355aa71a:{name}"], cwd=ROOT)
    if p.read_bytes() != old:
        raise ValueError("Shared owner source changed")
    src[name] = hashlib.sha256(old).hexdigest()
write(RELEASE/"preservation.json", {"status": "PASS", "source_anchor": "9104839400d5053152ca423be6da1efe355aa71a",
                                      "unchanged_shared_sources": src, "predecessor_files": checked,
                                      "scope": "All 45 shared-owner sources and all 108 explicitly collected GG-GUARD-001/SHARED-GG-001 files. The initial broader 3000-file repository scan was partial, not a full device inventory."})
write(RELEASE/"reconciliation.json", {"sera_head": collector["repos"][0]["head"]["stdout"].strip(),
    "kavi_head": collector["repos"][1]["head"]["stdout"].strip(),
    "initial_sera_branch": "main", "initial_worktrees_clean": True,
    "collector_sha256": hashlib.sha256(collector_path.read_bytes()).hexdigest(), "collector_wall_seconds": collector["wall_seconds"],
    "collector_scope": "SERA scan capped at 3000 files/128MiB; Kavi and two explicitly selected predecessor run directories fully inventoried under collector exclusions. Private raw collector remains local.",
    "jobs": "CIM process command-line lookup denied by OS; Get-Process fallback found no matching Python/pytest/git/SERA/Kavi processes. No pre-existing job stopped.",
    "pack": {"archive": "SERA_v3_Update.zip", "sha256": "81bee4c749f36fa9dcdd5af64bbdfca273804e6c740f308e04e972f7211b5d1e",
             "entries_verified_after_baseline_restoration": 133, "verified_bytes": 22604808,
             "baseline_sha256": "4e9b829f67aca7bbb543b0c03eff022b019b67698a0f27f899fbd7a8186e814f"},
    "tests": {"discovered": 54, "passed": 44, "skipped_optional_old_demo_patch": 10},
    "strict_replay": "Calibration canonical hash and alias exact parameter equality FAIL; explicit saved-data compatibility diagnosis passes without replacing these failures.",
    "reused_evidence": "Existing 174-test/CI, GG-GUARD-001 refit and independent audit, original source audits and old training results were not restarted."})
state_path = ROOT/"research-continuation/RESEARCH_STATE.json"
state = json.loads(state_path.read_text())
state["latest_continuation"] = "16_v3/README.md"
state["active_direction"] = "V2-B conditional executable knowledge: independent-world guard calibration and finite trace consolidation; common-owner positive neural transfer remains open."
for stage in state["stages"]:
    if stage["id"] == "08-guarded-consolidation":
        stage.update(status="finite_component_acquired_proved_corrected_and_migrated", evidence="16_v3/GC-002/report.md",
                     detail="GC-001 fails full-cost proxy; GC-002 reduces per-candidate proxy 15.1% vs local reasoning, with 18 observations and no new neural parameters. Failed proof-domain contract fixed; original and repaired graphs retained. No learned eta or neural transfer.")
    if stage["id"] == "09-learned-applicability":
        stage.update(status="prospective_conditional_calibration_passes_shared_integration_open", evidence="16_v3/GG-GUARD-002/report.md",
                     detail="Frozen GG-GUARD-001 weights, 1800 new calibration worlds and 2012 final worlds. Matched grouped policy accepts 664/1000 with zero errors; abrupt-jump coverage zero; global policy has higher utility. Only declared IID world/query risk is certified.")
state["next_action"] = "A06: connect qualified conditional policy/guarded programs to actual shared learning and compare no-program, detached, integrated, extra-capacity arms on a different route at matched full cost; then A08 hidden-state continuing interaction. Do not rerun completed v3 cohorts."
state["completed_local_evidence"]["v3"] = {"report": "16_v3/README.md", "current_guard_model": "15_applicability/guard-model.json",
    "group_certificate": "16_v3/GG-GUARD-002/certificates.json", "current_library": "16_v3/GC-002/repair/corrected-library.json",
    "library_parent": "../runs/SHARED-GG-001/circle/corrected", "scope": "Research components; operational solver/parameters unchanged"}
write(state_path, state)
for identifier, claim, evidence, scope in (
    ("GG-GUARD-002", "3812 independent worlds; matched selected 664/1000 with zero errors; frozen gate passes", "16_v3/GG-GUARD-002/report.md", "Conditional IID world/query calibration, not arbitrary mechanism validity; no shared route promotion"),
    ("GC-001", "Correct guarded finite programs; full-cost proxy loses to local reasoning", "16_v3/GC-001/summary.json", "Rejected candidate preserved, including proof and restore costs"),
    ("GC-002", "Fresh cohort correct; indexed finite proof reduces per-candidate full-cost proxy 15.1%; repaired graph migration checks 1024 fresh cases", "16_v3/GC-002/report.md", "Finite supplied semantics; full failed-trial and repair costs remain separate; no eta or neural-transfer claim"),
):
    append(ROOT/"research-continuation/EVIDENCE_LEDGER.jsonl", {"id": identifier, "claim": claim, "class": "newly_executed_and_audited", "evidence": evidence, "boundary": scope})
    append(ROOT/"research-continuation/EXPERIMENT_REGISTRY.jsonl", {"id": identifier, "status": "completed_preserved", "report": evidence, "raw": f"16_v3/{identifier}"})
for identifier, finding in (
    ("v3-strict-portability", "Original calibration hash and alias exact equality fail; bounded compatibility diagnosis preserved separately"),
    ("v3-resource-counter", "Oversized allocation rejected; failed test incorrectly assumed reported high-water count could not exceed cap; original test retained"),
    ("v3-preflight-launch", "Direct script import path failed before protocol creation; guarded model run exited before any world generation; module launch fixed"),
    ("GC-001-cost-failure", "732614 full-cost proxy vs 621064 local reasoning; retained and diagnosed"),
    ("GC-002-domain-contract", "Indexed proof was unsound if the forward relation depended on RHS c; explicit rejection added. Studied affine/offset relations do not depend on c"),
    ("GC-002-interpreter-contract", "Library previously trusted the declared interpreter; current code checks actual source/runtime and requires explicit reproof migration"),
):
    append(ROOT/"research-continuation/FAILURE_LEDGER.jsonl", {"id": identifier, "finding": finding, "evidence": "16_v3/README.md", "status": "preserved_and_diagnosed"})
print("Copied completed evidence; recorded predecessor preservation and current state.")
