"""Freeze the current code/metadata after the completed scientific cycles."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/16_v3"


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


code = [p for p in (ROOT/"src/sera").glob("*.py")]
for folder in ("world_calibration", "guarded_consolidation", "generative_memory"):
    code += list((ROOT/"experiments"/folder).glob("*.py"))
code += [ROOT/"scripts"/n for n in ("run_v3_bounded.py", "windows_job_v3.py", "resume_v3_library.py", "freeze_world_cycle.py", "verify_v3_release.py")]
code += [ROOT/"tests"/n for n in ("test_world_calibration.py", "test_guarded_consolidation.py", "test_v3_resources.py")]
code += [ROOT/"pyproject.toml"]
sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)}
write(RELEASE/"current-source-manifest.json", sources)
with zipfile.ZipFile(RELEASE/"current-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sources:
        archive.write(ROOT/name, name)
budget = json.loads((ROOT/"runs/v3-batch-001/budget.json").read_text())
write(RELEASE/"state.json", {
    "schema": "sera.v3.continuation-state.1", "date": "2026-09-15", "status": "three_candidate_cycles_and_affected_repairs_complete",
    "parent_head": "9104839400d5053152ca423be6da1efe355aa71a", "work_branch": "research/v3-calibration-consolidation",
    "completed": ["local_reconciliation", "one_time_packet_verification_and_portability_diagnosis", "GG-GUARD-002", "GC-001", "GC-002", "affected_contract_repairs_and_explicit_migration", "199_regressions"],
    "report": "README.md", "current_guard_weights": "../15_applicability/guard-model.json",
    "certificates": "GG-GUARD-002/certificates.json", "current_finite_library": "GC-002/repair/corrected-library.json",
    "exact_local_parent": "../../runs/SHARED-GG-001/circle/corrected/versions/v0.pt",
    "parent_checkpoint_sha256": "eaf0b8fc315c943f4408f921f16ecfe8582bf2f675977c166f53e8242cbc4f37",
    "parent_solver_sha256": "f5f68a475c18e59fa85ee96aa3669fcfe78ac29b609f52e238a2f6413a5a85c1",
    "approved_resources": {"worker_seconds_remaining_before_release_checks": budget["remaining_seconds"],
                           "authority": "Live local runs/v3-batch-001/budget.json supersedes this snapshot", "memory_limit_bytes": budget["memory_limit_bytes"],
                           "active_experiments": [], "no_automatic_background_work": True},
    "do_not_restart": ["GG-GUARD-001", "GG-GUARD-002", "GC-001", "GC-002", "prior geometry/cloud/Kavi/bundle experiments"],
    "next_uncompleted": "A06 shared positive transfer, then A08 hidden-state interaction and A09 independently improved investigator",
    "scope_limits": ["conditional IID risk only", "supplied finite algebra and role bindings", "no positive neural transfer", "no learned eta improvement", "no M2-M4 claim"],
    "resume_command": ".venv/Scripts/python.exe -X utf8 -m scripts.resume_v3_library --a 4 --b 5 --c 6",
    "release_receipt": "release-verification/result.json", "remote_publication": "not performed by this continuation"})
evidence_path = ROOT/"research-continuation/EVIDENCE_MAP.json"
mapping = json.loads(evidence_path.read_text())
mapping["entries"].append({"claim": "V3 independent-world guard calibration and finite guarded trace consolidation",
                           "class": "newly_executed_audited_and_repaired", "evidence": "16_v3/README.md",
                           "limit": "Conditional components; failed first compiler and strict portability errors preserved; no positive neural transfer or eta improvement"})
evidence_path.write_text(json.dumps(mapping, indent=2)+"\n", encoding="utf-8", newline="\n")
names = list(RELEASE.rglob("*")) + code
names += [ROOT/"scripts"/n for n in ("reconcile_v3_replay.py", "record_v3_release.py", "finalize_v3_release.py")]
entries = []
for path in sorted(set(p for p in names if p.is_file())):
    if "release-verification" in path.parts or path.name == "release-manifest.json":
        continue
    raw = path.read_bytes()
    entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
write(RELEASE/"release-manifest.json", {"schema": "sera.v3.release-manifest.1", "files": entries,
    "scope": "Scientific evidence, reports, complete source snapshots and current code. Final verification receipts are separate to avoid recursive hashes."})
print(f"Frozen {len(entries)} evidence/source files")
