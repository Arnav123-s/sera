"""Seal the completed local cycle and its complete measured cost; no model execution."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/25_constraint_inquiry"
NAV = ["README.md", "research-continuation/RESEARCH_STATE.json", "research-continuation/NEXT_ACTIONS.md",
       "research-continuation/NEXT_EXPERIMENT_PROTOCOL.md", "research-continuation/DECISION_LOG.md"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def append(name, record):
    with (ROOT / "research-continuation" / name).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    if (OUT / "release-manifest.json").exists():
        raise FileExistsError("Preserve the sealed release")
    budget_path = ROOT / "runs/v3-batch-001/budget.json"
    if (budget_path.parent / "active.lock").exists():
        raise ValueError("An owned numerical job is still active")
    baseline, budget = read(OUT / "reconciliation.json"), read(budget_path)
    jobs = sorted(set(budget["jobs"]) - set(baseline["budget"]["jobs"]))
    receipts = []
    for job in jobs:
        path = ROOT / job / "state.json"
        state = read(path)
        assert state["status"] != "RUNNING" and state["peak_job_committed_bytes"] <= 2147483648
        assert row(ROOT / job / "process.log")["sha256"] == state["log_sha256"]
        receipts.append({"job": job, "status": state["status"], "charged_seconds": state["charged_seconds"],
                         "peak_job_committed_bytes": state["peak_job_committed_bytes"], "state_sha256": row(path)["sha256"]})
    charged = sum(r["charged_seconds"] for r in receipts)
    assert abs(baseline["budget"]["remaining_seconds"] - budget["remaining_seconds"] - charged) < 1e-7
    tests = ET.parse(OUT / "full-regression.xml")
    assert len(list(tests.iter("testcase"))) == 302 and not list(tests.iter("failure")) and not list(tests.iter("error"))
    assert read(OUT / "independent-audit.json")["records"] == 7680
    for name, prior in baseline["pointers"].items():
        assert row(ROOT / "runs" / name / "current.json")["sha256"] == prior["sha256"]
    source_rows = read(ROOT / "research-continuation/24_language_inquiry/source-manifest.json")["sources"]
    unchanged = []
    for item in source_rows:
        if item["path"].endswith(".py"):
            assert row(ROOT / item["path"])["sha256"] == item["sha256"]
            unchanged.append(item["path"])
    costs = {"cycle_charged_seconds": charged, "starting_remaining_seconds": baseline["budget"]["remaining_seconds"],
             "remaining_seconds_at_seal": budget["remaining_seconds"], "owned_jobs": receipts,
             "peak_job_committed_bytes": max(r["peak_job_committed_bytes"] for r in receipts),
             "threads_per_worker": 1, "memory_limit_bytes": 2147483648,
             "unmeasured": ["Research and engineering wall time", "Static hashing/extraction/archival work",
                            "Complete FLOPs", "Energy", "Peak RSS"],
             "counter_scope": "Runtime conditional_transition_points counts gradient transitions only; experimental records include their documented gate work. Supervisor wall charges include all child computation.",
             "duplicate_prefix": "Two extra-exposure fits repeat the primary 200-step prefix exactly. These costs remain charged and are not extra independent evidence."}
    write(OUT / "costs.json", costs)
    write(OUT / "resource-budget-at-seal.json", budget)
    with zipfile.ZipFile(OUT / "supervision.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for receipt in receipts:
            directory = ROOT / receipt["job"]
            for name in ("state.json", "process.log"):
                archive.write(directory / name, (directory / name).relative_to(ROOT).as_posix())
            if directory.name == "worker":
                for name in ("request.json", "response.json"):
                    if (directory.parent / name).exists():
                        archive.write(directory.parent / name, (directory.parent / name).relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUT / "navigation-before.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in NAV:
            archive.write(ROOT / name, name)
    state = read(ROOT / NAV[1])
    state["active_direction"] = "Learned role-to-constraint binding and continuous conditional imagination through the actual owner; source-qualified missing-location acquisition, exact persistence and correction. New wording, omitted mechanisms, positive shared adaptation and learned eta remain open."
    state["latest_continuation"] = "25_constraint_inquiry/report.md"
    state["latest_user_steering"]["v10"] = "Reconcile all v5-v10 addenda, preserve newer work and applicability, integrate actual shared-owner learned constraints, grounded acquisition and efficient settling."
    state["next_action"] = "Continue runs/sera-constraints unfinished via scripts/sera_constraints.py solve --id unfinished; next research must freeze fresh semantic productions and world-disjoint applicability evidence. Read the live resource balance first."
    state["completed_local_evidence"]["constraint_inquiry"] = {
        "report": "25_constraint_inquiry/report.md", "manifest": "25_constraint_inquiry/release-manifest.json",
        "full_suite_tests": 302, "new_contract_cases": 19, "language_records": 384, "numeric_records": 7680,
        "workspace": "../runs/sera-constraints", "owner": read(OUT / "integration-transcript.json")[-1]["response"]["owner"],
        "remaining_seconds_at_release": budget["remaining_seconds"], "learned_eta": False,
        "physical_applicability_calibrated": False, "unseen_surface_exact": .5,
    }
    (ROOT / NAV[1]).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8", newline="\n")
    notice = ("Latest: [v10 actual-owner constraint integration](25_constraint_inquiry/report.md). "
              "302 regressions and two finite cycles completed; learned directed pairs work, new wording and omitted laws still fail. "
              "[Use the saved learner](25_constraint_inquiry/README.md); all older checklists below remain historical.\n\n")
    for name in NAV[2:4]:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        first, rest = text.split("\n", 1)
        path.write_text(first + "\n\n" + notice + rest.lstrip("\n"), encoding="utf-8", newline="\n")
    with (ROOT / NAV[4]).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n\n## v10 actual-owner constraint inquiry\n\nI integrated finite learned role binding, an acquired-model initializer, continuous conditional search, grounded simulator acquisition and persistent correction. CI-001 retains its 50% new-wording failure and omitted-law errors; CI-002 adds a fixed residual stop on fresh same-family cases. The analytic baseline remains cheaper. This is an optional conditional route, not a replacement applicability certificate or learned investigator. See release 25 for the full records.\n")
    with (ROOT / "README.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n\nThe [v10 constraint-inquiry continuation](research-continuation/25_constraint_inquiry/README.md) connects learned actor/target/modality binding, acquired dynamics, learned initialization and conditional settling through the actual owner. It can acquire a missing simulated landmark, preserve unfinished work, restore it exactly and invalidate plans after new evidence. The release passes 302 regressions and preserves its failed new-wording and omitted-law tests. [Results](research-continuation/25_constraint_inquiry/report.md), [architecture audit](research-continuation/25_constraint_inquiry/architecture-audit.md) and [costs](research-continuation/25_constraint_inquiry/costs.json) retain the limits. Run `.venv/Scripts/python.exe scripts/sera_constraints.py result --id reach` in the prepared local workspace.\n")
    append("EXPERIMENT_REGISTRY.jsonl", {"id": "CI-001", "status": "restricted_local_integration", "protocol": "25_constraint_inquiry/protocol.md", "report": "25_constraint_inquiry/report.md"})
    append("EXPERIMENT_REGISTRY.jsonl", {"id": "CI-002", "status": "exploratory_fixed_stopping", "protocol": "25_constraint_inquiry/followup-protocol.md", "report": "25_constraint_inquiry/report.md"})
    append("EVIDENCE_LEDGER.jsonl", {"id": "CI-OWNER-001", "class": "finite_shared_owner_integration", "claim": "Actual shared-owner role binding, continuous imagination, source-qualified location acquisition and exact persistent correction; 302 tests, 7680 independent scalar outcome checks", "evidence": "25_constraint_inquiry/release-manifest.json", "boundary": "Supplied grammar/objective/update/scheduler; no new applicability, positive neural transfer or learned eta"})
    for identifier, detail in [
        ("CI-surface-failure", "Both ordered seeds score 50% on withheld surface composition; extra exposure does not repair it. Full family withheld from operational admission; original final records unchanged."),
        ("CI-omitted-law", "All 256 primary learned-refinement nominal witnesses reduce to 26 successful expanded-law outcomes; follow-up learned stop has 7/128. Applicability remains unqualified."),
        ("CI-fixed-refinement-cost", "Twelve unconditional gradient steps cost more wall time than the small grid; fresh exploratory stopping improves the local tradeoff, but analytic remains cheaper."),
        ("CI-repeated-prefix-cost", "Extra-exposure controls repeated byte-identical 200-step training prefixes. Costs and checkpoints preserved; future extra-exposure work should resume exact optimizer state."),
    ]:
        append("FAILURE_LEDGER.jsonl", {"id": identifier, "status": "preserved", "detail": detail, "evidence": "25_constraint_inquiry/report.md"})
    append("LITERATURE_LEDGER.jsonl", {"id": "CI-primary-review", "class": "targeted_primary_review", "evidence": "25_constraint_inquiry/literature.md", "scope": "EBM, EqProp, MPS/Born and mixed transport; methods are not all implemented"})
    write(OUT / "preservation.json", {"old_python_sources_unchanged": unchanged, "old_pointers_unchanged": baseline["pointers"],
                                      "prior_release": {"status": "PASS", "artifacts": 69, "archived_sources": 288, "local_artifacts": 71},
                                      "handbook_sha256": "2371a18a074680d147f1f43cb7ffca2c0970d3dc2a08bccc095c1f71a688b528"})
    with (OUT / "report.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\nAt sealing, this cycle charged **{charged:.3f} seconds** across **{len(receipts)} supervised jobs**, "
                     f"with peak process-tree committed memory **{costs['peak_job_committed_bytes']:,} bytes**. "
                     f"The existing allowance has **{budget['remaining_seconds']:.3f} seconds** left. "
                     "These are measured supervised costs, not total research effort or a new grant.\n")
    raw_paths = [OUT / n for n in ("evaluation/language-records.json", "evaluation/numerical-records.json", "evaluation/cases.json",
                                   "followup/records.json", "followup/cases.json")]
    raw_rows = [row(p) for p in raw_paths]
    with zipfile.ZipFile(OUT / "raw-records.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in raw_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    write(OUT / "raw-records-manifest.json", raw_rows)
    local_paths = []
    for folder in ("runs/CI-pilot-001", "runs/CI-fits-final-001", "runs/sera-constraints/revisions"):
        local_paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    local_rows = [row(p) for p in sorted(local_paths)]
    with zipfile.ZipFile(OUT / "checkpoints.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        stored = set()
        for path, item in zip(sorted(local_paths), local_rows):
            member = "objects/" + item["sha256"]
            item["archive_member"] = member
            if member not in stored:
                archive.write(path, member)
                stored.add(member)
    write(OUT / "local-state-manifest.json", {"immutable_artifacts": local_rows, "release_pointer": read(ROOT / "runs/sera-constraints/current.json")})
    source_paths = set()
    for folder in ("src", "tests", "scripts", "experiments", "workbench"):
        source_paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".html", ".js", ".css"))
    source_paths.update(ROOT / n for n in [*NAV, "pyproject.toml", ".gitignore", ".gitattributes"])
    with zipfile.ZipFile(OUT / "sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
    write(OUT / "source-manifest.json", {"sources": [row(p) for p in sorted(source_paths)]})
    write(OUT / "test-summary.json", {"full_suite_tests": 302, "new_tests": 19, "prior_targeted_tests": 16,
                                     "independent_numeric_records": 7680, "packet_tests": 56})
    excluded = set(raw_paths)
    artifacts = [row(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p not in excluded]
    write(OUT / "release-manifest.json", {"schema": "sera.constraint-inquiry.release.1", "time_utc": datetime.now(timezone.utc).isoformat(),
                                          "parent_head": baseline["sera_head"], "artifacts": artifacts})
    print(json.dumps({"artifacts": len(artifacts), "local_artifacts": len(local_rows), "sources": len(source_paths),
                      "charged_seconds": charged, "remaining_seconds": budget["remaining_seconds"]}))


if __name__ == "__main__":
    main()
