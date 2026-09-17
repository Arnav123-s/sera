"""Preserve every local C01/C02 attempt and publish an exact artifact inventory."""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/28_concept_refinement"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(name, files):
    destination = OUT / name
    if destination.exists():
        raise FileExistsError("Do not overwrite a sealed archive")
    hashes = {}
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(set(files)):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                member = path.relative_to(ROOT).as_posix()
                bundle.write(path, member)
                hashes[member] = sha(path)
    return hashes


def main():
    if (OUT / "release-manifest.json").exists() or (
        ROOT / "runs/v3-batch-001/active.lock"
    ).exists():
        raise ValueError("Seal once after all owned numerical jobs have settled")
    before = read(OUT / "budget-before.json")
    after = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {}
    for name in sorted(set(after["jobs"]) - set(before["jobs"])):
        state = read(ROOT / name / "state.json")
        if state["status"] in ("RESERVED", "RUNNING"):
            raise ValueError("Unsettled numerical reservation")
        jobs[name] = state
        folder = OUT / "jobs" / Path(name).name
        folder.mkdir(parents=True)
        for source in ("state.json", "process.log"):
            shutil.copyfile(ROOT / name / source, folder / source)
    charged = sum(j["charged_seconds"] for j in jobs.values())
    if abs(before["remaining_seconds"] - after["remaining_seconds"] - charged) > 1e-5:
        raise ValueError("Resource ledger does not reconcile")
    write(
        OUT / "costs.json",
        {
            "initial_seconds": before["remaining_seconds"],
            "charged_seconds": charged,
            "remaining_seconds": after["remaining_seconds"],
            "new_grant_seconds": 0,
            "jobs": jobs,
            "peak_committed_bytes": max(j["peak_job_committed_bytes"] for j in jobs.values()),
            "dispatch_note": "One audit launch was refused by the occupied lock before reservation; no job started or interrupted.",
            "scope": "One CPU thread, 2 GiB committed process-tree memory; failures, F01 rerun, replay and checks included.",
        },
    )
    shutil.copyfile(ROOT / "runs/v3-batch-001/budget.json", OUT / "budget-after.json")
    audit = read(OUT / "integration-audit.json")
    for name, expected in read(OUT / "reconciliation.json")["protected"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Protected Stage 27 artifact changed")
    for path in (ROOT / "runs/CR-study-r2").glob("*/fit.json"):
        write(OUT / "r2/fits" / (path.parent.name + ".json"), read(path))
    completed = {
        "stage": "C01/C02",
        "status": "integrated_and_evaluated",
        "date": datetime.now(timezone.utc).isoformat(),
        "parent_head": "9da1b21b06da4504d7d2b83777d69e874d4caeff",
        "actual_owner": audit["final_owner"],
        "live_store": "runs/sera-refinement-live",
        "live_pointer_sha256": sha(ROOT / "runs/sera-refinement-live/current.json"),
        "source": audit["source"],
        "contracts": read(OUT / "r2/freeze.json")["contracts"],
        "remaining_seconds_at_seal": after["remaining_seconds"],
        "new_allowance_granted": False,
        "owned_jobs": "all settled; no other jobs stopped",
        "sealed_final": "runs/CR-study-r2/final.json; do not use for tuning",
        "completed_training": [
            p.parent.name for p in sorted((ROOT / "runs/CR-study-r2").glob("*/fit.json"))
        ],
        "next_executable_action": ".venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-next-use-001 --module experiments.concept_refinement.runtime -- predict --subject body-1 --commands 0.8,0.8,0.8,0.8",
        "next_research_package": "C03: prospectively freeze new worlds to compare complexity/adequacy selection and output-error fitting; retain this runtime and both completed cohorts",
        "resume_training": "No incomplete fit; each corrected run retains step-0240.pt with optimizer/RNG state, and selected.pt. Ten exact resumptions passed.",
        "resume_runtime": "Restore the checksummed current.json revision through RefinementSession; observe later measurements, refine, then retry the unchanged task.",
    }
    write(OUT / "continuation.json", completed)
    state_path = ROOT / "research-continuation/RESEARCH_STATE.json"
    shutil.copyfile(state_path, OUT / "inherited-research-state.json")
    state = read(state_path)
    state.update(
        date=completed["date"],
        active_direction="Progressively refine grounded concepts through the actual learner, keeping valid simple operators and old abilities.",
        latest_continuation="28_concept_refinement/report.md",
        next_action=completed["next_research_package"],
    )
    state["stages"].append(
        {
            "id": "28-concept-refinement",
            "status": completed["status"],
            "evidence": "28_concept_refinement/report.md",
        }
    )
    state["completed_local_evidence"]["concept_refinement"] = {
        "report": "28_concept_refinement/report.md",
        "owner": audit["final_owner"],
        "workspace": "../runs/sera-refinement-live",
        "final_views": 216,
        "independent_worlds": 36,
        "optimizer_resumptions": 10,
        "retained_tensors": 162,
        "language_development_probes": 128,
        "remaining_seconds_at_release": after["remaining_seconds"],
    }
    state["latest_user_steering"]["v13"] = (
        "Read v12/v13; complete C01/C02 on actual StudyR1; test existing memory, preserve exact algebra and old language weights, distinguish empirical predictions and supplied concepts."
    )
    write(state_path, state)
    record = {
        "id": "CR-C01-C02",
        "status": "complete",
        "protocol": "28_concept_refinement/PROTOCOL.md",
        "report": "28_concept_refinement/report.md",
    }
    with (ROOT / "research-continuation/EXPERIMENT_REGISTRY.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(record) + "\n")
    with (ROOT / "research-continuation/FAILURE_LEDGER.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        for identifier, detail in (
            (
                "CR-PACKET-PORTABILITY",
                "One neural array and 72 physics leaves fail original exact/tolerance replay; independent checks and sizes retained without silently waiving thresholds.",
            ),
            (
                "CR-F01",
                "Simple teacher had a hidden initial transient; fixed before final access, complete first attempt preserved and charged.",
            ),
            (
                "CR-SELECTION-TRADEOFF",
                "Frozen simplicity selection passes admission but always-memory forecasts are more accurate on delayed cohorts; noisy/missing estimation and simple-model precision remain measured C03 targets.",
            ),
        ):
            stream.write(
                json.dumps(
                    {
                        "id": identifier,
                        "status": "preserved",
                        "detail": detail,
                        "evidence": "28_concept_refinement/report.md",
                    }
                )
                + "\n"
            )
    checklist = OUT / "checklist.md"
    checklist.write_text(
        checklist.read_text(encoding="utf-8").replace("- [ ]", "- [x]"), encoding="utf-8"
    )
    # Publication is completed after the seal and recorded separately to avoid self-reference.
    with checklist.open("a", encoding="utf-8") as stream:
        stream.write(
            "\nThe immutable local release is complete. Remote publication is confirmed separately in publication.json.\n"
        )
    archives = {
        "original-attempt.zip": archive(
            "original-attempt.zip", (ROOT / "runs/CR-study").rglob("*")
        ),
        "corrected-study.zip": archive(
            "corrected-study.zip", (ROOT / "runs/CR-study-r2").rglob("*")
        ),
        "runtime-state.zip": archive(
            "runtime-state.zip",
            list((ROOT / "runs/sera-refinement-live").rglob("*")) + [OUT / "parent-study.json"],
        ),
    }
    sources = list((ROOT / "experiments/concept_refinement").glob("*.py")) + [
        ROOT / name
        for name in (
            "scripts/run_concept_bounded.py",
            "scripts/run_v3_bounded.py",
            "scripts/windows_job_v3.py",
            "scripts/verify_continuation.py",
            "scripts/verify_applicability.py",
            "scripts/verify_v3_release.py",
            "scripts/verify_concept_release.py",
            "scripts/seal_concept_release.py",
            "tests/test_concept_refinement.py",
            "pyproject.toml",
        )
    ]
    archives["sources.zip"] = archive("sources.zip", sources)
    manifest = {
        "schema": "sera.concept.release.1",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "parent_head": completed["parent_head"],
        "archives": archives,
        "artifacts": [],
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in ("release-manifest.json", "publication.json"):
            manifest["artifacts"].append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
            )
    write(OUT / "release-manifest.json", manifest)
    print(
        json.dumps(
            {
                "sealed": len(manifest["artifacts"]),
                "charged_seconds": charged,
                "remaining_seconds": after["remaining_seconds"],
                "archives": {n: (OUT / n).stat().st_size for n in archives},
            }
        )
    )


if __name__ == "__main__":
    main()
