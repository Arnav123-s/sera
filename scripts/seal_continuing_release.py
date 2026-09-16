"""Summarize and seal completed continuing-control evidence without new evaluation."""

import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from seal_transfer_release import archive, read, row, write

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/19_continuing"


def mean(values):
    return sum(values)/len(values)


def main():
    started = time.perf_counter()
    if (RELEASE/"release-manifest.json").exists() or (ROOT/"runs/v3-batch-001/active.lock").exists():
        raise ValueError("Release already sealed or an owned job is active")
    audit = read(RELEASE/"A08-FINAL-001/independent-audit.json")
    owner = read(RELEASE/"integration/owner.json")
    retention = read(RELEASE/"retention/result.json")
    tests = ET.parse(RELEASE/"tests.xml").getroot().findall("testsuite")
    test_count = sum(int(t.attrib["tests"]) for t in tests)
    if test_count != 219 or any(int(t.attrib.get(k, 0)) for t in tests for k in ("errors", "failures", "skipped")):
        raise ValueError("Missing complete local regression evidence")
    ledger = read(ROOT/"runs/v3-batch-001/budget.json")
    jobs = {k: read(ROOT/k/"state.json") for k in ledger["jobs"] if "/A08-" in k}
    charged = sum(r["charged_seconds"] for r in jobs.values())
    logs = []
    for key, value in jobs.items():
        if row(ROOT/key/"process.log")["sha256"] != value["log_sha256"]:
            raise ValueError("Resource log changed")
        if value["charged_seconds"] != ledger["jobs"][key]["charged_seconds"]:
            raise ValueError("Budget and process charges differ")
        logs.extend([ROOT/key/"process.log", ROOT/key/"state.json"])
    archive(RELEASE/"worker-logs.zip", logs)
    write(RELEASE/"worker-budget.json", ledger)
    costs = {"schema": "sera.continuing-full-worker-costs.1", "jobs": jobs,
             "charged_worker_seconds": charged, "remaining_worker_seconds": ledger["remaining_seconds"],
             "peak_job_committed_bytes": max(j["peak_job_committed_bytes"] for j in jobs.values()),
             "development_lifetimes": 42, "prospective_lifetimes": 168,
             "development_observations": 2646, "prospective_observations": 12600,
             "post_restore_observations": 4, "all_new_observations": 15250,
             "development_actions": 2604, "prospective_actions": 12432, "post_restore_actions": 4,
             "original_acquisition_cost": "16 geometric observations plus one correction and previous project costs remain in their preserved cumulative lineage; they are not free or charged again as newly acquired data.",
             "boundary": "Includes every supervised test, failed development candidate, pilot, freeze, prospective execution, independent audit, integration, status, regression and fresh retention job. Interactive research/engineering/file operations are outside measured worker time; subsequent release verification records its own elapsed time. No exact FLOPs or total human/interactive-time estimate. No compute purchased.",
             "logs": row(RELEASE/"worker-logs.zip")}
    write(RELEASE/"costs.json", costs)
    old_manifest = read(ROOT/"research-continuation/18_instance_transfer/release-manifest.json")
    for expected in old_manifest["files"]:
        if row(ROOT/expected["path"]) != expected:
            raise ValueError(f"Preserved release changed: {expected['path']}")
    old = read(ROOT/"research-continuation/18_instance_transfer/preservation.json")
    for expected in old["predecessor_files"]:
        if row(ROOT/expected["path"]) != expected:
            raise ValueError(f"Original checkpoint changed: {expected['path']}")
    write(RELEASE/"preservation.json", {"status": "PASS", "unchanged_previous_release_files": len(old_manifest["files"]),
          "previous_manifest": row(ROOT/"research-continuation/18_instance_transfer/release-manifest.json"),
          "unchanged_original_predecessors": len(old["predecessor_files"]),
          "unchanged_shared_sources": old["unchanged_shared_sources"], "original_operational_store_replaced": False,
          "completed_experiments_restarted": 0, "full_tensor_and_behavior_retention": retention})
    local = [row(p) for name in ("A08-PILOT-001", "A08-PILOT-002", "A08-FINAL-001")
             for p in sorted((ROOT/"runs"/name).glob("*.json"))]
    write(RELEASE/"local-artifacts.json", {"files": local, "bytes": sum(r["bytes"] for r in local),
          "immutable_base": row(ROOT/"runs/A06-integrated-001/solver/versions/v0.pt"),
          "scope": "Local per-lifetime state snapshots; full public raw records also embed their state. Original neural weights are referenced, not duplicated per lifetime."})
    write(RELEASE/"state.json", {"schema": "sera.continuing-state.1", "status": "qualified_control_component",
          "starting_head": "29b06a1173bd525e31920c8388f2ab5a8b185ea7", "branch": "research/v3-calibration-consolidation",
          "completed_lifetimes": 210, "prospective_lifetimes": 168, "world_units": 24,
          "independent_decisions_checked": 12096, "tests": 219, "fresh_retention_groups": 61,
          "current_solver": owner["solver_sha256"], "owner_bundle": "integration/owner.json",
          "continuing_observations": 79, "current_world_clock": 78, "owned_jobs": [],
          "resume_command": ".venv/Scripts/python.exe -X utf8 -m experiments.continuing_control.integration status",
          "live_budget": "runs/v3-batch-001/budget.json", "remaining_worker_seconds": ledger["remaining_seconds"],
          "remote_publication": False, "allowance_exhausted": False,
          "open": ["Independent eta learning and K-by-eta future-task controls", "Calibrated adaptive-policy uncertainty", "Retained shared-representation plasticity", "General perception/relational acquisition and learned latent migration"],
          "next_protocol": "../NEXT_EXPERIMENT_PROTOCOL.md"})
    a = audit["aggregate"]["gain_reversal"]
    q = audit["paired_nominal_comparisons"]
    lines = ["# Useful imagination and online correction in a continuing learner", "",
             "I completed two development candidates (42 continuing lifetimes) and a separately frozen 168-lifetime study in 24 paired simulated worlds. The revised component passes its declared gate and is integrated into a persistent experimental descendant of the actual A06 learner. The original operational model and all rejected work remain preserved.", "",
             "## What it learned and did", "",
             "The learner observes noisy positions and its own actions, infers motion without receiving phase or velocity, and learns three action-response coefficients from real transitions. It imagines four-action sequences, executes the first action, observes the result and revises its dynamics. An observed-innovation detector handles a reversal of actuator response without discarding factual history. Every target change happens within the same continuing world; there is no reset between questions.", "",
             "The acquired circle coefficients remain useful input, while the action grammar, linear estimator, detector and planning procedure are supplied. The same shared R1 gains 248 bytes of registered statistics/state; no new neural parameters are trained. This is K acquisition under an engineered eta, not learned learning-to-learn.", "",
             "## Prospective performance", "",
             "Lower physical cost is better. It combines squared target-position error, action effort and velocity penalty. Each nominal comparison uses 12 paired worlds; the other 12 worlds add a force omitted from the learned dynamics grammar.", "",
             "| Controller | Nominal physical cost | Omitted-force cost | Nominal priced work proxy | Mean nominal lifetime seconds |",
             "|---|---:|---:|---:|---:|"]
    for arm, values in a.items():
        stress = audit["aggregate"]["omitted_torque"][arm]
        lines.append(f"| {arm} | {values['cost']:.6f} | {stress['cost']:.6f} | {values['priced_cost']:.6f} | {values['seconds']:.3f} |")
    lines += ["", f"Four-step imagination reduces nominal physical cost by **{100*q['adaptive_reactive']['relative_cost_reduction']:.2f}%** versus reactive control and **{100*q['adaptive_one_step']['relative_cost_reduction']:.2f}%** versus one-step prediction. Every one of the 12 nominal worlds improves both comparisons.", ""]
    for key in ("adaptive_reactive", "adaptive_one_step"):
        lower, upper = q[key]["interval_97_5_percent"]
        lines.append(f"The paired absolute-gain 97.5% bootstrap interval against `{key}` is [{lower:.6f}, {upper:.6f}].")
    lines += ["", f"The prespecified priced proxy improves **{100*(1-a['adaptive_mpc']['priced_cost']/a['adaptive_reactive']['priced_cost']):.2f}%** over reactive control. Post-reversal physical cost falls **{100*(1-a['adaptive_mpc']['post_change_cost']/a['frozen_mpc']['post_change_cost']):.2f}%** versus frozen dynamics. Actual computation is higher: nominal lifetime time is {a['adaptive_mpc']['seconds']:.3f}s versus {a['adaptive_reactive']['seconds']:.3f}s. Equal measurement counts and maximum decision allowances are not exact machine-cost equality. The proxy's explicit prices are in the frozen protocol.", "",
             f"Using corrected rather than the actual incorrect predecessor geometry reduces nominal mean cost by {100*q['incorrect_geometry']['relative_cost_reduction']:.2f}%; two individual worlds go the other way. This secondary contrast is not a uniform transfer guarantee. The privileged linear-coefficient reference is stronger on average and still lacks hidden state and the omitted force.", "",
             "## What failed or remains uncertain", "",
             "The first development candidate barely beats one-step prediction (0.70260 versus 0.70289). Retaining every old transition in one stationary fit is harmful after response reversal. Those results and source versions remain preserved. The revised detector was tested on new development seeds, then frozen before prospective worlds; no final-bank tuning occurred.", "",
             "Conditional uncertainty is incomplete. The table below reports actual coverage of the diagnostic coordinatewise 1.96-standard-deviation rectangles, alongside rollout error. These rectangles are not jointly calibrated 95% regions and are not an answer certificate.", "",
             "| World family | One-step rollout RMSE | Four-step rollout RMSE | One-step rectangle coverage | Four-step rectangle coverage |",
             "|---|---:|---:|---:|---:|"]
    for family in audit["aggregate"]:
        records = [r for r in audit["records"] if r["family"] == family and r["arm"] == "adaptive_mpc"]
        mse = [mean([r["rollout_mse_by_horizon"][j] for r in records]) for j in range(4)]
        coverage = [mean([r["conditional_interval_coverage_by_horizon"][j] for r in records]) for j in range(4)]
        lines.append(f"| {family} | {math.sqrt(mse[0]):.5f} | {math.sqrt(mse[3]):.5f} | {100*coverage[0]:.2f}% | {100*coverage[3]:.2f}% |")
    lines += ["", "Unpropagated geometry/state error and omitted dynamics limit these conditional intervals. Higher decision utility does not cure that uncertainty failure. Fresh policy-specific qualification is required before any supported-answer claim. All earlier statistical certificates remain attached to their original policies/owners.", "",
             "## Audit, retention and actual continuation", "",
             "The independent auditor reconstructs all **12,096 decisions**, checks noisy sensor provenance, uses complex-plane physical dynamics and QR-based coefficient fitting, and verifies action choices and counterfactual predictions. Maximum coefficient disagreement is below 6e-14. Each of the 168 final owners restores from its real history with original tensors unchanged. The 12,096 repeated decisions are not 12,096 independent worlds.", "",
             "The predeclared first nominal stream becomes the new experimental descendant. Its 75-observation session reloads and takes four further paid actions/observations, reaching 79 observations and world clock 78. Original facts and the sensor correction replay under the new owner; stale factual/library bindings are rejected and finite programs fully re-proved. The finite example `4*x-5=6 modulo 11` still returns 0.", "",
             "A separate fresh retention bank gives **exactly identical group and case scores across 61 groups**. **219 regressions pass**, including evidence isolation, mutation-bound stale plans, genuine correction, future-decision equivalence after restore, and migration-scope rejection. This does not establish long-term neural plasticity: old neural representations are held fixed.", "",
             "## Costs, preservation and next state", "",
             f"This continuation charges **{charged:.2f} supervised worker seconds ({charged/60:.2f} minutes)**, including development, all verification and fresh retention. Peak job committed memory is **{costs['peak_job_committed_bytes']:,} bytes** under the original 2 GiB ceiling. The cumulative allowance has **{ledger['remaining_seconds']:.2f} seconds remaining**; it is not exhausted or expanded. There are 15,250 newly acquired sensor measurements across development, prospective controls and four continuation steps. Historical source-acquisition costs remain in the original lineage. Interactive research/engineering time is outside the measured worker boundary.", "",
             "The previous 451-file release and 108 original predecessor artifacts remain byte-identical, including all 45 shared interpreter sources. The full local ledger, failed candidates, per-lifetime raw records, frozen source archives and checkpoint inventory are retained. No background experiment is left running and no remote publication was performed.", "",
             "See [the source-architecture audit](architecture-audit.md), [complete independent outcomes](A08-FINAL-001/independent-audit.json), [costs](costs.json), [failures](failures.md), [source research](research-notes.md) and [resumption commands](resume.md). The next major requirement is independently learned eta on unfamiliar continuing tasks, with K and eta crossed separately; uncertainty qualification and retained shared-representation learning remain open. M2–M4 are not declared complete."]
    with (RELEASE/"report.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines)+"\n")
    sources = {ROOT/name for name in read(RELEASE/"A08-FINAL-001/protocol.json")["payload"]["sources"]}
    sources.update((ROOT/"experiments/continuing_control").glob("*.py"))
    sources.update(ROOT/name for name in ("experiments/continuing_postflight.py", "scripts/run_continuing_postflight_bounded.py",
                    "scripts/seal_continuing_release.py", "scripts/verify_continuing_release.py", "tests/test_continuing_integration.py"))
    write(RELEASE/"current-source-manifest.json", {p.relative_to(ROOT).as_posix(): row(p)["sha256"] for p in sorted(sources)})
    archive(RELEASE/"current-sources.zip", sources)
    files = sources | {p for p in RELEASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    write(RELEASE/"release-manifest.json", {"schema": "sera.continuing-release.1", "files": [row(p) for p in sorted(files)],
          "scope": "Exact public evidence and source bytes; excludes this manifest and the subsequent verification receipt. Per-lifetime local state and the immutable neural base are separately inventoried."})
    print(json.dumps({"status": "SEALED", "files": len(files), "local_checkpoints": len(local), "tests": test_count,
                      "worker_seconds": charged, "remaining_seconds": ledger["remaining_seconds"], "sealing_seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
