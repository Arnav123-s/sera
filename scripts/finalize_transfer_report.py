"""Summarize completed immutable results; never train or resample assessment data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/17_transfer"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8", newline="\n")


def main():
    cohorts = {name: read(RELEASE/name/"audit-summary.json") for name in ("A06-TRANSFER-001", "A06-TRANSFER-002")}
    audit = read(RELEASE/"acquisition-intervention/result.json")
    budget = read(ROOT/"runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in budget["jobs"].items() if "/A06-" in k}
    records = {k: read(ROOT/k/"state.json") for k in jobs}
    costs = {"jobs": records, "charged_worker_seconds": sum(j["charged_seconds"] for j in jobs.values()),
             "remaining_allowance_seconds": budget["remaining_seconds"],
             "peak_job_committed_bytes": max(j["peak_job_committed_bytes"] for j in records.values()),
             "failed_or_interrupted_jobs": [k for k, v in jobs.items() if v["status"] != "PASS"],
             "boundary": "All bounded worker jobs, including imports, failed tests, pilots, checkpoint writing and assessment. Interactive research/file review/lint time is not included as worker time. No cloud expenditure.",
             "previous_costs": "The earlier 240.2498017 seconds and all earlier project studies remain in the cumulative ledger and their own releases."}
    write(RELEASE/"costs.json", costs)
    write(RELEASE/"worker-budget.json", budget)
    with (RELEASE/"report.md").open("x", encoding="utf-8", newline="\n") as out:
        out.write("# Neural motion transfer and acquired-knowledge audit\n\n")
        out.write("I completed two prospectively frozen scopes, five independent teaching/optimizer streams per scope, and five controls per stream: **50 trained descendants**. Three development candidates, one failed mechanism test and every checkpoint/cost remain preserved.\n\n")
        out.write("The numerical question is whether conditional circle teaching improves a different neural route. The stronger acquired-knowledge question has a separate answer: normalizing the source circle cancels all its fitted coefficients. I therefore reject a claim of causal transfer from those acquired coefficients, regardless of a numerical component gate.\n\n")
        out.write("## What was taught\n\n")
        out.write("Each learning stream has 32 independently simulated observed circle trajectories. Each trajectory supplies three equally spaced coordinate observations and one next-position target. Conditional arms add 512 generated trajectories; the simulator-exposure arm receives independent ground-truth targets for the same 512 worlds. Observed-only arms resample their original 32 worlds. The neural route never receives center, radius, angular velocity or phase and never calls the analytic generator at assessment. The grammar, transforms, noiseless-circle assumption and optimizer are supplied.\n\n")
        out.write("Every arm takes 256 AdamW updates with 32 target and 32 old-replay examples per update, plus 32 parent-output anchor examples. A separate 128-world development bank selects among four saved steps. Final assessment uses 256 new ordinary and 128 faster-turning worlds per stream. The two scopes share those sealed banks for a paired comparison, so they comprise 1,920 unique final worlds, not 19,200 independent worlds. The 19,200 descendant predictions reuse those worlds across arms/scopes.\n\n")
        out.write("## Prospective results\n\n| Scope | Control | Final RMSE | Faster-turning RMSE | Worst old-score drop | Training seconds | Extra parameters |\n|---|---|---:|---:|---:|---:|---:|\n")
        for name, summary in cohorts.items():
            scope = "Shared path (478,483)" if name.endswith("001") else "Readout (514)"
            for arm, row in summary["aggregate"].items():
                out.write(f"| {scope} | {arm} | {row['rmse']:.5f} | {row['extent_mean_mse']**.5:.5f} | {100*row['maximum_retention_drop']:.3f} pp | {row['training_seconds']:.2f} | {row['extra_owner_parameters']:,} |\n")
        out.write("\nRMSE is in the simulator's coordinate units. The worst retained drop covers all five streams and 61 groups: 40 existing output capabilities plus 21 neural-only typed groups. It is an empirical gate, not a population-wide retention guarantee.\n\n")
        for name, summary in cohorts.items():
            interval = summary["paired_stream_bootstrap_97_5_percent"]
            out.write(f"**{name}:** conditional teaching changes MSE by {100*summary['relative_mse_gain_vs_observed']:.2f}% relative to observed-only learning. The paired absolute MSE-gain interval is [{interval[0]:.6f}, {interval[1]:.6f}]. The predefined conditional component gate **{'passes' if summary['gate']['conditional_component_pass'] else 'fails'}**. Gate details: `{summary['gate']}`.\n\n")
        out.write("Detached and integrated arms must produce exactly the same trained state; their distinction is which owner is used by old routes and whether an additional full owner is retained. The extra-capacity observed-only control likewise matches observed-only training. All controls have the same optimizer starts, target/replay/anchor schedules and event shapes. Wall time and parameter/memory cost are measured separately; I do not turn matching update counts into a claim of exact hardware-cost equality.\n\n")
        out.write("## Acquired knowledge: a real intervention\n\n")
        out.write(f"I restored the original circle model before and after its independent sensor correction. Their neural parameters are bitwise identical, while their acquired coefficients differ. Across 1,024 transformed teaching worlds, the largest coordinate change was **{audit['normalized_maximum_difference']:.3g}**. The teaching transformation erased the intervention. On 257 raw-coordinate queries, the maximum prediction change was **{audit['raw_maximum_intervention_effect']:.5f}**; MSE changed from **{audit['before_raw_mse']:.6f}** to **{audit['after_raw_mse']:.6f}**. The original learned correction is useful; this teaching representation does not identify its contribution.\n\n")
        out.write("The new executable preflight rejects acquisition-attribution claims when an otherwise matched K intervention leaves the teaching channel unchanged. A detectable change is only necessary: future work must also show beneficial transfer, matched exposure/cost and retained competence.\n\n")
        out.write("## Persistence and integration\n\n")
        integration = RELEASE/"integration/result.json"
        if integration.exists():
            item = read(integration)
            out.write(f"The predeclared selection rule chose stream {item['chosen_stream']} from `{item['chosen_cell']}` for an **experimental persistent descendant**, `{item['solver_record']['solver_sha256']}`. I replayed 16 original observations and one correction, retained cumulative operation costs, rejected both stale state/library bindings, re-proved the finite programs under the new owner, and restored the complete solver exactly. The old operational solver and all original libraries/checkpoints remain unchanged. No statistical answer certificate was transferred, and this is not learned latent migration.\n\n")
        else:
            out.write("No descendant was integrated: integration is gated on the predefined readout component result. The original operational solver, factual state and finite library remain current. Rejected descendants remain available for analysis.\n\n")
        out.write("## Costs and evidence\n\n")
        out.write(f"At this report snapshot, bounded jobs charged **{costs['charged_worker_seconds']:.2f} seconds**, including failures, pilots, source freezes and verification. Peak owned-job committed memory was **{costs['peak_job_committed_bytes']:,} bytes** under a 2 GiB ceiling. The remaining cumulative allowance was **{costs['remaining_allowance_seconds']:.2f} seconds**. Final release verification may have a later ledger snapshot. Event counts, per-arm timing, all immutable checkpoints and exact source/runtime hashes are recorded separately. No paid compute or remote publication was used.\n\n")
        out.write("See [full costs](costs.json), [failed candidates](failures.md), [architecture comparison](architecture-audit.md), [primary research](research-notes.md), [shared-path audit](A06-TRANSFER-001/audit-summary.json), [readout audit](A06-TRANSFER-002/audit-summary.json), and [knowledge-intervention audit](acquisition-intervention/result.json).\n\n")
        out.write("The next research requirement is an acquisition-dependent task channel that passes the intervention preflight, followed by retained shared-representation learning. Hidden-state continuing interaction and independently improved eta remain open. I have not demonstrated M2–M4 or a self-improving general learner.\n")
    source = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("experiments/acquisition_dependence", "experiments/transfer_integration")
              for p in (ROOT/folder).glob("*.py")}
    write(RELEASE/"additional-source-manifest.json", source)
    print(json.dumps({"report": str(RELEASE/"report.md"), "costs": costs["charged_worker_seconds"]}))


if __name__ == "__main__":
    main()
