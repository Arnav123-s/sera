"""Write the acquisition-dependent results from independently audited artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/18_instance_transfer"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    result = read(RELEASE/"audit-summary.json")
    protocol = read(RELEASE/"protocol.json")["payload"]
    baseline = sum(read(RELEASE/f"seed-{seed}/observed_only/result.json")["metrics"]["final_parent"]["mse"]
                   for seed in protocol["seeds"])/len(protocol["seeds"])
    with (RELEASE/"report.md").open("x", encoding="utf-8", newline="\n") as out:
        out.write("# Did corrected acquired knowledge help the neural route?\n\n")
        verdict = "passes" if result["gate"]["causal_component_pass"] else "fails"
        out.write(f"The prospectively frozen acquisition-dependent component **{verdict} its gate**. I completed 20 trained descendants, five new learning streams and four controls. Every source/checkpoint, raw prediction, failed result and cost is preserved. This follows the [50-run normalized-teaching study](../17_transfer/report.md); its useful readout result remains separate from its failed acquired-coefficient attribution.\n\n")
        out.write("## What changed and what the model learned\n\n")
        out.write("I removed the normalization that canceled the source instance's fitted coefficients. The conditional teacher now generates trajectories in the original simulated circle's coordinates. Its actual pre-correction and post-correction checkpoints provide the K intervention; their initial neural weights are bitwise equal. A preflight verifies a nonzero teaching-target change before training.\n\n")
        out.write("Each stream receives 32 new simulator trajectories, each with three coordinate observations and a next-position target. Conditional arms add 512 trajectories from either corrected or incorrect K; the oracle-exposure control instead receives independent simulator trajectories at the matched phase schedules. The learner never sees those phases, center, radius or angular speed. The circle law and noiseless future simulator are supplied. This is one simulated physical instance, not a collection of independently acquired physical laws.\n\n")
        out.write("The same 514 existing readout parameters are trained for 256 updates, using a target batch of 32, an old-replay batch of 32 and 32 parent-output anchors. All other neural parameters and acquired generator buffers remain fixed. A separate 128-trajectory development set selects among four immutable steps. The final 256 ordinary and 128 faster-turning trajectories per stream are not used for training or selection.\n\n")
        out.write(f"The original K was acquired from {protocol['historical_acquisition']['observations_each']} observations. The corrected source additionally has {protocol['historical_acquisition']['corrections_after']} independent sensor recheck; its cost remains in the original study. I do not describe that extra information as free.\n\n")
        out.write("## Performance\n\n| Teaching source | Ordinary RMSE | Faster-turning RMSE | Worst old-score drop | Training seconds |\n|---|---:|---:|---:|---:|\n")
        for arm, row in result["aggregate"].items():
            out.write(f"| {arm} | {row['rmse']:.6f} | {row['extent_mse']**.5:.6f} | {100*row['maximum_retention_drop']:.3f} pp | {row['training_seconds']:.2f} |\n")
        out.write(f"\nThe unadapted neural parent's ordinary RMSE was **{baseline**.5:.6f}**. Scores above pool five paired learning streams from one pretrained parent. The 20 descendants make 7,680 final predictions on 1,920 unique final trajectories; repeated control predictions are not independent worlds.\n\n")
        for name, contrast in result["contrasts"].items():
            low, high = contrast["paired_bootstrap_97_5_percent"]
            out.write(f"Against **{name}**, corrected teaching changes mean MSE by **{100*contrast['relative_mse_gain']:.2f}%** in the favorable direction. The paired absolute-gain interval is **[{low:.7f}, {high:.7f}]**. This planned contrast **{'passes' if contrast['passes'] else 'fails'}**.\n\n")
        out.write("The gate requires at least 10% mean MSE gain against both observed-only and incorrect-K controls, positive paired 97.5% bootstrap intervals, improvement in every learning stream, and at most two points of loss in any of 61 retained groups. The intervals are descriptive estimates over five learning streams, not a universal or finite-sample guarantee.\n\n")
        if result["gate"]["causal_component_pass"]:
            out.write("Correcting the acquired instance helped this separate neural prediction route under the stated controls. The result concerns a protected readout within a supplied family. It does not show improved shared representations, learned grammar, hidden-state online investigation or an independently improved learning procedure.\n\n")
        else:
            out.write("The representation now preserves the acquired-state intervention, but that necessary condition did not suffice to meet the full useful-transfer gate. I retain the numerical gains and failures separately and do not promote this candidate. Another attempt must change the mechanism or evidence and reserve a new final bank; these outcomes will not be repeatedly tuned into a confirmation.\n\n")
        out.write("## Audit, persistence and next step\n\n")
        out.write(f"The independent auditor restored every selected checkpoint and checked all **{result['checkpoint_predictions_checked']:,}** saved predictions exactly. Its separate geometric recurrence verified final targets with maximum error **{result['maximum_independent_geometric_target_error']:.3g}**. Retention summaries were reconstructed from the saved paired controls. Original owner/checkpoint identities remain unchanged.\n\n")
        integration = ROOT/"research-continuation/17_transfer/integration/result.json"
        if integration.exists():
            item = read(integration)
            out.write(f"The predeclared integration rule selected `{item['chosen_cell']}`. The [migration report](../17_transfer/integration/result.json) records a new experimental shared solver, explicit factual replay, full finite reproof, stale-record rejection and exact reload. Its original cumulative costs and knowledge remain preserved. The existing operational store was not replaced.\n\n")
        out.write("Full costs across both cycles, including failed checkers and development, are in the [shared cost ledger](../17_transfer/costs.json). The latest release ledger includes final verification. See the [architecture audit](../17_transfer/architecture-audit.md), [actual K intervention](../17_transfer/acquisition-intervention/result.json), [frozen protocol](protocol.json) and [independent results](audit-summary.json).\n\n")
        out.write("The unresolved architectural work is retained shared-representation learning, continuing hidden-state interaction and a separate test of eta. A readout improvement or accumulated K does not establish learning-to-learn. M2–M4 remain open.\n")
    print(str(RELEASE/"report.md"))


if __name__ == "__main__":
    main()
