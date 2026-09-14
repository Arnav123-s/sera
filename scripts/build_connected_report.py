"""Build the connected report from measured JSON, including a clean-checkout path."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
METHODS = ("none", "update", "replay", "evidence", "planning", "program")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def mean(values):
    return statistics.mean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def estimate(values, *, scale=1, suffix="", digits=2):
    values = [scale * value for value in values]
    return f"{mean(values):.{digits}f} ± {sd(values):.{digits}f}{suffix}"


def rounds(run):
    return [*run["symbolic_generations"], run["feedback_round"],
            *[row["result"] for row in run["autonomous_generations"]]]


def operation_totals(runs, episodes):
    counts = Counter()
    def add(work):
        counts.update(work["operations"])
    for run in runs:
        add(run["initial_work"])
        for row in rounds(run):
            add(row["work"])
        add(run["r2"]["training_work"])
        for trial in run["r2"]["trials"]:
            add(trial["work"])
        counts["policy_optimizer_steps"] += run["policy_training"]["steps"]
    for episode in episodes:
        add(episode["diagnosis_and_acquisition_work"])
        for trial in episode["outcomes"].values():
            # charged_work includes construction and a planning difference already present
            # in evaluation_work. Adding it again would double-count planning.
            add(trial["construction_work"])
            add(trial["evaluation_work"])
    return dict(sorted(counts.items()))


def build(input_directory=None):
    OUT.mkdir(exist_ok=True)
    if input_directory is not None:
        data = read(input_directory / "summary.json")
        episodes = [read(path) for run in data["runs"]
                    for path in sorted((input_directory / str(run["seed"]) / "policy-episodes").glob("*.json"))]
        verification = read(input_directory / "verification.json")
        write(OUT / "connected-study-data.json", data)
        write(OUT / "connected-policy-episodes.json", episodes)
        write(OUT / "connected-verification.json", verification)
    data = read(OUT / "connected-study-data.json")
    episodes = read(OUT / "connected-policy-episodes.json")
    verification = read(OUT / "connected-verification.json")
    runs = data["runs"]
    learned = [r["policy_test"]["mean_utility"] for r in runs]
    fixed = {method: [r["policy_test"]["fixed_policy_utilities"][method] for r in runs] for method in METHODS}
    strongest_fixed = max(fixed, key=lambda method: mean(fixed[method]))
    choices = [case for run in runs for case in run["policy_test"]["cases"]]
    autonomous = [row["result"] for run in runs for row in run["autonomous_generations"]]
    counts = operation_totals(runs, episodes)
    write(OUT / "connected-operation-counts.json", {
        "counts": counts, "recorded_operation_subtotal": sum(counts.values()),
        "scope": "Nonoverlapping recorded phases; heterogeneous units, not FLOPs. Meta-episode baseline evaluation counters were not serialized. Total wall time includes that work.",
        "study_wall_seconds": sum(r["total_wall_seconds"] for r in runs)})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
    fig.set_facecolor("#f7fafc")
    positions = np.arange(3)
    for offset, suffix, label, color in ((-.18, "", "Recurrent state", "#147e80"),
                                       (.18, "_reset_memory", "Reset state each step", "#94a6bb")):
        values = [[100 * r["r1_evaluation"][name + suffix]["accuracy"] for r in runs]
                  for name in ("visible", "partial", "long")]
        axes[0].barh(positions + offset, [mean(v) for v in values], xerr=[sd(v) for v in values],
                     height=.32, color=color, capsize=3, label=label)
    axes[0].set_yticks(positions, ["Visible / 12 actions", "40% missing / 12", "40% missing / 24"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 103)
    axes[0].set_xlabel("Next-sensor accuracy (%)")
    axes[0].set_title("R1: retained state under missing observations", loc="left", weight="bold", pad=15)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    names = ["learned", *METHODS]
    values = [learned, *[fixed[m] for m in METHODS]]
    axes[1].barh(np.arange(len(names)), [mean(v) for v in values], xerr=[sd(v) for v in values],
                 color=["#147e80", *["#94a6bb"] * len(METHODS)], capsize=3, height=.65)
    axes[1].set_yticks(np.arange(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#64748b", linewidth=.8)
    axes[1].set_xlabel("Measured held-out intervention utility")
    axes[1].set_title("Policy: learned choice versus fixed methods", loc="left", weight="bold", pad=15)
    for ax in axes:
        ax.set_facecolor("#f7fafc")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("SERA / Connected learning study", x=.02, ha="left", weight="bold", fontsize=18)
    fig.text(.02, .02, "Three seeds; error bars are sample SD, not confidence intervals. Policy utility includes prior-world regression and counted-cost penalties.",
             fontsize=8.5, color="#4b5b6b")
    fig.tight_layout(rect=(0, .10, 1, .93), w_pad=3)
    fig.savefig(OUT / "connected-study.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    lines = ["# SERA: connected learning study", "",
             "I trained three independent SERA solvers from scratch, connected their R1/R2 learning paths, and measured both successful and rejected improvements. This report describes version 0.2; the first release's experiments remain historical evidence.", "",
             "## Findings", "",
             f"After compact R1 training, next-sensor accuracy with 40% missing readings was {estimate([r['r1_evaluation']['partial']['accuracy'] for r in runs], scale=100, suffix='%')}. Resetting the same model's memory at every step reduced it to {estimate([r['r1_evaluation']['partial_reset_memory']['accuracy'] for r in runs], scale=100, suffix='%')}. This ablation measures dependence on retained state; it is not a comparison with a separately trained memoryless model.", "",
             f"The learned improvement policy achieved utility {estimate(learned, digits=4)} on the withheld reset family. The strongest fixed-method mean was {strongest_fixed}, at {estimate(fixed[strongest_fixed], digits=4)}. " +
             ("The learned policy did not beat that fixed alternative on the three-seed mean. I do not claim a general improvement-policy advantage." if mean(learned) <= mean(fixed[strongest_fixed]) else
              "Its mean exceeded every tested fixed method, but twelve finite test episodes and three seeds do not establish a general policy advantage."), "",
             f"Across {len(autonomous)} later policy-selected proposals, {sum(r['status'] == 'promoted' for r in autonomous)} were promoted and {sum(r['status'] == 'rejected' for r in autonomous)} rejected. Ordinary reload reproduced the accepted behavior, and all three parent-rollback checks passed.", "",
             "![R1 state ablation and learned-policy comparison](connected-study.png)", "",
             "## What I taught it", "",
             "| Component | Available information and target | Budget per seed |",
             "|---|---|---|",
             "| Symbolic foundation | Explicit task instruction and 20-feature symbolic events; marked retrieval, latest binding, ordered actions and majority labels | 500 AdamW updates × 64 = 32,000 example draws; length 12; validation-selected weights |",
             "| R1 world learner | Visible/missing colors, previous action, actual reward, goal and public world identity; next visible color and actual reward targets | 512 × 8 = 4,096 collected transitions; 900 updates × 32 trajectories = 230,400 transition draws |",
             "| Ordered-control skill | Active reset/execution queries to the supplied deterministic world | Bounded transition induction and independent verification at lengths 12, 24 and 96 |",
             "| Earliest-binding skill | 128 support labels distinguish four supplied sequence rules; 256 independent verification examples | Select and verify a rule from the finite candidate set; no neural weight update |",
             "| Improvement policy | Observed diagnostics and measured counterfactual utilities for six supplied methods | 8 training + 4 validation episodes; 500 fitting steps; final 4 reset-family episodes after freezing |",
             "| Planned-feedback update | Actual consequences of executing the current R1 planner in a new permutation world | 32 × 8 practice transitions; 120 replay updates; fresh old/new-world admission |",
             "| Autonomous generations | New-world support, saved policy and incumbent solver | Two generations; 32 support trajectories each; 16 updates when selected; fresh retention/admission |",
             "| Independent R2 study | 256 × 8 masked trajectories and eight independently verified start/goal demonstrations | 200 instrument updates; 51,200 likelihood transition draws plus 1,600 verified-program credit draws |", "",
             "These are deliberately small symbolic tasks. A visible color identifies a state within its world; the public world identifier, action alphabet, goals, finite grammar and task router are supplied. Internal simulator state and transition tables do not enter neural training. Missing sensor labels remain absent from admitted experience; evaluation truth is kept separately. Program discovery receives stronger reset/execution feedback than passive neural sequence supervision.", "",
             "## R1 prediction and actual control", "",
             "Each condition contains 256 independent trajectories per seed. Values below are mean ± sample SD across seeds. Training uses length 8 with 30% missing readings; tests use the declared longer lengths. NLL is in nats; reward MSE compares predicted probabilities to actual binary rewards.", "",
             "| Evaluation condition | Accuracy | NLL | Reward MSE |", "|---|---:|---:|---:|"]
    for name in ("visible", "visible_reset_memory", "partial", "partial_reset_memory", "long", "long_reset_memory"):
        rows = [r["r1_evaluation"][name] for r in runs]
        lines.append(f"| {name} | {estimate([r['accuracy'] for r in rows], scale=100, suffix='%')} | {estimate([r['nll'] for r in rows], digits=4)} | {estimate([r['reward_mse'] for r in rows], digits=4)} |")
    lines += ["", "| Executed control | Success across 12 start/goal pairs per seed |", "|---|---:|"]
    for name in ("reactive", "planned", "partial_planned"):
        values = [mean([float(c["success"]) for c in r["r1_controls"][name]]) for r in runs]
        lines.append(f"| {name} | {estimate(values, scale=100, suffix='%')} |")
    lines += ["", "The base rotation worlds permit every nontrivial goal in one action. These control results exercise the connected path but cannot demonstrate a multi-step planning advantage. The new permutation/reset worlds below test adaptation under different dynamics. Partial-planned control masks later readings, so one-action success also limits how much that condition tests memory. Longer masked prediction is the stronger memory test here.", "",
              "## Symbolic behavior after all generations", "",
              "Ordinary solver inference uses accepted skills automatically. The final fresh pool has 512 examples per task per seed at length 12.", "",
              "| Task | Final accuracy | Mechanism |", "|---|---:|---|"]
    for task in runs[0]["symbolic_final"]["tasks"]:
        values = [r["symbolic_final"]["tasks"][task]["accuracy"] for r in runs]
        kind = "Verified executable skill" if task in {"ordered_control", "earliest_binding"} else "Trained neural model"
        lines.append(f"| {task} | {estimate(values, scale=100, suffix='%')} | {kind} |")
    lines += [f"| Equal-task macro | {estimate([r['symbolic_final']['macro_accuracy'] for r in runs], scale=100, suffix='%')} | Combined solver |", "",
              "Perfect finite-task scores do not show that the neural network learned the corresponding procedure: the two acquired skills execute supplied program representations, with explicit applicability routing. The world learner uses separate parameters, so its updates do not directly change the symbolic neural model.", "",
              "## Learned improvement policy", "",
              f"The study measured {sum(len(e['outcomes']) for e in episodes)} actual method outcomes across {len(episodes)} episodes. The final {len(choices)} test episodes use a reset family absent from policy development. Development and validation include repeated known worlds with separately sampled support/diagnostics; I count episodes, not independent world families. Policy weights were frozen before final outcomes were generated.", "",
              "Utility = target-world score gain − 2 × worst prior-world score loss − 0.002 × log(1 + charged operations). A world score equally weights prediction accuracy and executed goal success. Fixed program search falls back to no change when reset access is unavailable. This utility is a declared experimental objective, not accuracy or a FLOP-normalized rate.", "",
              "| Selection policy | Held-out utility |", "|---|---:|",
              f"| Learned selector | {estimate(learned, digits=4)} |"]
    for method in METHODS:
        lines.append(f"| Always {method} | {estimate(fixed[method], digits=4)} |")
    lines += ["", f"Mean regret to the post-hoc best available method: {estimate([r['policy_test']['mean_regret'] for r in runs], digits=4)}. Post-hoc best is an analysis upper bound, not a deployable policy. Selected methods: {dict(Counter(c['chosen'] for c in choices))}.", "",
              "| Test episode | Learned choice | Post-hoc best | Actual utility | Regret |", "|---|---|---|---:|---:|"]
    for case in choices:
        lines.append(f"| {case['episode_id']} | {case['chosen']} | {case['posthoc_best']} | {case['utility']:.4f} | {case['regret']:.4f} |")
    lines += ["", "This trains a selector over fixed learning procedures. It does not discover optimizers or invent learning algorithms. The model remains fixed during the subsequent generations; their outcomes do not establish a recursively improving improver. The complete meta-development cost must also be paid before deployment.", "",
              "## Every persistent proposal", "",
              "Each candidate is frozen before drawing fresh paired admission samples. Gain is the equal-task/world score difference; the lower bound is the declared one-sided bound. Old-world/task loss is the largest positive empirical regression. Both symbolic gates include five tasks; later gates include all worlds known at that point.", "",
              "| Seed / round | Method | Version / parent | Mean gain (pp) | Lower bound (pp) | Max loss (pp) | Decision |", "|---|---|---|---:|---:|---:|---|"]
    for run in runs:
        for row in rounds(run):
            decision = row["decision"]
            max_loss = max(0, *decision["retention_loss_by_task"].values())
            why = ", ".join(decision["reasons"])
            lines.append(f"| {run['seed']} / {row['round_index']} | {row['proposal']['method']} | {row['version']} / {row['parent']} | {100*decision['mean_gain']:.2f} | {100*decision['gain_lower_bound']:.2f} | {100*max_loss:.2f} | {row['status']}{': ' + why if why else ''} |")
    lines += ["", "Fresh promotion seeds and complete per-world/task scores appear in the JSON evidence. Empirical retention gates are not confidence bounds. Finite deterministic control outcomes are enumerated and independently sampled; this is not evidence of an unlimited distribution of goals. Rejected snapshots remain available, and no rejected candidate replaces current behavior.", "",
              "The generational study replays the original 512-trajectory world buffer; later support is archived separately. This restricted coverage can leave the previously learned permutation world vulnerable to forgetting, and the gate checks it even though it is absent from replay. `scripts/prepare_solver.py` creates a separate copy for continued learning and merges actual initial/planned/generation support without changing weights or admitting meta/query/test data. Preparation is not another learning result.", "",
              "| Seed | Current accepted version | Separate rollback | Total original study seconds |", "|---|---|---|---:|"]
    for run in runs:
        lines.append(f"| {run['seed']} | {run['current_version']} | {run['rollback']['before']} → {run['rollback']['restored']}: passed | {run['total_wall_seconds']:.1f} |")
    lines += ["", "## R2 learned proposals under an equal execution-attempt budget", "",
              "A separate permutation world supplies eight verified demonstrations per seed. Four other start/goal pairs per seed are withheld from those demonstrations. All methods have at most four candidate executions; a success also receives a separate verification execution. Transition likelihood training can see dynamics used by test programs. The holdout is goal composition within a trained world.", "",
              "| Search method | Successful pairs / 12 | Seed success rate | Mean attempts per pair | Proposal-model transitions |", "|---|---:|---:|---:|---:|"]
    for method in ("fixed", "untrained_instrument", "learned_instrument"):
        trials = [t for r in runs for t in r["r2"]["trials"] if t["method"] == method]
        rates = [mean([float(t["success"]) for t in r["r2"]["trials"] if t["method"] == method]) for r in runs]
        transitions = sum(t["work"]["operations"].get("program_proposal_model_transitions", 0) for t in trials)
        lines.append(f"| {method} | {sum(t['success'] for t in trials)} / {len(trials)} | {estimate(rates, scale=100, suffix='%')} | {mean([t['attempts'] for t in trials]):.2f} | {transitions:,} |")
    residual = max(max(r["r2"]["training"]["validity"].values()) for r in runs)
    lines += ["", f"Largest trained channel/instrument completeness residual: {residual:.3g}. Numerical validity is an implementation property, not proof of useful program learning. Execution-attempt equality does not mean equal action count, training compute or wall time; guided search pays for model proposals and training.", "",
              "The interpreter implements action, sequence, bounded repeat and library call. These trials mainly learn action sequences; I have not established that acquired abstractions reduce future search cost. The full controlled instrument is a classical numerical model and provides no quantum-device speedup claim.", "",
              "## Examples of behavior after reload", ""]
    for audited in verification["runs"]:
        examples = audited["execution_examples"]
        if examples:
            e = examples[0]
            lines.append(f"- Seed {audited['seed']}, {e['world_id']}: start {e['start']}, goal {e['goal']}; loaded skill executed actions {e['actions']}, observed {e['observations']}, received rewards {e['rewards']}; success={e['success']}.")
    lines += ["", "These are actual executions reproduced from the accepted solver, not generated narrative examples. The verification artifact includes further traces and parameter counts.", "",
              "## Costs, provenance and reproduction", "",
              f"The three original studies took {sum(r['total_wall_seconds'] for r in runs)/60:.2f} minutes in total on the shared CPU workstation. This includes development counterfactuals, final tests, failed/rejected candidates and rollback checks, but excludes earlier development probes and the additional reproduction audit ({verification['additional_audit_wall_seconds']/60:.2f} minutes). It is not an isolated throughput benchmark.", "",
              f"The independent artifact audit reproduced {sum(len(r['rounds']) for r in verification['runs'])} paired admission decisions, checked {verification['unique_meta_support_records']:,} nonoverlapping meta-support identifiers and {verification['unique_meta_query_datasets']} unique meta-query datasets, reloaded all accepted component identities, re-executed saved goal skills and reproduced final symbolic scores. This audit reused recorded seeds to verify software; it is not a new generalization estimate.", "",
              "[Operation counters](connected-operation-counts.json) aggregate nonoverlapping recorded phases. Categories include acquisition, observed sensors, optimizer work, replay, environment actions, program verification, model proposals/planning and paired evaluation. They are unlike units, not FLOPs. Meta-episode baseline-evaluation counters were not serialized, and individual phase timers are not complete; total study wall time includes those computations. I make no full-compute efficiency claim. Nested construction/charged-work snapshots are not counted twice.", "",
              f"Frozen implementation source SHA-256: `{data['manifest']['environment']['source_sha256']}`. Environment: Python {data['manifest']['environment']['python']}, PyTorch {data['manifest']['environment']['torch']}, NumPy {data['manifest']['environment']['numpy']}, CPU. Each run records its data/solver identities and promotion seeds.", "",
              "The report builder reads [study records](connected-study-data.json), [all policy outcomes](connected-policy-episodes.json) and [reproduction checks](connected-verification.json). The [evidence manifest](connected-evidence-manifest.json) records their hashes. Raw local checkpoints, candidate snapshots, trajectories, paired-score arrays and journals are retained outside Git history. See the [reproduction guide](../research/reproducing-connected-study.md) for exact commands.", "",
              "## What remains unproven", "",
              "The full 256-wide reference preset and its 35,840-byte core are implemented and tested, but this study trains a compact delta R1. Sensor colors, world identity, action vocabulary, goals, program grammar, task routing and intervention procedures are supplied. There is no learned perception/language stack, broad program-structure transfer, R3–R8 implementation, open-ended optimizer invention, independent evaluator process isolation, quantum advantage or demonstrated recursive research acceleration. These remain separate acceptance criteria in the [roadmap](../research/roadmap.md).", "",
              "I preserve the [development failures](../research/development-log.md), [original architecture audit](architecture-alignment-audit.md), [resolution](architecture-audit-resolution.md) and [implementation checklist](../research/implementation-checklist.md). Passing engineering checks does not turn a negative learning comparison into a capability claim.", ""]
    (OUT / "connected-study.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    names = ["connected-study-data.json", "connected-policy-episodes.json", "connected-verification.json",
             "connected-operation-counts.json"]
    write(OUT / "connected-evidence-manifest.json", [
        {"file": name, "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
         "bytes": (OUT / name).stat().st_size} for name in names])
    print(f"Built connected report from {len(runs)} seeds and {len(episodes)} measured policy episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Replace selected evidence with this audited local study")
    args = parser.parse_args()
    build(args.input)
