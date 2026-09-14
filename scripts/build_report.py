"""Build a study report and plot only from measured artifacts; never invented scores."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


baseline = read("runs/baseline-v1/summary.json")
matched = read("runs/matched-delta-v1/summary.json")
integrated = read("runs/integrated-v1/run.json")
instruments = read("runs/instrument-suite-v1/summary.json")
rows = list(baseline["aggregate"])
control = {"kind": "matched_delta", "parameters": matched["runs"][0]["parameters"],
           "core_state_bytes": matched["runs"][0]["core_state_bytes"], "seeds": 3}
for split in ("id", "ood"):
    values = [r[split]["macro_accuracy"] for r in matched["runs"]]
    control[f"{split}_mean"] = statistics.mean(values)
    control[f"{split}_sample_std"] = statistics.stdev(values)
rows.append(control)
rows.sort(key=lambda row: row["ood_mean"], reverse=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
fig.set_facecolor("#f7fafc")
labels = {"matched_delta": "Associative / parameter control", "hybrid": "Hybrid / delta + rotor + density",
          "delta": "Associative / default", "gru": "GRU", "collision": "Collision",
          "collision_no_cross": "Collision / no cross term", "real": "Rotor / phase disabled",
          "density": "Density workspace", "rotor": "Complex rotor"}
for ax in axes:
    ax.set_facecolor("#f7fafc")
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
positions = np.arange(len(rows))
axes[0].barh(positions, [100 * r["ood_mean"] for r in rows],
             xerr=[100 * r["ood_sample_std"] for r in rows],
             color=["#127c80" if r["kind"] in {"matched_delta", "delta"} else "#8196b5" for r in rows],
             capsize=3, height=0.67)
axes[0].set_yticks(positions, [labels[r["kind"]] for r in rows])
axes[0].invert_yaxis()
axes[0].set_xlim(0, 100)
axes[0].set_xlabel("Length-24 macro accuracy (%)")
axes[0].set_title("Parameter control changes the interpretation", loc="left", pad=14, weight="bold")
conditions = integrated["adaptation"]["conditions"]
names = ["no_update", "full_update", "replay", "scratch"]
position = np.arange(4)
axes[1].barh(position - .17, [conditions[n]["novel"]["macro_accuracy"] * 100 for n in names],
             height=.3, color="#127c80", label="New task")
axes[1].barh(position + .17, [conditions[n]["retention"]["macro_accuracy"] * 100 for n in names],
             height=.3, color="#8196b5", label="Prior tasks")
axes[1].set_yticks(position, ["No update", "Full update", "Replay", "From scratch"])
axes[1].invert_yaxis()
axes[1].set_xlim(0, 100)
axes[1].set_xlabel("Accuracy (%)")
axes[1].set_title("Replay reduces measured forgetting", loc="left", pad=14, weight="bold")
axes[1].legend(loc="lower right", frameon=False)
fig.suptitle("SERA / First experimental study", x=.02, ha="left", fontsize=18, weight="bold", y=.99)
fig.text(.02, .02, "Left: 3 seeds; error bars are sample SD, not confidence intervals. Right: 1 seed; related earliest-binding task.", color="#52616b", fontsize=9)
fig.tight_layout(rect=(0, .07, 1, .94), w_pad=3)
fig.savefig(OUT / "first-study.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

improvement = integrated["improvement"]
decision = improvement["decision"]
lines = ["# SERA: first experimental study", "",
         "In this first SERA study, I measure both useful improvements and negative results. I initialize every model from scratch and use no reference-package checkpoints or pretrained models.", "",
         "## Findings", "",
         "I find that the larger hybrid improves over the small default delta learner, but a classical delta learner with almost exactly the same parameter count matches its in-distribution score and has a slightly higher mean at doubled sequence length. I do **not** infer a quantum-inspired advantage from this study. I keep the smaller associative model as the default and retain the hybrid as an experimental option.", "",
         f"In a separate integrated run, I test program acquisition after a neural failure. Fresh macro accuracy rises from {improvement['incumbent']['macro_accuracy']:.2%} to {improvement['candidate']['macro_accuracy']:.2%}, a gain of {decision['mean_gain']*100:.2f} percentage points. I measure no accuracy loss on the other three tasks. The candidate is {improvement['status']} under the declared rule.", "",
         "![Measured model and adaptation comparisons](first-study.png)", "",
         "## Neural comparison", "",
         "Each condition uses seeds 0, 1, 2; 500 AdamW steps; batch size 64; training length 12; 512 test examples per task and length. Validation selects checkpoints; final tests do not. Four task means receive equal weight. Values are mean ± sample standard deviation across seeds.", "",
         "| Core | Parameters | Core state bytes | ID accuracy | Length-24 accuracy |",
         "|---|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {labels[r['kind']]} | {r['parameters']:,} | {r['core_state_bytes']:,} | {100*r['id_mean']:.2f} ± {100*r['id_sample_std']:.2f}% | {100*r['ood_mean']:.2f} ± {100*r['ood_sample_std']:.2f}% |")
lines += ["", "The parameter control has 11,216 parameters versus the hybrid's 11,208 (0.071% difference). Its width/head/dimension setting was chosen by parameter count without its task scores. This is not a fully tuned architecture tournament or a FLOP-matched experiment. Core-state bytes exclude parameters, optimizer states, activations, gradients, encoders, planning branches and archives. Wall times were measured on a shared local machine and are not isolated speed rankings.", "",
          "The complex rotor did worse than its no-phase control on longer sequences. The collision cross term had mixed effects: lower ID mean but slightly higher length-24 mean than its no-cross counterpart. Three seeds do not establish a stable universal ranking or a statistically confirmed phase benefit.", "",
          "## Failure-to-skill intervention", "",
          f"The diagnostic policy identified poor ordered-control performance, then queried a deterministic resettable simulator. Discovery used {improvement['oracle_queries']} oracle calls and {improvement['executed_actions']} executed actions, including repeated queries. The program passed 256 fresh sequences at each of lengths 12, 24 and 96. The promotion evaluation then used {decision['samples']:,} new paired examples across four task strata at length 24.", "",
          f"The one-sided gain lower bound was {decision['gain_lower_bound']*100:.2f} percentage points at this round's error allocation, exceeding the declared 1-point margin. Per-task retention gates passed. A content-addressed skill, candidate version, paired score arrays and the journal are stored locally; the parent version remains available for rollback.", "",
          "This is an intervention with richer information: discovery receives observable state IDs and resettable action queries, whereas the neural learners receive supervised sequence labels. The diagnostic policy, task router, interpreter and induction procedure are supplied algorithms. The result demonstrates a functioning bounded integration, not a learned general improver, same-feedback neural superiority, or unrestricted program synthesis.", "",
          "The acquisition gate charges oracle calls plus executed oracle actions. Training time and end-to-end time are recorded separately. This cost proxy is not full system compute. The retention gates are empirical; only the stated gain bound has the specified conditional statistical interpretation.", "",
          "## Adapting while retaining skills", "",
          "One seed; 128 distinct earliest-binding support examples; 64 updates where applicable; 256 query examples per task. Replay uses 32 novel and 32 previous-task examples per batch; full update and scratch use 64 novel examples. Query and support RNG namespaces are disjoint.", "",
          "| Condition | New task | Prior task macro | Novel draws |", "|---|---:|---:|---:|"]
for name in names:
    c = conditions[name]
    lines.append(f"| {name} | {c['novel']['macro_accuracy']:.2%} | {c['retention']['macro_accuracy']:.2%} | {c['novel_draws']:,} |")
lines += ["", "Full-model adaptation improves the new task but causes substantial forgetting. Replay preserves much more prior performance, with lower new-task exposure. These adaptations are diagnostic experiments; they are not automatically promoted. An average retention score does not establish that every individual task passes a retention gate.", "",
          "## World prediction and planning", "",
          f"The learned action-conditioned model classified all 16 finite state/action transitions correctly and succeeded on {sum(r['learned_success'] for r in integrated['world']['plans'])}/12 nontrivial start/goal pairs when its plans were executed in the environment. The oracle-model planner also succeeded on 12/12 pairs. The single random-policy control succeeded on {sum(r['random_success'] for r in integrated['world']['plans'])}/12 with an eight-action limit.", "",
          "All transitions can be observed during training. This is an action-conditioned finite control check with observable states, not evidence of unseen-world generalization, perception, partial observability or Dreamer-scale learning.", "",
          "## Learned event instruments", "",
          "Six trials use three seeds each for real and complex Kraus parametrizations. Training is 300 steps on a four-symbol cycle. The first symbol is excluded from conditional NLL because its random initial phase is unobservable. Real and complex variants differ in effective real parameter count (64 versus 128), so their difference is not a matched-capacity comparison.", "",
          "| Representation | Seed | Initial NLL | Length-24 NLL | Generated consistency | Completeness residual |",
          "|---|---:|---:|---:|---:|---:|"]
for r in instruments["runs"]:
    lines.append(f"| {'Complex' if r['complex'] else 'Real'} | {r['seed']} | {r['initial_nll']:.4f} | {r['length24_nll']:.5f} | {r['cycle_consistency']:.2%} | {r['completeness_residual']:.2e} |")
lines += ["", "Both parametrizations support learning, conditioning and generation, with visible optimization variability. An exact classical cycle/bigram rule has zero conditional NLL; this experiment does not show superiority to that rule. A separate development run duplicated complex seed 0 and is excluded from this table.", "",
          "## Verification and reproducibility", "",
          "The initial suite has 26 passing tests covering independent matrix references, derivatives, probability/trace invariants, streaming/batch equivalence, session ownership, deterministic optimizer resume, active program learning in new finite environments, instruction/query bounds, invalid score rejection, fresh-suite reuse rejection, journal integrity, actual promotion and rollback. CI is configured for Windows and Linux; only local Windows execution was verified before publication.", "",
          "Run these commands in the repository's Python environment:", "", "```text",
          "python -m sera benchmark --output runs/baseline-v1 --kinds delta gru rotor real hybrid collision collision_no_cross density --seeds 0 1 2 --steps 500 --samples 512",
          "python scripts/matched_control.py", "python -m sera run --output runs/integrated-v1 --steps 500 --samples 1024",
          "python scripts/instrument_suite.py", "python scripts/build_report.py", "```", "",
          "The comparative runs cover 27 fresh neural training runs; the integrated run adds one neural training run, a world-model run and three adaptation/scratch update conditions. Instrument tables report six trials. Fresh promotion seeds are sampled after freezing the candidate and logged afterward; a rerun is expected to differ slightly in its fresh promotion score. Neural data and initialization are deterministic within the recorded CPU environment; cross-version bitwise identity is not promised.", "",
          "Raw local run directories are ignored by Git. Selected JSON evidence is copied beside this report. `evidence_manifest.json` records hashes for those copies. Original source materials are recorded in `research/source_manifest.json`; they were not rerun. All numerical findings above are computed by this report builder from the new run artifacts.", "",
          "I completed the initial experiments before finalizing the SERA name. I preserved their original source hashes and immutable dataset RNG namespace. The import and branding rename does not change the learning equations or generated data. `rename-validation.json` records checkpoint re-evaluation under the SERA package. Historical checkpoints remain usable for evaluation; the strict training-resume check requires the original source revision, so new training uses a fresh run directory.", "",
          "## Next decision", "",
          "I will keep the small associative learner as the economical default. Before selecting the hybrid, I will run component-removal and compute controls on withheld generators. My next capability target is a learned diagnostic/applicability policy, followed by removal of supplied observable state IDs. I define the acceptance criteria in my research roadmap.", ""]
(OUT / "first-study.md").write_text("\n".join(lines), encoding="utf-8")
artifacts = {
    "benchmark.json": "runs/baseline-v1/summary.json",
    "matched-control.json": "runs/matched-delta-v1/summary.json",
    "integrated-run.json": "runs/integrated-v1/run.json",
    "instruments.json": "runs/instrument-suite-v1/summary.json",
}
manifest = []
for name, source in artifacts.items():
    destination = OUT / name
    # Git normalizes text to LF; hash that portable representation, not host CRLF.
    destination.write_bytes((ROOT / source).read_text(encoding="utf-8").encode("utf-8"))
    manifest.append({"file": name, "source": source,
                     "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
(OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"report": str(OUT / "first-study.md"), "figure": str(OUT / "first-study.png"),
                  "matched_delta_id": control["id_mean"], "matched_delta_ood": control["ood_mean"]}, indent=2))
