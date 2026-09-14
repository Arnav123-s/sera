"""Build teaching/performance tables from committed source-separated study evidence."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
data = json.loads((reports / "stage-three-data.json").read_text(encoding="utf-8"))
original = json.loads((reports / "stage-three-source-reproduction.json").read_text(encoding="utf-8"))
runs = data["runs"]


def average(values, *, percent=False, places=2):
    values = np.asarray(values, float) * (100 if percent else 1)
    return f"{values.mean():.{places}f} ± {values.std(ddof=1):.{places}f}" if len(values) > 1 else f"{values[0]:.{places}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


text = ["# SERA 0.3 teaching and performance study", "",
        "I completed the next R1/R2 implementation stage and tested its behavior against the original architecture references. This report separates working software, measured learning and remaining research claims. Values below are means ± sample standard deviations across three independent seeds unless a count is shown. All final measurements use the source and configuration in the manifest.", "",
        f"Executable source SHA-256: `{data['manifest']['environment']['source_sha256']}`. Original archive SHA-256: `{data['manifest']['archive_sha256']}`.", "",
        "The [protocol](../research/stage-three-protocol.md) records data exposure, supplied representations, controls and cost boundaries. The [architecture audit](stage-three-architecture-audit.md) maps each earlier finding to implementation and evidence. [Raw study data](stage-three-data.json), [policy episodes](stage-three-policy-episodes.json), [independent verification](stage-three-verification.json) and [original benchmark reproduction](stage-three-source-reproduction.json) retain the details.", "",
        "![Controlled results](stage-three-results.png)", "",
        "## Original material reproduced independently", "",
        f"I reproduced all {len(original['mathematics'])} original numerical/gradient checks and rechecked {len(original['checkpoint_checks'])} supplied checkpoints. The maximum checkpoint accuracy difference was {max(row['max_accuracy_difference'] for row in original['checkpoint_checks']):.8f}. I also trained all {len(original['retraining'])} original model/seed combinations from scratch using their original 500-step protocol. These runs belong to the supplied mechanism benchmark; they are not additional SERA 0.3 seeds.", ""]
names = sorted({row["kind"] for row in original["retraining"]})
text += [table(["Original core", "Source ID accuracy %", "Fresh ID accuracy %"],
               [[name, average([row["source_id_macro"] for row in original["retraining"] if row["kind"] == name], percent=True),
                 average([row["fresh_id_macro"] for row in original["retraining"] if row["kind"] == name], percent=True)] for name in names]), "",
         "## What I taught the solver", "",
         "The saved solver includes a separately trained legacy sequence decoder, recurrent world model, general event instrument, typed observation learner, acquired procedures and learned improvement policy. The architecture documents determine the design; they are not a corpus of physics facts that this model has learned to recite. Training labels come from declared simulators and verified execution.", "",
         "Each seed receives 192 examples in each of seven typed tasks, 48 validation examples per task and 1,400 joint updates. Five input adapters process symbolic events, numerical values, short byte strings, 4x4 image patches and 16-sample audio frames. Units, scale, position and availability masks are explicit. The arithmetic procedure is selected from 21 supplied candidates by support consistency and separate validation. I report its contribution with the neural weights frozen.", ""]
tasks = list(runs[0]["typed"]["tests"]["ordinary"]["tasks"])
rows = []
for task in tasks:
    if task == "motion":
        continue
    rows.append([task,
                 average([run["typed"]["tests"]["ordinary"]["tasks"][task]["score"] for run in runs], percent=True),
                 average([run["typed"]["tests"]["structure"]["tasks"][task]["score"] for run in runs], percent=True),
                 average([run["typed"]["with_programs"]["structure"]["tasks"][task]["score"] for run in runs], percent=True)])
text += [table(["Task", "Ordinary neural accuracy %", "Structural neural accuracy %", "Structural with procedures %"], rows), "",
         "Motion predicts two coordinates in meters. Ordinary MSE: " + average([run["typed"]["tests"]["ordinary"]["tasks"]["motion"]["loss"] for run in runs], places=4) +
         "; structural MSE: " + average([run["typed"]["tests"]["structure"]["tasks"]["motion"]["loss"] for run in runs], places=4) + " square meters. This is a regression error, not classification accuracy.", "",
         "Longer sums and binding histories, whitespace changes, spatial range shifts, patch occlusion, lower-amplitude tones and faster motion define the declared shifts. Full NLL/Brier scores and concrete input/expected/predicted records are in each seed's `typed` evidence. High patch/tone accuracy refers to these small generators. Arithmetic procedure success does not make the neural arithmetic result disappear.", "",
         "## History-preserving event prediction", ""]
predictor_names = list(runs[0]["predictors"]["models"])
text += [table(["Predictor", "Likelihood real parameters", "Repeated accuracy %", "Repeated NLL", "Length-40 accuracy %"],
               [[name, runs[0]["predictors"]["models"][name]["likelihood_real_parameters"],
                 average([run["predictors"]["models"][name]["tests"]["repeated"]["accuracy"] for run in runs], percent=True),
                 average([run["predictors"]["models"][name]["tests"]["repeated"]["nll"] for run in runs], places=4),
                 average([run["predictors"]["models"][name]["tests"]["long-random"]["accuracy"] for run in runs], percent=True)] for name in predictor_names]), "",
         "The general instruments retain distinguishable history after an event and pass independent likelihood/physical checks. The projective control loses that capacity. Classical HMM and GRU controls receive the same support and training-time cap; unused proposal parameters are excluded from the likelihood comparison and reported separately. I do not claim quantum advantage. Poor GRU performance is a result of this small architecture and protocol, not evidence against recurrent models in general.", "",
         "## World learning, planning and adaptation", "",
         "Long masked-world next-observation accuracy: " + average([run["world"]["prediction"]["accuracy"] for run in runs], percent=True) +
         "%; resetting the trained memory: " + average([run["world"]["memory_reset"]["accuracy"] for run in runs], percent=True) + "%.", "",
         table(["Planner", "Success on 12 distinct start/goal pairs %"], [[name, average([np.mean([row["success"] for row in run["world"]["planning"][name]]) for run in runs], percent=True)] for name in ("reactive", "learned", "oracle")]), ""]
rows = []
for count in (8, 32, 128):
    for method in ("none", "update", "adapter", "replay", "scratch"):
        selected = [next(row for row in run["world"]["adaptation"] if row["support_count"] == count and row["method"] == method) for run in runs]
        rows.append([count, method, average([row["new"]["accuracy"] for row in selected], percent=True),
                     average([row["old"]["accuracy"] for row in selected], percent=True), average([row["new"]["brier"] for row in selected])])
text += [table(["New trajectories", "Update", "New accuracy %", "Old accuracy %", "New Brier"], rows), "",
         "Each update condition has 32 optimizer steps; no update has none. These are sample-size curves at fixed update budget. The oracle alone can query exact world transitions. The learned planner uses most-probable imagined observations, so this comparison does not establish calibrated belief planning.", "",
         "## Full reference memory and conditional work", ""]
rows = []
for routing in ("all", "top1"):
    for rank in (2, 4, 8):
        selected = [next(row for row in run["world"]["reference"] if row["routing"] == routing and row["rank"] == rank) for run in runs]
        rows.append([routing, rank, selected[0]["state_bytes"], average([row["prediction"]["accuracy"] for row in selected], percent=True),
                     average([row["evaluation_seconds"] for row in selected]), f"{max(row['discarded_mass_max'] for row in selected):.6f}"])
text += [table(["Routing", "Density rank", "Core bytes", "Accuracy %", "Evaluation seconds", "Largest discarded mass"], rows), "",
         "Both routing modes train the full-size reference preset. The rank sweep reuses rank-four-trained weights; ranks also change the initial factors. Actual executed branch-row counts and diagnostics are preserved. Top-1 routing executes only selected branches and uses a declared surrogate gradient. The discarded mass is a single-truncation quantity and does not certify total trajectory error.", "",
         "This training budget did not establish competitive prediction for the full preset. It does not justify replacing the compact core. The independent audit reconstructs each all-branch model from its original seed/support, matches its learning curve and evaluation, and saves an actual trained-session diagnostic trace.", "",
         "## Acquired program composition", ""]
rows = []
for name in ("fixed", "general", "classical"):
    for enabled in (False, True):
        cases = [row for run in runs for row in run["programs"]["cases"] if row["guide"] == name and row["library"] == enabled]
        rows.append([name, enabled, sum(row["success"] for row in cases), len(cases),
                     f"{np.mean([row['environment_actions'] for row in cases]):.2f}"])
text += [table(["Guide", "Acquired library", "Solved", "Queries", "Mean executed actions"], rows), "",
         "The requests specify whole four-input transformations absent from all one/two-action training programs. Earlier verified macros become callable units in later searches. Each condition has eight executions, a 128-action cap and a three-token interface; a macro can expand to several actions. This measures reuse under a token budget. Per-world guide training and library acquisition remain separate recorded costs. The automatic learning path also passes existing libraries into search, with domain and dependency-version checks.", "",
         "## Improving the intervention policy", ""]
rows = []
for generation in range(3):
    rows.append([generation + 1, 4 * (generation + 1),
                 average([run["outer"]["generations"][generation]["anchor"]["mean_utility"] for run in runs]),
                 average([run["outer"]["generations"][generation]["frontier"]["mean_utility"] for run in runs]),
                 average([run["outer"]["generations"][generation]["anchor"]["uniform_expected_utility"] for run in runs]),
                 average([max(run["outer"]["generations"][generation]["anchor"]["fixed_policy_utilities"].values()) for run in runs])])
text += [table(["Generation", "Cumulative training episodes/seed", "Anchor utility", "New-frontier utility", "Uniform anchor utility", "Posthoc best fixed anchor"], rows), "",
         f"The study measured {sum(run['outer']['measured_interventions'] for run in runs)} intervention outcomes across {sum(sum(run['outer']['episode_counts'].values()) for run in runs)} episodes. All nine policy versions have actual parameter updates. A fixed four-world anchor is reused only for evaluation; the frontier adds two worlds per generation. Fixed, uniform, difficulty-based and explicit-diagnosis comparisons use the same recorded counterfactual outcomes. The best fixed method column is posthoc and is not a deployable oracle-free selection rule.", "",
         "Utility is new-world gain minus twice the largest old-world regression and a declared logarithmic operation-cost penalty. Meta-training enumerates all available methods and is a real research cost. The inner solver is held fixed in this policy comparison. The separate saved-solver experiment below measures actual successive task updates. Improved policy weights alone do not prove sustained improvement of the learning procedure.", "",
         "## Persistent behavior, retention and rejection", ""]
rows = []
for run in runs:
    p = run["persistence"]
    decisions = [row["result"] for row in p["rounds"]] + [row["admission"] for row in p["mutations"] if "admission" in row] + [p["retrieved_lineage"]["admission"]]
    rows.append([run["seed"], p["current"]["version"], p["replay_records"], sum(row["status"] == "promoted" for row in decisions),
                 sum(row["status"] == "rejected" for row in decisions), f"{100 * max(max(row['decision']['retention_loss_by_task'].values()) for row in decisions):.2f}",
                 ", ".join(p["archive"]["useful_versions"])])
text += [table(["Seed", "Current", "Retained trajectories", "Promoted", "Rejected", "Worst attempted regression (points)", "Useful archived versions"], rows), "",
         "Learning support, active queries and program traces are retained across completed rounds, including rejected candidates. Budget and previous-attempt features come from persisted history. A finite typed adapter proposal passes validity and development screening before any fresh admission. Archive selection uses development data and retains immutable specialist versions; a retrieved specialist actually seeds a subsequent replay proposal. The current accepted version remains the rollback parent. All proposal decisions, rejection reasons, per-world retention and archive bytes are in the raw record.", "",
         "## Cost and limits", "",
         f"The final study invocation took {data['costs']['wall_seconds']/60:.2f} minutes of wall time and {data['costs']['process_cpu_seconds']/60:.2f} process CPU minutes. Process peak RSS was {data['costs']['process_peak_rss_bytes']/2**20:.1f} MiB. The separate original reproduction took {original['costs']['wall_seconds']/60:.2f} wall minutes. Phase records include acquisition, training, validation, execution, evaluation, rejected proposals and archive work. Pilot and interrupted-run records are preserved in the development audit. RSS is a cumulative process high-water mark, and heterogeneous operation counts are not FLOPs. External assistance, human labor and energy are not assigned a fabricated numeric cost.", "",
         "Three independent processes each use one Torch thread on the shared host. CPU time sums workers; reported peak RSS is the largest worker high-water mark rather than simultaneous aggregate memory. Hardware contention is included in observed wall timing.", "",
         "Some externally interrupted development processes did not flush complete wall/CPU records. Those costs are explicitly unavailable, not zero, in the development audit. This further prevents a claim about improvement per complete research cost.", "",
         "The engineering gaps have concrete implementations and tests. Positive learning claims remain limited by the observed outcomes: finite synthetic tasks, three seeds, fixed vocabularies, short programs, engineered arithmetic primitives and small policy episode sets. Broad perception/language, a learned optimizer, hostile-process evaluator isolation, R3–R8 integration, quantum advantage and sustained research acceleration remain unestablished. I keep these research limits explicit instead of converting completed experiments into unsupported capability claims.", ""]
(reports / "stage-three-study.md").write_text("\n".join(text), encoding="utf-8", newline="\n")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.hashsalt": "sera-stage-three"})
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), layout="constrained")
for index, name in enumerate(predictor_names):
    values = [100 * run["predictors"]["models"][name]["tests"]["repeated"]["accuracy"] for run in runs]
    axes[0].bar(index, np.mean(values), yerr=np.std(values, ddof=1), color="#276f8e", capsize=3)
axes[0].set(xticks=range(5), xticklabels=predictor_names, ylim=(0, 105), ylabel="Accuracy (%)", title="Aliased history: repeated actions")
axes[0].tick_params(axis="x", rotation=30)
selected_tasks = [task for task in tasks if task != "motion"]
x = np.arange(len(selected_tasks))
for offset, key, label, color in ((-.18, "tests", "Neural", "#799aa9"), (.18, "with_programs", "+ acquired procedure", "#28735c")):
    values = [[100 * run["typed"][key]["structure"]["tasks"][task]["score"] for run in runs] for task in selected_tasks]
    axes[1].bar(x + offset, [np.mean(v) for v in values], width=.35, color=color, label=label)
axes[1].set(xticks=x, xticklabels=[task.replace("_", "\n") for task in selected_tasks], ylim=(0, 105), title="Typed tasks: structural shift")
axes[1].tick_params(axis="x", labelsize=8)
axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.23), frameon=False)
for seed, run in enumerate(runs):
    axes[2].plot([1, 2, 3], [row["anchor"]["mean_utility"] for row in run["outer"]["generations"]], marker="o", label=f"Seed {seed}")
axes[2].axhline(0, color="#7c7c7c", lw=.8)
axes[2].set(xticks=[1, 2, 3], xlabel="Outer generation", ylabel="Declared utility", title="Policy updates: fixed anchor")
axes[2].legend(frameon=False, fontsize=8)
fig.suptitle("SERA 0.3 · controlled local experiments", fontsize=14, fontweight="bold")
fig.savefig(reports / "stage-three-results.png", dpi=180)
fig.savefig(reports / "stage-three-results.svg", metadata={"Date": None})
svg_path = reports / "stage-three-results.svg"
svg_path.write_text(svg_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
print("Built the SERA 0.3 report and figures from committed evidence.")
