"""Publish compact measured evidence, costs and figures from immutable shared runs."""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sera.storage import write_json
from sera.training import source_hash

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/shared-learner-repaired"
REPORTS = ROOT / "reports"
METHODS = ("full", "replay", "adapter", "scratch", "scoped", "separate")
COLORS = {"delta": "#246b83", "reference": "#be6847"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def metric(values):
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "seeds": len(values)}


def fmt(row):
    return f"{100*row['mean']:.2f} ± {100*row['sample_std']:.2f}"


def clean_evaluation(value):
    result = copy.deepcopy(value)
    for task in result["tasks"].values():
        task.pop("control_cases", None)
    return result


def main():
    trials = [read(RUNS / str(seed) / kind / "trial.json") for seed in range(3) for kind in ("delta", "reference")]
    if any(r["status"] != "completed" or r["environment"]["source_sha256"] != source_hash() for r in trials):
        raise ValueError("Cannot report incomplete or source-mismatched runs")
    audits = [read(ROOT / f"runs/shared-learner-review/audit-{seed}.json") for seed in range(3)]
    if not all(row["passed"] for row in audits):
        raise ValueError("A replay audit failed")
    assembly = read(ROOT / "runs/shared-learner-assembly-repaired/assembly.json")
    if assembly["status"] != "completed":
        raise ValueError("Persistent solver assembly is incomplete")
    initial = read(ROOT / "runs/shared-learner-review/initial-run-disposition.json")
    preservation = read(ROOT / "runs/shared-learner-review/preservation-verification.json")
    live_audit = read(ROOT / "runs/shared-learner-review/live-verification.json")
    package = read(ROOT / "runs/shared-learner-review/package-check/verification.json")
    sources = read(ROOT / "runs/shared-learner-review/source-identities.json")
    suite = ET.parse(ROOT / "runs/shared-learner-review/test-results.xml").getroot().find("testsuite")
    tests = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    tests["seconds"] = float(suite.attrib["time"])
    if tests["failures"] or tests["errors"]:
        raise ValueError("A required local test failed")
    aggregate = []
    for kind in ("delta", "reference"):
        selected = [r for r in trials if r["kind"] == kind]
        for count in (32, 128, 512):
            for method in METHODS:
                rows = [next(x for x in r["runs"] if x["method"] == method and x["support_count"] == count) for r in selected]
                aggregate.append({"kind": kind, "method": method, "support_count": count,
                    "binding": {family: metric([r["evaluation"]["binding"]["earliest"][family]["accuracy"] for r in rows])
                                for family in ("ordinary", "long", "composition")},
                    "worst_retention_loss": metric([max(0, max(r["retention_audit"]["retention_loss_by_capability"].values())) for r in rows]),
                    "retention_passes": sum(not r["retention_audit"]["failed_capabilities"] for r in rows),
                    "parameters": rows[0]["parameters"]})
    compact = []
    for trial in trials:
        row = {key: copy.deepcopy(trial[key]) for key in ("seed", "kind", "environment", "budget", "parameters", "core_state_bytes", "base_tensor_sha256", "base_checkpoint", "pretraining", "costs", "work", "world")}
        row["baseline"] = clean_evaluation(trial["baseline"])
        row["runs"] = [{**{key: copy.deepcopy(value) for key, value in item.items() if key != "evaluation"},
                         "evaluation": clean_evaluation(item["evaluation"])} for item in trial["runs"]]
        compact.append(row)
    artifacts = [file_record(p) for p in sorted(RUNS.rglob("*")) if p.is_file()]
    data = {"schema_version": 1, "status": "completed", "source_sha256": source_hash(),
            "source_commit": "47429eb627a3add7cdc3dc77fe618aa492250e3d", "seeds": [0, 1, 2],
            "trials": compact, "aggregate": aggregate, "initial_attempt": initial,
            "assembly": assembly, "artifacts": artifacts, "preservation": preservation,
            "live_verification": live_audit, "package_verification": package, "source_materials": sources, "tests": tests}
    write_json(REPORTS / "shared-learner-data.json", data)
    verified = {"passed": True, "source_sha256": source_hash(), "audits": audits,
                "checkpoint_replays": sum(len(r["replay_checks"]) for a in audits for r in a["trials"]),
                "decision_arithmetic_checks": sum(r["decision_arithmetic_checks"] for a in audits for r in a["trials"]),
                "live": live_audit, "preservation": preservation, "package": package, "source_materials": sources, "tests": tests}
    write_json(REPORTS / "shared-learner-verification.json", verified)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    x = np.arange(len(METHODS))
    for kind, offset in (("delta", -.18), ("reference", .18)):
        rows = [next(r for r in aggregate if r["kind"] == kind and r["method"] == m and r["support_count"] == 128) for m in METHODS]
        axes[0, 0].bar(x+offset, [100*r["binding"]["ordinary"]["mean"] for r in rows], .34,
                       yerr=[100*r["binding"]["ordinary"]["sample_std"] for r in rows], color=COLORS[kind], label=kind, capsize=3)
        axes[0, 1].bar(x+offset, [100*r["worst_retention_loss"]["mean"] for r in rows], .34, color=COLORS[kind], label=kind)
        for index, method in enumerate(METHODS):
            individual = [next(r for r in t["runs"] if r["method"] == method and r["support_count"] == 128)
                          for t in trials if t["kind"] == kind]
            axes[0, 1].scatter(x[index]+offset+np.array([-.06, 0, .06]),
                [100*max(0, max(r["retention_audit"]["retention_loss_by_capability"].values())) for r in individual],
                facecolors="white", edgecolors=COLORS[kind], s=26, zorder=3)
    for ax in axes[0]:
        ax.set_xticks(x, METHODS, rotation=20)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].set_title("New first-binding accuracy · 128 support · 192 updates")
    axes[0, 0].set_ylabel("Accuracy (%) · mean ± sample SD, 3 seeds")
    axes[0, 0].legend()
    axes[0, 1].set_title("Largest old-capability loss · dots show each seed")
    axes[0, 1].set_ylim(-3, 105)
    axes[0, 1].set_ylabel("Score loss (percentage points)")
    axes[0, 1].axhline(2, color="#222222", linestyle=":", linewidth=1)
    for ax, kind in zip(axes[1], ("delta", "reference")):
        for method, color in (("full", "#a34140"), ("replay", "#705397"), ("scoped", "#246b83")):
            rows = [next(r for r in aggregate if r["kind"] == kind and r["method"] == method and r["support_count"] == c) for c in (32, 128, 512)]
            for family, style in (("ordinary", "-"), ("composition", "--")):
                ax.plot([32, 128, 512], [100*r["binding"][family]["mean"] for r in rows], marker="o", color=color,
                         linestyle=style, label=f"{method} / {family}")
        ax.set_xscale("log", base=4)
        ax.set_xticks([32, 128, 512], [32, 128, 512])
        ax.set_ylim(0, 103)
        ax.set_xlabel("Distinct support cases · equal 192-update budget")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{kind.capitalize()} support and extra-overwrite transfer")
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc="lower left" if kind == "delta" else "upper right")
    fig.suptitle("SERA shared learning: acquisition, transfer and retention", fontsize=17)
    fig.savefig(REPORTS / "shared-learner-results.png", dpi=170)
    plt.close(fig)
    lines = ["# SERA 0.5 shared-learner study", "",
        "I built and trained one R1 parameter owner for world prediction, typed tasks and sequence inference. I compared the associative core with the handbook-sized reference, repaired numerical failures and destructive adaptation, and tested a persistent corrective update. The original experiments, failures and alternative models remain preserved.", "",
        "The [source audit](shared-source-audit.md) maps the implementation to the packet. The [protocol and amendment](../research/shared-learner-protocol.md) distinguish the first frozen attempt from the repaired follow-up. [Machine-readable evidence](shared-learner-data.json) includes per-seed scores, validation curves, costs, checkpoint identities and preservation records.", "",
        "![Acquisition and retention](shared-learner-results.png)", "", "## What I taught", "",
        "Each core received 1,600 joint updates at batch size 32: 12,800 world-trajectory draws, 12,800 sequence draws and 25,600 typed-example draws. The fixed stream order was world, sequence, typed, typed. World supervision used visible sensor/action/reward data; hidden simulator states were reserved for scoring. Typed support contained 192 examples for each of seven tasks plus 512 explicitly instructed latest-binding examples. Sequence training covered marked retrieval, latest binding, ordered control and majority counting. Motion targets used meters. Arithmetic procedures were selected separately from a supplied 21-candidate grammar.", "",
        "The new rule requested the first value assigned to an entity, despite later different assignments. Instructions explicitly distinguished first from latest. Every query-key overwrite changed the first/latest answer. The six-write training family was tested on six writes, twelve writes, and eight writes with an extra query-key overwrite. No query labels updated weights or chose checkpoints.", "",
        "## Pretraining performance", "", "| Shared core | Parameters | Core bytes | World prediction | Finite goal success | Legacy four-task mean |", "|---|---:|---:|---:|---:|---:|"]
    for kind in ("delta", "reference"):
        rows = [r for r in trials if r["kind"] == kind]
        world = [next(iter(r["baseline"]["tasks"].values())) for r in rows]
        seq = [np.mean([v["accuracy"] for name, v in r["baseline"]["legacy_retention"]["tasks"].items() if name != "earliest_binding"]) for r in rows]
        lines.append(f"| {kind} | {rows[0]['parameters']:,} | {rows[0]['core_state_bytes']:,} | {fmt(metric([w['prediction']['accuracy'] for w in world]))} | {fmt(metric([w['all_pair_control_success'] for w in world]))} | {fmt(metric(seq))} |")
    lines += ["", "Values are percentages, mean ± sample standard deviation across three seeds. World prediction uses masked 12-step episodes in the trained world; goal success covers the twelve finite unequal start/goal pairs. It is not general-world planning accuracy.", "",
              "| Typed task, neural route | Delta ID | Reference ID | Delta composition | Reference composition |", "|---|---:|---:|---:|---:|"]
    for task in trials[0]["baseline"]["typed_neural_only"]["test-id"]["tasks"]:
        scores = [fmt(metric([r["baseline"]["typed_neural_only"][partition]["tasks"][task]["score"] for r in trials if r["kind"] == kind]))
                  for partition in ("test-id", "test-composition") for kind in ("delta", "reference")]
        lines.append(f"| {task} | " + " | ".join(scores) + " |")
    lines += ["", "Motion uses `exp(-MSE)` rather than classification accuracy. Neural arithmetic and procedure-assisted arithmetic are separate measurements; detailed route scores are in the JSON evidence.", "", "## Corrective learning and retention", "",
              "All following controls use 192 updates, batch size 32 and identical support/validation pools within a seed. Replay uses 16 new and 16 retained examples per update; other trained controls use 32 new examples. No-update baselines use no correction. The scoped condition trains only low-rank weight residuals on the supplied first-binding input scope. The separate condition reuses full-update typed weights while preserving a second frozen owner for world/sequence outputs.", "",
              "| Core | Method | Support | Ordinary | Long | Extra overwrite | Worst old loss | Retention passes |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("delta", "reference"):
        selected = [r for r in trials if r["kind"] == kind]
        before = [fmt(metric([r["baseline"]["binding"]["earliest"][family]["accuracy"] for r in selected]))
                  for family in ("ordinary", "long", "composition")]
        lines.append(f"| {kind} | no update | 0 | " + " | ".join(before) + " | 0.00 | 3/3 |")
    for row in aggregate:
        lines.append(f"| {row['kind']} | {row['method']} | {row['support_count']} | {fmt(row['binding']['ordinary'])} | {fmt(row['binding']['long'])} | {fmt(row['binding']['composition'])} | {fmt(row['worst_retention_loss'])} | {row['retention_passes']}/3 |")
    lines += ["", "Worst old loss is the largest loss among 34 individually scored capabilities in each seed, then averaged across seeds. The empirical limit is 2 percentage points. These fixed-suite comparisons are descriptive and do not promote models. Scoped retention outside its applicability domain follows from unchanged computation; its novel-task success must still be learned and evaluated. Three seeds provide limited uncertainty estimates. Equal update/example budgets do not imply equal FLOPs, parameters or CPU time. Replay uses a fixed schedule and novel-only checkpoint criterion; tuning retention-aware selection or loss weights remains untested.", "",
              "| Core | Base/full/replay/scratch parameters | Global adapter total | Scoped total | Separate-model total |",
              "|---|---:|---:|---:|---:|"]
    for kind in ("delta", "reference"):
        base_parameters = next(r["parameters"] for r in trials if r["kind"] == kind)
        counts = {method: next(r["parameters"] for r in aggregate if r["kind"] == kind and r["method"] == method)
                  for method in ("adapter", "scoped", "separate")}
        lines.append(f"| {kind} | {base_parameters:,} | {counts['adapter']:,} (+{counts['adapter']-base_parameters:,}) | {counts['scoped']:,} (+{counts['scoped']-base_parameters:,}) | {counts['separate']:,} |")
    lines += ["", "Adapter ranks and affected weights differ: the global adapter has rank 8 at fusion, while scoped residuals have rank at most 16 across eligible core projections. The separate control retains two complete owners; a pruned expert was not tested. These are implemented-control costs, not capacity-matched architecture claims.", "",
              "## Persistent solver result", ""]
    live = assembly["learning"]
    admission = live["admission"]
    lines += [f"The preselected seed-0 delta solver trained a fresh bounded R2 instrument and finite intervention selector, then attempted 1,024 scoped corrective updates using 128 support cases. Its fresh admission result was **{live['status']}**. The cumulative round was {admission['round_index']}; current version is `{live['current']['version']}`. This longer update is separate from the 192-update table. The binding correction uses an explicitly selected scoped method; the learned controller covers world interventions.", "",
              "| Fresh first-binding test | Before | After | After NLL | After Brier |", "|---|---:|---:|---:|---:|"]
    for family in ("ordinary", "long", "composition"):
        before = admission["incumbent"]["binding"]["earliest"][family]["accuracy"]
        after = admission["candidate"]["binding"]["earliest"][family]["accuracy"]
        scores = admission["candidate"]["binding"]["earliest"][family]
        lines.append(f"| {family} | {100*before:.2f}% | {100*after:.2f}% | {scores['nll']:.4f} | {scores['brier']:.4f} |")
    decision = admission["decision"]
    lines += ["", f"The mean paired gain was {100*decision['mean_gain']:.2f} points and its conservative lower bound was {100*decision['gain_lower_bound']:.2f} points across {decision['samples']:,} independent objective examples. The largest retained-capability loss was {100*max(decision['retention_loss_by_capability'].values()):.4f} points. Latest-binding Brier quality and accuracy were checked separately. Parents, rejected candidates, original evidence and cumulative decisions remain available.", "",
              f"The controller was freshly fitted to eight measured training episodes and three validation episodes, then checked on four reset-family episodes. Its mean test utility was {assembly['controller_test']['mean_utility']:.4f}; fixed replay scored {assembly['controller_test']['fixed_policy_utilities']['replay']:.4f}. Utility includes new-world improvement, old-world loss and the declared operation-cost penalty; it is not accuracy. The learned selector did not beat fixed replay here. It remains a selector trained against a fixed initial solver; this is not evidence of sustained sequential meta-improvement.", "", "## Costs and failures", "",
              "| Invocation group | Summed process CPU seconds | Summed invocation wall seconds | Maximum process peak MiB |", "|---|---:|---:|---:|"]
    groups = [("Initial attempt: three completed delta, three failed reference", [r["costs"] for r in initial["trials"]]),
              ("Repaired six-trial comparison", [r["costs"] for r in trials]), ("R2/controller assembly and ordinary correction", [assembly["costs"]])]
    for name, costs in groups:
        lines.append(f"| {name} | {sum(c['process_cpu_seconds'] for c in costs):.2f} | {sum(c['wall_seconds'] for c in costs):.2f} | {max(c['process_peak_rss_bytes'] for c in costs)/2**20:.2f} |")
    lines += ["", "Seeds ran concurrently with one CPU thread per process. Summed invocation wall times are not elapsed calendar time, and peak RSS is a process high-water mark. Nested correction costs are already included in assembly. Phase counts, validation and evaluation work remain in raw records; heterogeneous operation sums are not FLOPs. Independent verification is separate from these training/assembly totals. Initial failed assembly, the first delta replay audits and some scoped development pilots lack measured elapsed/CPU costs; implementation labor, energy and external costs are also unmeasured. These omissions prevent a claim of complete research cost or research acceleration.", "",
              "The first reference runs stopped on non-finite gradients. The repaired reference uses a documented frozen-projector gradient approximation and clamped mixture gates. Initial full/replay delta updates exposed destructive retention; the scoped method addresses the observed failure with a supplied applicability boundary. Program acquisition also exposed overdepth candidate wrapping, now rejected before execution. Every earlier result and failure remains labeled; repeated delta seeds are not counted as additional independent evidence.", "", "## Verification and remaining research", "",
              f"All {verified['checkpoint_replays']} saved inference conditions were reconstructed and replayed; {verified['decision_arithmetic_checks']} fixed-suite decisions were recalculated. The audit checks exact score vectors, neural/calibration reports, semantic partition separation, checkpoint hashes and shared ownership. The separate live verification repeats promotion scoring and checks restored ordinary inference. Preservation accounts for {preservation['baseline_files']:,} original files, with no missing or unpreserved changes.", "",
              f"All {tests['tests']} local tests passed. The installable 0.5.0 wheel was built in a fresh directory and installed into an isolated target. Its source hash matches the training code, and it restores the shared R1/R2 solver with the same parameter owner. The original archive and handbook hashes were checked again against the supplied files.", "",
              "The shared R1 defect is repaired in the new variants. Broad learned representations, calibrated stochastic planning, general program abstraction, compatible model growth, external evaluator isolation and coupled learner/improver improvement remain open. The [source audit](shared-source-audit.md) states these limits explicitly. Earlier HMM, hybrid, compact-world and typed models remain useful research records; their task protocols differ and should not be pooled into a single model ranking."]
    (REPORTS / "shared-learner-study.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    names = ("shared-learner-data.json", "shared-learner-verification.json", "shared-learner-study.md", "shared-source-audit.md", "shared-learner-results.png", "../research/variants.json", "../research/variants.md")
    write_json(REPORTS / "shared-learner-evidence-manifest.json", [{"file": name, "bytes": (REPORTS/name).stat().st_size,
        "sha256": hashlib.sha256((REPORTS/name).read_bytes()).hexdigest()} for name in names])
    print("Built shared learner report with", verified["checkpoint_replays"], "replayed conditions")


if __name__ == "__main__":
    main()
