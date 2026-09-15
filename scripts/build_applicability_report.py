"""Package GG-GUARD-001 evidence and render its results without altering the cohort."""

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/15_applicability"
RUN = ROOT/"runs/GG-GUARD-001-final"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def main():
    summary = read(RUN/"summary.json")
    if summary["gate"]["component_promotion"]:
        raise ValueError("This report describes the recorded rejected candidate")
    for name in ("guard-model.json", "thresholds.json", "before-final.json", "records.json.gz", "summary.json"):
        shutil.copyfile(RUN/name, RELEASE/name)
    for source, name in (
        ("runs/GG-GUARD-001-replay/replay.json", "replay.json"),
        ("runs/GG-GUARD-001-frozen-replay/replay.json", "frozen-workspace-replay.json"),
        ("runs/GG-GUARD-001-audit-v3/independent.json", "independent.json"),
    ):
        shutil.copyfile(ROOT/source, RELEASE/name)
    jobs = ("final", "replay", "audit", "audit-v2", "audit-v3")
    for job in jobs:
        directory = ROOT/f"runs/GG-GUARD-001-{job}-supervisor"
        for filename in ("state.json", "process.log"):
            target = RELEASE/("failures" if job in {"audit", "audit-v2"} else "supervision")
            target.mkdir(exist_ok=True)
            shutil.copyfile(directory/filename, target/f"{job}-{filename}")
    for filename in ("state.json", "process.log"):
        shutil.copyfile(ROOT/"runs/GG-GUARD-001-frozen-replay/supervisor"/filename,
                        RELEASE/"supervision"/f"frozen-replay-{filename}")
    protocol = read(RELEASE/"protocol.json")
    auditor_name = "experiments/generative_memory/applicability_audit.py"
    repair = {
        "path": auditor_name, "original_sha256": protocol["payload"]["sources"][auditor_name]["sha256"],
        "corrected_sha256": hashlib.sha256((ROOT/auditor_name).read_bytes()).hexdigest(),
        "intermediate_sha256": hashlib.sha256((RELEASE/"failures/audit-v2.py").read_bytes()).hexdigest(),
        "attempts": [
            {"revision": "v1", "result": "FAILED", "reason": "Checker expected synthetic instead of the serialized simulator_ground_truth evidence kind; stopped on first event."},
            {"revision": "v2", "result": "FAILED", "reason": "Independent predictions/features matched, but coefficient comparisons used axis-grouped rather than interleaved storage for ellipse/line."},
            {"revision": "v3", "result": "PASS", "reason": "Correct evidence enum and coefficient layout; seven new regression cases. The 2e-8 algebra tolerance is unchanged."}],
        "scope": "Auditor repair after final collection. Model, learner, feature extractor, thresholds, evaluator results and original protocol remain byte-identical. Original auditor is in frozen-sources.zip; v2 is in failures/audit-v2.py.",
    }
    write(RELEASE/"audit-repair.json", repair)
    families = list(summary["by_family"])
    labels = [f.replace("_", " ") for f in families]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), layout="constrained")
    y = np.arange(len(families))
    for method, offset, color, label in (
        ("always_predict", -.23, "#a6a9ad", "Always predict"),
        ("residual_uncertainty", 0, "#db8a3c", "Residual / uncertainty guard"),
        ("learned", .23, "#2476a8", "Learned guard"),
    ):
        for ax, metric in zip(axes, ("coverage", "selective_risk")):
            values = [summary["by_family"][f][method][metric] for f in families]
            ax.barh(y+offset, [100*v if v is not None else np.nan for v in values],
                    height=.22, color=color, label=label)
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Accepted queries (%)")
    axes[1].set_xlabel("Wrong among accepted queries (%)")
    axes[1].axvline(10, color="#aa3030", linestyle="--", linewidth=1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("SERA applicability: useful selection, failed conditional-risk gate\n"
                 "48 withheld worlds per family · 33 queries each · frozen calibration thresholds", fontsize=13)
    fig.savefig(RELEASE/"results.png", dpi=180)
    fig.savefig(RELEASE/"results.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 5.5), layout="constrained")
    for family in ("circle", "local_exception", "random_values", "piecewise_drift", "chirp"):
        rows = [r for r in summary["risk_coverage"][family] if r["selective_risk"] is not None]
        ax.plot([100*r["coverage"] for r in rows], [100*r["selective_risk"] for r in rows],
                marker=".", markersize=3, label=family.replace("_", " "))
    ax.axhline(10, color="#aa3030", linestyle="--", linewidth=1, label="10% risk reference")
    ax.set(xlabel="Accepted queries (%)", ylabel="Wrong among accepted queries (%)",
           title="Risk / acceptance curves on the frozen final set\nDescriptive curves; thresholds were not selected from this set",
           xlim=(0, 102), ylim=(0, 102))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(RELEASE/"risk-coverage.png", dpi=180)
    fig.savefig(RELEASE/"risk-coverage.svg")
    plt.close(fig)
    table = []
    for family, row in summary["by_family"].items():
        a, b = row["learned"], row["residual_uncertainty"]
        table.append(f"| {family.replace('_', ' ')} | {a['coverage']:.1%} | {a['selective_risk']:.1%} | {a['accepted_clean_mse']:.6g} | {b['coverage']:.1%} | {b['selective_risk']:.1%} |")
    paired = summary["paired_utility"]["residual_uncertainty"]
    ci = paired["world_stratified_bootstrap_95"]
    model = read(RELEASE/"guard-model.json")
    selection = read(RELEASE/"thresholds.json")["selections"]["learned"]["selected"]
    doc = f"""# GG-GUARD-001: learned applicability results

I trained and audited a learned validity guard on **1,344 distinct worlds**. It improved the declared selection score over the fixed controls and preserved all in-menu predictions, but **failed the preregistered conditional-risk gate**. I retain its model and complete evidence as an experimental candidate. It is not admitted into the operational shared learner.

## What I taught it

The prediction engine stayed the four-class supplied-phase Gaussian generator: circle, ellipse, line and growing orbit. A separate 16-input, 16-hidden-unit tanh network learned the probability that a proposed prediction would lie within Euclidean distance 0.2 of an independently observed noisy outcome. Observation noise was 0.05 standard deviation per coordinate. This label concerns a measured answer at a declared tolerance, not proof that the underlying mechanism is correct.

The guard received public query coordinates, distance to actual observations, predictive and coefficient uncertainty, class disagreement, and prequential residual summaries. It received no mechanism names, environment seeds, coefficients, clean targets or future answers as features. Clean mechanism values were used only for the secondary MSE measurement. I checked that changing hidden query answers could not change the guard's features.

There were **576 teaching worlds**, **192 probability-calibration worlds**, **192 threshold-selection worlds**, and **384 final worlds**. Every world contributed ten prefix observations and 33 separately observed query outcomes. All of a world's queries stayed in its partition. Teaching included four in-menu families, local exceptions and random-value negatives. Piecewise drift and chirp mechanisms were completely excluded from fitting and both calibration banks; final evaluation used 48 fresh worlds from each of eight families.

The model has **291 learned scalars**, including two probability-calibration parameters, plus 32 feature-normalization scalars. It used 500 full-batch Adam updates and 250 separate calibration updates. Training BCE decreased from **0.75165 to 0.11235**; this is training fit, not a generalization score. One optimizer seed was frozen independently of all environment seeds; robustness across optimizer initializations is untested.

## Final performance at the frozen operating point

Acceptance is the fraction of queries answered. Risk is the fraction of those answers that miss the observed-outcome tolerance. MSE is mean squared coordinate error against the hidden clean mechanism. Each table row contains 1,584 final queries clustered within 48 worlds.

| Family | Learned acceptance | Learned risk | Accepted clean MSE | Fixed residual acceptance | Fixed residual risk |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

![Acceptance and error among accepted predictions](results.png)

The declared utility was `correct accepted - 4 × wrong accepted`, averaged per world. The learned guard improved it by **{paired['mean_difference']:.4f}** over the residual/uncertainty guard; the paired, family-stratified world-bootstrap 95% interval was **[{ci[0]:.4f}, {ci[1]:.4f}]**. This is uncertainty over instances in the eight specified families, not over all possible mechanisms.

The distance-only control found no nonempty acceptance set meeting its calibration risk target, so it rejected every final query. Its zero acceptance is explicitly reported; an improvement over it is weaker evidence than the comparison with the residual/uncertainty control. Always predict and always abstain are also retained in [all metrics](summary.json).

## Why the guard failed

Threshold selection met its pooled calibration target: **{selection['selective_risk']:.2%}** observed risk at **{selection['coverage']:.2%}** acceptance. But many safe in-menu predictions dominated that average. The selected minimum predicted-validity probability was only **{selection['threshold']:.6f}**. It admitted too many poor predictions from minority mechanisms. A pooled error target did not establish a valid boundary for each mechanism.

All four in-menu families retained 100% acceptance with 0.06–0.25% risk. Local exceptions nevertheless had **46.3%** risk, and protected piecewise drift had **65.8%** risk. The learned guard rejected most random/chirp queries, but its few accepted answers there were overwhelmingly wrong. Positive utility relative to a weaker control does not make those answers supported knowledge.

The frozen descriptive curve at a 0.9 probability threshold gives 12.0% acceptance / 10.5% risk on local exceptions and 9.2% / 13.7% on piecewise drift, while rejecting all random/chirp queries. These are post-evaluation diagnostics of an already declared curve, not a newly validated operating policy. Even that stricter point does not establish the requested applicability guarantee.

![Risk and acceptance across probability thresholds](risk-coverage.png)

The observational-alias diagnostic is more fundamental: two worlds share the exact ten observations, but a compact bump makes their unobserved answers differ by `(2, -2)`. The guard assigns both **98.48%** validity probability and accepts both. No method can identify the missing exception from that unchanged prefix alone. Further evidence or an explicit conditional answer is required; this is not a numerical defect to tune away.

## Audit, preservation and cost

- Exact replay checked **44,352 query feature/prediction rows**, all 1,344 posterior histories, final guard probabilities, calibration thresholds and final summaries. Refitting reproduced the entire model artifact exactly. A second replay from a fresh frozen-source workspace also passed.
- Independent observation-space Gaussian algebra checked **37,632 component posteriors**, including prequential prefixes. The largest feature/prediction difference was about **2.1e-10**, below the unchanged 2e-8 tolerance. NumPy evaluation of the learned network differed by at most **2.22e-16**. Labels, decision counts, partitions and fixed-control scores agree.
- I corrected two checker defects after retaining their failed attempts: the serialized evidence enum and ellipse/line coefficient ordering. Both checker versions and the original source are preserved. No learner, data, model, threshold, result or tolerance was changed. [Repair record](audit-repair.json).
- **174 tests pass**, including seven checker regressions. Ruff and both historical release verifiers pass. Existing shared-owner source files are byte-identical, and **16,556 predecessor files / 511,596,737 bytes** remain unchanged. The earlier 40-capability retention study remains its own historical result; this component made no new shared-owner update or transfer claim.
- The final job took **55.93 supervised seconds**, with **586,358,784 peak job committed bytes** (not RSS). Worker analysis/fitting/scoring took 50.50 seconds; guard fitting and probability calibration took 2.64 seconds. Process startup, final serialization and shutdown account for additional job time. Separate replay, repaired audits and their failed attempts are charged in [supervision records](supervision/).
- The cohort used **13,440 support observations**, **31,680 teaching/calibration outcomes**, and **12,672 final assessment outcomes**. Controls received identical prefix observations. Fixed controls shared threshold-selection exposure; the learned guard's extra teaching/probability-calibration exposure remains an additional cost.
- The complete raw cohort is **11,102,953 compressed bytes**. The model file is **71,957 bytes**, including provenance and witness identities. Most of that model file is audit metadata rather than weights. The decoder, full factual witnesses and frozen-source archive are additional storage; a parameter count alone is not the full memory cost.

Model SHA-256: `{model['sha256']}`. Protocol SHA-256: `{protocol['sha256']}`. All raw observations, clean assessor values, exact features, per-world metrics and learner weights are published in this directory. Historical large cohorts remain locally indexed rather than duplicated.

## Architecture decision and next experiment

This completes a **component experiment for W08**, not W08 as a whole. The packet requires corrected applicability plus common-owner transfer and retention. The reliability gate failed, so I preserve the learned guard as an explicit experimental candidate and keep it out of supported-answer execution. See the [source-packet comparison](architecture-audit.md).

The next prospective study will test evidence-conditioned calibration with a minimum validity probability, then paid additional observations when existing evidence is insufficient. It must use new instances and newly protected mechanisms: the two families tested here are no longer unseen research material. It must keep each control's observation costs matched and cannot turn imagined answers into factual labels. M2 remains open; M3–M4 are not established.

## Reproduce

```powershell
.venv/Scripts/python.exe scripts/verify_applicability.py
.venv/Scripts/python.exe scripts/replay_guard_release.py --output runs/my-guard-replay
.venv/Scripts/python.exe -m experiments.generative_memory.applicability_audit --run research-continuation/15_applicability --output runs/my-independent-guard-audit.json
```

Use a fresh output path. Exact retraining is checked under the frozen Python 3.12.14 / Torch 2.10.0+cpu / NumPy 2.5.3 runtime. The replay helper verifies and extracts the original source; the current independent auditor is the documented corrected revision. Different runtimes are not silently accepted as byte-identical reproduction.
"""
    (RELEASE/"report.md").write_text(doc, encoding="utf-8", newline="\n")
    print("Published report, complete compact cohort, plots, supervision and repair evidence.")


if __name__ == "__main__":
    main()
