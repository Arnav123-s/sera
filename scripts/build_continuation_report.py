"""Build the continuation's compact public reports from preserved local results."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research-continuation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip()+"\n", encoding="utf-8", newline="\n")


def copy_evidence():
    release = WORK / "14_release"
    release.mkdir(exist_ok=True)
    sources = {
        "GG-ACT-001-summary.json": "runs/GG-ACT-001-final/summary.json",
        "GG-ACT-001-manifest.json": "runs/GG-ACT-001-final/manifest.json",
        "GG-ACT-001-replay.json": "runs/GG-ACT-001-final/replay-report.json",
        "GG-ACT-001-resources.json": "runs/GG-ACT-001-final-supervisor/state.json",
        "GG-ACT-001-replay-resources.json": "runs/GG-ACT-001-replay-supervisor/state.json",
        "SHARED-GG-001-result.json": "runs/SHARED-GG-001/result.json",
        "SHARED-GG-001-resources.json": "runs/SHARED-GG-001-supervisor/state.json",
        "current-tests.xml": "runs/continuation-full-tests/pytest.xml",
        "current-tests-resources.json": "runs/continuation-full-tests/state.json",
        "GG-P0-001-replay.json": "runs/generative-memory-GG-P0-001-verification/verification.json",
        "GG-P0-001-refit.json": "runs/generative-memory-GG-P0-001-refit-resource-v2/verification.json",
        "GG-P0-001-refit-resources.json": "runs/GG-P0-resource-supervisor-v2/state.json",
        "R7-replay.json": "runs/parameter-cloud-verification/verification.json",
        "pinned-SERA-tests.xml": "research-continuation/12_reproductions/sera-pinned/pytest.xml",
        "pinned-SERA-plumbing-resources.json": "research-continuation/12_reproductions/sera-pinned/plumbing-supervisor/state.json",
    }
    for target, source in sources.items():
        shutil.copyfile(ROOT / source, release / target)
    failures = WORK / "09_failures/shared-generative"
    failures.mkdir(exist_ok=True)
    for attempt in (1, 2, 3, 4):
        for name in ("process.log", "state.json", "pytest.xml"):
            path = ROOT / f"runs/shared-generative-tests-attempt{attempt}/{name}"
            if path.exists():
                shutil.copyfile(path, failures / f"attempt{attempt}-{name}")
    members = {}
    first = ROOT / "runs/generative-memory-GG-P0-001/source"
    for path in sorted(first.glob("*.py")):
        members[f"GG-P0-001/experiments/generative_memory/{path.name}"] = path.read_bytes()
    members["GG-P0-001/protocol.json"] = (ROOT / "runs/generative-memory-GG-P0-001/protocol.json").read_bytes()
    # The ACT cohort pins its decoder and common interfaces, tests, and supervisor.
    # Include the whole current SERA source closure for import completeness.
    for name in ("GG-ACT-001", "SHARED-GG-001"):
        protocol = read(WORK / f"04_protocols/{name}.json")
        expected = {**protocol["source_hashes"], **protocol.get("frozen_input_hashes", {})}
        for relative, checksum in expected.items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == checksum
        paths = set(expected) | {p.relative_to(ROOT).as_posix() for p in (ROOT/"src/sera").glob("*.py")}
        paths |= {"experiments/generative_memory/__init__.py", "tests/conftest.py"}
        for relative in sorted(paths):
            members[f"{name}/{relative}"] = (ROOT / relative).read_bytes()
        members[f"{name}/research-continuation/04_protocols/{name}.json"] = (WORK / f"04_protocols/{name}.json").read_bytes()
    # Fixed ZIP timestamps make rebuilds byte-stable.
    with zipfile.ZipFile(release / "frozen-protocol-sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    write(release / "frozen-source-members.json", json.dumps(
        {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
         for name, raw in sorted(members.items())}, indent=2))


def acquisition_report():
    summary = read(ROOT / "runs/GG-ACT-001-final/summary.json")
    folder = WORK / "08_analysis/GG-ACT-001"
    lines = ["# GG-ACT-001: fixed active inquiry", "",
        "I completed 300 policy runs across 60 paired latent worlds: 12 frozen instances in each of five families. "
        "Every run paid for 10 observed pairs, including the same two initial observations. "
        "The learner received a supplied phase coordinate, known noise, a finite action grid and four generator classes. "
        "It learned coefficients and class weights by Bayesian updating. The five inquiry policies were fixed; no investigator was trained.", "",
        "The grid offered 15 target measurements and 15 known zero-design nuisance measurements. "
        "Nuisance noise had standard deviation 20, versus 0.05 on target measurements. "
        "The reference space-filling policy used the supplied channel metadata. This tests raw entropy against epistemic usefulness "
        "under known noise, not discovery of which sensor is informative.", "",
        "## External prediction after ten paid observations", "",
        "Clean MSE per coordinate, averaged over 12 worlds and 41 independent query positions on [-1.8, 1.8]. "
        "Lower is better. Each policy sees the same potential observations when it chooses the same action.", "",
        "| Family | Random | Space filling | Raw entropy | Disagreement | Information gain |",
        "|---|---:|---:|---:|---:|---:|"]
    families = ("circle", "ellipse", "line", "radial_orbit", "menu_exception")
    policies = ("random", "space_filling", "predictive_entropy", "disagreement", "information_gain")
    values = {(r["family"], r["policy"]): r for r in summary["aggregates"]}
    for family in families:
        lines.append("| "+family+" | "+" | ".join(f"{values[family,p]['clean_mse']:.6g}" for p in policies)+" |")
    lines += ["", "Information gain reduced mean MSE relative to random sampling on all five families. "
        "Its exploratory paired bootstrap intervals excluded zero on the four in-menu families, but included zero on the omitted-pattern family. "
        "All four in-menu information-gain versus space-filling intervals included zero, and space filling had lower mean error in each. "
        "On omitted patterns, information gain reduced MSE versus space filling, while uncertainty remained badly miscalibrated. "
        "I do not promote an overall information-gain advantage.", "",
        "| Family | IG minus space-filling MSE | Exploratory 95% interval |", "|---|---:|---:|"]
    for row in summary["paired_comparisons"]:
        if row["control"] == "space_filling" and row["metric"] == "clean_mse":
            low, high = row["bootstrap_95_percentile_interval"]
            lines.append(f"| {row['family']} | {row['mean_difference']:.6g} | [{low:.6g}, {high:.6g}] |")
    lines += ["", "The bootstrap resamples paired worlds with 1,000 frozen-seed draws. These are small exploratory comparisons without multiplicity correction.",
        "", "## Confidence and model inadequacy", "",
        "| Family, information gain | Joint query NLL | Marginal 95% coverage | Mean best-class weight | Any inadequacy alarm |",
        "|---|---:|---:|---:|---:|"]
    for family in families:
        row = values[family, "information_gain"]
        lines.append(f"| {family} | {row['observed_joint_nll']:.5g} | {row['marginal_95_coverage']:.1%} | "
                     f"{row['final_max_class_weight']:.3%} | {row['alarm_any']:.1%} |")
    lines += ["", "The omitted family is a circle with a localized oscillatory exception absent from the menu. "
        "Information gain ended with 99.738% mean best-class weight but only 51.3% marginal coverage. "
        "The prequential alarm fired in 9 of 12 runs and missed 3. Confidence among supplied alternatives is not confidence that an adequate alternative exists. "
        "The alarm threshold is uncorrected across adaptive repeated tests and is not an applicability guarantee.", "",
        "Raw predictive entropy chose nuisance measurements on all eight discretionary steps in every run. "
        "Random sampling used 3.083 nuisance observations on average; the other policies used zero. "
        "High entropy alone rewarded irreducible noise. Broad intervals from the raw-entropy policy often covered values despite poor point predictions.", "",
        "## Verification and cost", "",
        "Every action, full candidate-score vector, hypothetical branch, observation, posterior, external prediction, joint likelihood, marginal interval vector and summary replayed exactly. "
        "A separate implementation checked all 300 expected policy records, all 3,000 observations and 12,000 batch Gaussian posteriors. "
        "Maximum independent mean difference was 1.14e-12 and maximum log-class-weight difference was 3.22e-9.", "",
        "Primary process time was 98.379 seconds wall and 95.109 seconds CPU; supervised wall time was 101.149 seconds, with 257,298,432 peak committed job bytes. "
        "Exact replay used 62.346 supervised seconds. These local measurements include scientific evaluation and writing; they are not a speed comparison against another implementation. "
        "The primary manifest covers 184,009,917 bytes of raw audit artifacts. Decoder source occupies 70,262 bytes; final retained side state is about 15.7–16.0 KB per run. "
        "Python, Torch, NumPy and SciPy installations are additional shared runtime costs. No complete packaged compression claim is made.", "",
        "Order-nine bivariate Gauss-Hermite quadrature approximates mixture entropy/EIG. Information gain used 98,496 density evaluations per run; raw entropy used 134,784. "
        "Space filling is substantially cheaper here. Selection effort and observations must both count.", "",
        "This panel keeps the analytic posterior as explicit side state. Its owner adapter is contract-tested, but the full fixed-policy panel does not train SERA's neural owner. "
        "The separate [trained-parent integration](../../05_experiments/shared-generative-integration.md) places a matching four-class generator in registered R1 buffers.", "",
        "[Frozen protocol](../../04_protocols/GG-ACT-001.json) · [Full compact statistics](../../14_release/GG-ACT-001-summary.json) · "
        "[Exact replay](../../14_release/GG-ACT-001-replay.json) · [Independent audit](../../01_audit/GG-ACT-001-independent.json)"]
    write(folder / "report.md", "\n".join(lines))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summary = read(ROOT / "runs/GG-ACT-001-final/summary.json")
    values = {(r["family"], r["policy"]): r for r in summary["aggregates"]}
    families = ("circle", "ellipse", "line", "radial_orbit", "menu_exception")
    labels = ("Circle", "Ellipse", "Line", "Growing orbit", "Omitted pattern")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    positions = np.arange(5)
    for offset, policy, label, color in ((-.23, "random", "Random", "#758497"),
                                       (0, "space_filling", "Space filling", "#235ab8"),
                                       (.23, "information_gain", "Information gain", "#15866a")):
        axes[0].bar(positions+offset, [values[f,policy]["clean_mse"] for f in families],
                    width=.22, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Clean query MSE per coordinate (log scale)")
    axes[0].set_title("Fixed inquiry: external prediction")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(positions, [100*values[f,"information_gain"]["marginal_95_coverage"] for f in families],
                color=["#15866a"]*4+["#c85045"], width=.6)
    axes[1].axhline(95, color="#40444a", linestyle="--", linewidth=1.2, label="Nominal 95%")
    axes[1].set_ylim(0,105)
    axes[1].set_ylabel("Observed marginal interval coverage (%)")
    axes[1].set_title("Information gain: uncertainty failure")
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("SERA · 12 paired worlds per family · 10 paid observations per policy", fontsize=13)
    folder = WORK / "08_analysis/GG-ACT-001"
    fig.savefig(folder / "inquiry-results.png", dpi=170)
    fig.savefig(folder / "inquiry-results.svg")
    plt.close(fig)


def main():
    copy_evidence()
    acquisition_report()
    figures()
    print("Built compact evidence, frozen source archive, inquiry report and figures.")


if __name__ == "__main__":
    main()
