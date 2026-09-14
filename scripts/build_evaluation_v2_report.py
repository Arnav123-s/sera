"""Publish compact evidence and tables for preserved variants and repaired gates."""

import copy
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sera.storage import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def average(values, percent=True):
    values = np.asarray(values) * (100 if percent else 1)
    places = 2 if percent else 4
    return f"{values.mean():.{places}f} ± {values.std(ddof=1):.{places}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    source = read("runs/evaluation-v2-complete/summary.json")
    verified = read("runs/evaluation-v2-review/verification.json")
    live = read("runs/evaluation-v2-review/live-continuation.json")
    preservation = read("runs/evaluation-v2-review/preservation-verification.json")
    data = copy.deepcopy(source)
    data["costs"]["scope"] = (
        "Whole timed study including generation, acquisition, updates, validation, search, evaluation and recorded failures. "
        "RSS is the process high-water mark; phase times may overlap if nested; counts are not FLOPs. "
        "Implementation effort, previous development, separate verification, energy and external services are outside this measurement."
    )
    for trial in data["typed"]:
        history = trial["training"]
        trial["training"] = {k: history[k] for k in ("steps", "examples", "validation_best_macro")}
        trial["training"]["support_record_ids_sha256"] = digest(history["support_ids"])
        trial["program_induction"] = {task: {"accepted": r["accepted"], "rule": r.get("record", {}).get("rule"),
                                               "identity": r.get("record", {}).get("identity")}
                                       for task, r in trial["program_induction"].items()}
    for row in data["retention"]:
        row.pop("incumbent")
        row.pop("candidate_report")
    data["artifact_root"] = "runs/evaluation-v2-complete"
    data["artifact_storage"] = "Training examples, score vectors, checkpoints and full trial reports are preserved locally at the hashed paths. Published JSON is a compact summary; these local artifacts are not implied to be hosted by GitHub."
    data["partition_interpretation"] = "ID, extent and composition holdout refer to the v2 development distribution. Frozen v1 saw some composition templates (mixed units and overwritten bindings) during its old training, but every concrete old teaching case is excluded from this comparison."
    data["verification"] = verified
    data["preservation"] = preservation
    data["live_continuation"] = {k: live[k] for k in ("source_sha256", "parent_unchanged", "parent", "current", "wall_seconds")}
    data["live_continuation"]["result"] = {k: live["result"][k] for k in ("status", "round_index", "proposal", "decision", "work", "dataset_id")}
    write_json(ROOT / "reports/evaluation-v2-data.json", data)
    write_json(ROOT / "reports/evaluation-v2-verification.json", verified)
    trials = data["typed"]

    def values(route, partition, task, metric="score"):
        return [r["results"][route][partition]["tasks"][task][metric] for r in trials]

    tasks = [t for t in trials[0]["results"]["typed-v1-neural"]["test-composition"]["tasks"] if t != "motion"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [3.1, 1.2]})
    x = np.arange(len(tasks))
    for shift, route, label, color in ((-.18, "typed-v1-neural", "Preserved v1", "#3b627e"),
                                       (.18, "typed-v2-neural", "New v2", "#ba6b37")):
        matrix = np.array([values(route, "test-composition", t) for t in tasks]) * 100
        axes[0].bar(x + shift, matrix.mean(1), .34, yerr=matrix.std(1, ddof=1), capsize=3, label=label, color=color)
    axes[0].set_xticks(x, ["Modular\nsum", "Spatial", "Byte\nsum", "Patch", "Tone", "Binding"])
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Composition cases (held out from v2 training)")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    for shift, route, color in ((-.18, "typed-v1-neural", "#3b627e"), (.18, "typed-v2-neural", "#ba6b37")):
        matrix = np.array([values(route, p, "motion", "loss") for p in ("test-id", "test-extent", "test-composition")])
        axes[1].bar(np.arange(3) + shift, matrix.mean(1), .34, yerr=matrix.std(1, ddof=1), capsize=3, color=color)
    axes[1].set_xticks(np.arange(3), ["ID", "Extent", "Compose"])
    axes[1].set_title("Motion error (lower is better)")
    axes[1].set_ylabel("Mean squared meters")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("SERA: preserve alternatives, compare on repaired tests", fontsize=15)
    fig.text(.04, .01, "Three seeds; error bars are sample SD, not confidence intervals. Arithmetic rules score 100% in both routes. Different training data; same neural architecture.", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(ROOT / "reports/evaluation-v2-results.png", dpi=160)
    plt.close(fig)

    lines = ["# SERA 0.4: preserved variants and corrected evaluation", "",
             "I preserved the promising alternative directions and completed the first repair from the source-packet audit: semantic evaluation partitions and separate retention checks. The new results support keeping several variants. They do not establish that the newest model is the best overall learner.", "",
             "The [variant registry](../research/variants.md) names 22 architectures, components and inference routes without creating duplicate experiments. The [data](evaluation-v2-data.json) and [independent verification](evaluation-v2-verification.json) retain numerical details. Old results remain under their original source identities.", "",
             f"Current executable source: `{data['environment']['source_sha256']}`. Frozen 0.3 source: `{data['frozen_source']}`. Values below are means ± sample standard deviations across three seeds unless a count is shown. SD describes variation across these seeds; it is not a confidence interval.", "",
             "![Repaired composition tests and motion errors](evaluation-v2-results.png)", "",
             "## What remains worth keeping", "",
             table(["Preserved variant", "Measured result", "Interpretation"], [
                 ["Sequence hybrid", "82.21 ± 1.00% ID; 77.16 ± 1.91% long", "Useful early alternative; 11,208 parameters versus delta's 3,225. Different task suite from world/typed studies."],
                 ["Sequence delta", "78.11 ± 2.61% ID; 72.95 ± 5.57% long", "Smaller baseline for the same early sequence comparison."],
                 ["Delta World", "86.81 ± 1.25% long masked prediction", "Falls to 78.61% with memory reset. Identifying sensors and deterministic worlds constrain the claim."],
                 ["HMM-22", "100% repeated-pattern prediction", "Strongest conventional control on the bounded aliased-event test; keep it for future belief/planning work."],
                 ["Complex instrument", "88.17 ± 1.29% on the same repeated-pattern test", "A valid controlled-instrument result, with HMM providing a stronger control."],
                 ["Program reuse", "22/22 with either learned guide and library; 21/22 without library", "Finite whole transformations under a token budget. Macro expansions can execute more primitive actions."],
                 ["Full reference state", "38.67 ± 3.04% rank-4/all-routing prediction", "Keep the architecture and negative result. Its training budget differs from Delta World."],
                 ["Intervention selector", "No sustained advantage over the best fixed procedure", "Preserved policies and measured intervention episodes; improvement quality remains open."]]), "",
             "These are separate benchmarks and mechanisms, not one aggregate score. The original full results remain in the [0.3 study](stage-three-study.md) and the earlier [research report](first-study.md).", "",
             "## Repaired teaching and test protocol", "",
             "The v1 generator remains available for exact historical reproduction. `typed-semantic-v2` gives development and test cases semantic identities that ignore provenance, split names, decimal whitespace, equivalent units and masked payload. Identity uses the input alone, so contradictory labels cannot evade an overlap check. Finite cases are sampled without replacement; requesting more than the available domain fails instead of silently repeating cases.", "",
             "Each new model receives 192 support and 48 validation examples for each of seven tasks, then 1,400 updates of 64 examples. Five learned adapters feed the same 26,219-parameter typed architecture. Checkpoint selection uses validation only. Each test partition has 128 cases per task. The 4,368 semantic cases within each seed are distinct across all five partitions, and none overlaps that frozen model's old support/validation cases. Training, validation and ID-test bucket ownership is independent of random seed.", "",
             table(["Task", "Development / ID", "Extent test", "Withheld composition"], [[task, *spec] for task, spec in trials[0]["manifest"]["templates"].items()]), "",
             "Motion predicts one time unit after the final observed timestamp. Composition tests retain the task's supplied rule while withholding an input pattern or combination. They do not demonstrate unseen rule invention, language understanding or general multimodal reasoning. ID, extent and held-out labels refer to the **v2 development distribution**. Frozen v1 already saw some composition templates, including mixed units and overwritten bindings, in its old training; its concrete teaching cases are excluded here. Its results measure retained capability on unseen cases, not unseen-template acquisition. The models receive identical tests but have different training distributions and initializations; changes are not attributable to architecture.", "",
             "## Typed performance on identical new tests", ""]
    for partition, title in (("test-id", "V2 ID cases"), ("test-extent", "V2 extent cases"), ("test-composition", "V2-held-out compositions")):
        lines += [f"### {title}", "", table(["Task", "Preserved v1 neural %", "New v2 neural %", "Support-majority baseline %"],
            [[task, average(values("typed-v1-neural", partition, task)), average(values("typed-v2-neural", partition, task)),
              average([r["baselines"][partition][task]["accuracy"] for r in trials])] for task in tasks]), ""]
    lines += ["The two selected arithmetic procedures score **100%** for modular sum and byte sum in every new partition, for both v1 and v2. The supplied 21-candidate grammar and decimal parser account for this capability. The neural-only arithmetic results remain close to chance.", "",
              table(["Motion split", "Preserved v1 MSE (m²)", "New v2 MSE (m²)", "Last-position baseline MSE (m²)"],
                    [[p, average(values("typed-v1-neural", p, "motion", "loss"), False), average(values("typed-v2-neural", p, "motion", "loss"), False),
                      average([r["baselines"][p]["motion"]["last_position_mse"] for r in trials], False)] for p in ("test-id", "test-extent", "test-composition")]), "",
              "The new cohort improves mean tone composition accuracy from **81.51% to 92.71%**, but binding falls from **47.92% to 35.16%** and extent motion error increases. Patch accuracy remains high. The seed variation is substantial for binding and spatial transfer. I retain both cohorts and have not promoted the v2 typed model into the current solver. Weak binding and neural arithmetic remain explicit research failures; I did not tune repeatedly against these test results.", "",
              "## Separate capability retention", "",
              "The gain objective still gives equal weight to world prediction and control. Admission now checks each world's prediction accuracy and control success separately, plus five legacy outputs and seven typed tasks across three partitions when the typed component is installed. An omitted capability or mismatched paired dataset fails closed. Correlated retention checks do not increase the sample count in the gain bound.", "",
              "The cap is 0.02 absolute score loss per capability. It is empirical, not a confidence guarantee. Motion uses exp(-MSE), so its cap is not a fixed bound in squared meters. Probability calibration, improver quality and every internal component's behavior are not independently gated here. The gain bound retains its fresh conditional sampling assumptions.", "",
              "The synthetic audit counterexample now fails: prediction 0.90 → 0.60 and control 0.40 → 0.90 produce a higher composite score, but the prediction regression blocks admission. This is a constructed contract test, not an observed released promotion.", "",
              "I reexecuted all **15** saved proposals under the old world objective and reproduced every old decision. Retrospective v2 reassessment leaves **3 passing and 12 rejected**, including additional evaluation operations in the cost check. The saved historical decisions and pointers were not changed. These reused historical proposals are not fresh admissions or additional independent trials.", "",
              table(["Seed / round", "Original", "V2", "Gain / lower bound (pp)", "Largest capability loss (pp)", "Failed critical checks"],
                    [[f"{r['seed']} / {r['round']}", r["original_status"], "pass" if r["decision_v2"]["admitted"] else "reject",
                      f"{100*r['decision_v2']['mean_gain']:.2f} / {100*r['decision_v2']['gain_lower_bound']:.2f}",
                      f"{100*max(r['decision_v2']['retention_loss_by_capability'].values()):.2f}",
                      ", ".join(r["decision_v2"]["failed_capabilities"]) or "None"] for r in data["retention"]]), "",
              "The three retained promotions lose at most **1.29 percentage points** on any measured capability, below the two-point cap. Negative values in the table indicate improvement, not a loss.", "",
              f"I also forked the existing continuation workspace once, preserving its full ledger and version history, and ran ordinary `sera learn` on world `permutation-880003`. This fresh attempt selected `{live['result']['proposal']['method']}`, was **{live['result']['status']}** at cumulative round **{live['result']['round_index']}**, and evaluated **{len(live['result']['decision']['retention_loss_by_capability'])}** separate capabilities. The current continuation is `runs/sera-0.4-current`; `runs/sera-0.3-current` remains byte-identical. The standalone v2 typed model was not installed in that continuation.", "",
              "## Verification, preservation and cost", "",
              f"Independent verification checked **{verified['artifact_hashes_checked']} artifact hashes**, regenerated all **15 semantic partitions**, reloaded and rescored **36 variant/partition combinations** (252 task scores), and found **zero score difference**. It independently checked all 15 decision calculations and reexecuted a complete paired capability decision for each seed. The new regression tests and existing suite pass: **59 tests**.", "",
              f"The preservation baseline covers **{preservation['baseline_files']:,} pre-existing files**: **{preservation['unchanged_files']:,} unchanged**, and **{len(preservation['revised_with_exact_original_in_git'])} revised tracked files** with their exact original bytes recoverable at `{preservation['original_commit']}`. Missing files: **0**. Unpreserved changes: **0**. Existing snapshots, unsuccessful experiments, wheels and source documents remain in place. New results use new output paths.", "",
              f"The three-seed study consumed **{data['costs']['wall_seconds']:.2f} seconds wall time**, **{data['costs']['process_cpu_seconds']:.2f} CPU seconds**, **{data['costs']['process_peak_rss_bytes']/2**20:.2f} MiB peak process RSS**, and **{data['work']['operation_sum']:,} heterogeneous counted operations**. Each model trained for the same 1,400 updates. The CPU ran one Torch thread. Phase costs and operation categories are in the data file; counts are not FLOPs. Two separate four-step plumbing checks cost 2.5 and 2.3 seconds; verification and development are outside the timed study. The live attempt took {live['wall_seconds']:.2f} seconds including process startup. No cloud compute was used.", "",
              "## Checklist and source-packet alignment", "",
              "- [x] Preserve old source, models, results, failed trials and continuation history.",
              "- [x] Name useful variants and record exact checkpoint/evidence identities.",
              "- [x] Add semantic overlap checks and explicit withheld composition patterns.",
              "- [x] Keep historical v1 evaluation available under its original source identity.",
              "- [x] Separate world prediction/control retention and cover installed legacy/typed outputs.",
              "- [x] Train three new typed models, compare against frozen models on identical cases, and retain negative findings.",
              "- [x] Reexecute historical proposals, audit the new evidence and exercise ordinary continuation.",
              "- [ ] Connect typed acquisition and replay to the same persistent world learner, with a separate-component control.",
              "- [ ] Demonstrate sequential improver benefit against fixed anchors, cumulative frontier and a full-budget baseline.",
              "- [ ] Gate calibration/improver quality and establish compatible model growth or cross-world abstraction.", "",
              "The original handbook's Recipe 2 (editable source lines 941–947) asks for task mixtures and held-out generators/compositions; Recipe 3 (951–957) asks for transfer, old-task retention, calibration and cost. The new protocol addresses the bounded composition and retention gaps from D03/D04. It does not close the shared-learner gap (D01), frozen-improver gap (D02), or broader lifecycle/growth contracts. The handbook's ranked hypotheses still favor a narrow R1 plus selected R2 path (1256–1272). Keeping HMM, hybrid and reference variants as controls is consistent with testing that path rather than erasing alternatives.", "",
              "The next task is a minimal shared binding-acquisition stream: failure, admitted corrective evidence, retained update, fresh transfer and replay in the same persistent learner. Binding's weak and variable scores make it an explicit acceptance target. The separately trained typed component remains the control.", "",
              "## Reproduce", "", "```text",
              "python scripts/evaluate_variants.py --study runs/stage-three-complete --output runs/my-evaluation-v2",
              "python scripts/audit_evaluation_v2.py --study runs/my-evaluation-v2 --output runs/my-evaluation-v2-audit.json",
              "python -m pytest",
              "python scripts/verify_release.py", "```", "",
              "The comparison requires the preserved 0.3 checkpoint cohort. The registry identifies local artifacts; it does not claim that ignored checkpoints are included in a fresh Git checkout. Historical source verification reads the pinned Git commit, so fetch full history if using a shallow checkout.", ""]
    (ROOT / "reports/evaluation-v2-study.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    paths = ["evaluation-v2-data.json", "evaluation-v2-verification.json", "evaluation-v2-study.md", "evaluation-v2-results.png",
             "../research/variants.json", "../research/variants.md"]
    write_json(ROOT / "reports/evaluation-v2-evidence-manifest.json",
               [{"file": p, "sha256": hashlib.sha256((ROOT / "reports" / p).read_bytes()).hexdigest()} for p in paths])
    print("Published compact evidence, variant tables, retention decisions and performance figure.")


if __name__ == "__main__":
    main()
