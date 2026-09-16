"""Derived comparisons and exportable figures from frozen evaluated artifacts."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .study import RELEASE, write


def main():
    audit = json.loads((RELEASE/"TT-FINAL-001/independent-audit.json").read_text())
    table = audit["table"]
    counts = {}
    for policy in ("scratch", "transfer", "wrong_transfer", "matched_label_scratch"):
        rows = [r for r in table if r["count"] == 12 and r["policy"] == policy]
        counts[policy] = {"accepted": sum(r["accepted"] for r in rows),
                          "fully_accurate": sum(r["fully_accurate"] for r in rows),
                          "accepted_over_5_percent_future_risk": sum(r["accepted_over_5_percent_future_risk"] for r in rows),
                          "accepted_wrong_targeted_probe": sum(r["accepted_with_wrong_targeted_probe"] for r in rows)}
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.7), gridspec_kw={"width_ratios": [1.45, 1]})
    families = ["related", "innovation", "noisy_innovation", "dense", "local_exception", "off_family"]
    labels = ["Related", "Sparse\nchange", "Noisy\nchange", "Unrelated\ndense", "Rare\nexception", "Outside\nfamily"]
    x = np.arange(len(families))
    styles = [("scratch", "Scratch · 12 fit", "#a3acbb"), ("transfer", "Acquired structure · 12 fit", "#087e8b"),
              ("matched_label_scratch", "Label-cost control · 44 fit", "#df8a27")]
    for offset, (policy, label, color) in enumerate(styles):
        values = [next(r["accepted"] for r in table if r["family"] == f and r["count"] == 12 and r["policy"] == policy) for f in families]
        axes[0].bar(x+(offset-1)*.25, values, width=.24, label=label, color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 27)
    axes[0].set_yticks([0, 6, 12, 18, 24])
    axes[0].set_ylabel("Tasks admitted by untouched calibration / 24")
    axes[0].set_title("Reuse helps when new-task examples are scarce", loc="left", fontweight="bold", fontsize=12)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0., -.17), frameon=False, fontsize=9, ncol=1)
    for family, label, color in (("related", "Related", "#087e8b"), ("innovation", "Sparse change", "#4361a9"),
                                 ("noisy_innovation", "Noisy change", "#df8a27"), ("dense", "Unrelated dense", "#9d3a5b")):
        values = [next(r["accepted"] for r in table if r["family"] == family and r["count"] == n and r["policy"] == "transfer") for n in (8, 12, 20)]
        axes[1].plot([8, 12, 20], values, "o-", label=label, color=color, linewidth=2)
    axes[1].set_xticks([8, 12, 20])
    axes[1].set_yticks([0, 6, 12, 18, 24])
    axes[1].set_ylim(-.5, 26)
    axes[1].set_xlabel("Fitting examples (+16 selection +64 calibration)")
    axes[1].set_ylabel("Admitted transfer tasks / 24")
    axes[1].set_title("More observations still matter", loc="left", fontweight="bold", fontsize=12)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0., -.17), ncol=2, frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    figure.suptitle("SERA · learned task structure and sparse innovations", x=.06, ha="left", fontweight="bold", fontsize=16)
    figure.text(.06, .014, "24 independent synthetic task families. Prior acquisition: 512 labels. Rare exceptions can pass calibration and still fail targeted queries.", fontsize=10, color="#404752")
    figure.subplots_adjust(left=.06, right=.98, top=.84, bottom=.3, wspace=.3)
    figure.savefig(RELEASE/"performance.png", dpi=160, facecolor="white")
    plt.close(figure)
    result = {"final_records": audit["records"], "independent_task_families": 24,
              "development_records": 648, "count_12_totals": counts,
              "fresh_labels_at_12_fit": 92, "prior_acquisition_labels": 512,
              "matched_label_control_fresh_labels": 124, "amortization_horizon_for_control": 16,
              "note": "The 16-task horizon is an explicit amortization assumption for a strong label-cost control, not a measured sixteen-task deployment lifetime.",
              "pairwise_improvement": audit["paired_mean_improvement"],
              "normal_lower": audit["paired_normal_95_lower"],
              "physical_research_labels_final": 24*(2*4*128+6*(52+16+64)),
              "physical_future_evaluations_final": 24*6*512,
              "physical_targeted_evaluations_final": 24*6*16,
              "readout_tensor_bytes_each": 20*8,
              "figure": "performance.png"}
    write(RELEASE/"results.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
