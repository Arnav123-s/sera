"""Standalone research figure from preserved results; no refitting or new evaluation."""

import matplotlib

from experiments.language_inquiry.study import read

from .study import OUT

matplotlib.use("Agg")


def main():
    import matplotlib.pyplot as plt

    language = read(OUT / "evaluation/language-summary.json")
    followup = read(OUT / "followup/summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    colors = ["#2e6d93", "#bd6049"]
    for j, partition in enumerate(("pairs", "surface")):
        rows = [r for r in language if r["kind"] == "ordered" and r["steps"] == 200 and r["partition"] == partition]
        value = 100 * sum(r["correct"] for r in rows) / sum(r["total"] for r in rows)
        axes[0].bar(j, value, color=colors[j], width=.55)
        axes[0].text(j, value + 3, f"{value:.0f}%", ha="center")
    axes[0].set(xticks=[0, 1], xticklabels=["New directed pairs", "New wording"], ylim=(0, 115),
                ylabel="Exact interpretation (%)", title="Roles transfer; wording fails")
    narrow = {r["method"]: r for r in followup if r["family"] == "narrow"}
    for j, method in enumerate(("uniform", "amortized")):
        value = narrow[method]["gradient_steps"]
        axes[1].bar(j, value, color=["#798594", "#2e6d93"][j], width=.55)
        axes[1].text(j, value + 3, str(value), ha="center")
    axes[1].set(xticks=[0, 1], xticklabels=["Uniform + stop", "Learned + stop"], ylim=(0, 175),
                ylabel="Total refinement steps", title="128/128 narrow cases solved")
    omitted = next(r for r in followup if r["family"] == "omitted_torque" and r["method"] == "amortized")
    for j, key in enumerate(("nominal_witnesses", "assessed_successes")):
        value = 100 * omitted[key] / omitted["cases"]
        axes[2].bar(j, value, color=colors[j], width=.55)
        axes[2].text(j, value + 3, f"{value:.1f}%", ha="center")
    axes[2].set(xticks=[0, 1], xticklabels=["Within model", "With omitted law"], ylim=(0, 115),
                ylabel="Success (%)", title="Confidence does not fix missing physics")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("SERA: shared-owner language and conditional settling", fontsize=15, x=.05, ha="left")
    fig.text(.05, .035, "Finite supplied grammar, two fit seeds. Follow-up: fresh numeric cases from inspected families.\n"
             "The analytic control remained cheaper. New route is conditional and uncalibrated; no learned investigator or quantum advantage.", fontsize=9)
    fig.tight_layout(rect=[.03, .12, .99, .91], w_pad=2)
    fig.savefig(OUT / "results.png", dpi=160)
    fig.savefig(OUT / "results.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
