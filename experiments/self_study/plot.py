"""Static research figure from completed, saved result records only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/27_self_study"


def read(path):
    return json.loads(path.read_text())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for seed, style in ((2711, "-"), (2712, "--")):
        for arm, color in (("replay", "#2463a0"), ("naive", "#ad5636")):
            name = f"SS-lang-{arm}-{seed}" + ("-r1" if arm == "replay" and seed == 2711 else "")
            record = read(ROOT / "runs" / name / "result.json")
            values = [100 * record["baseline"]["en-US"]["intent_accuracy"]]
            values.extend(100 * stage["development"]["en-US"]["intent_accuracy"] for stage in record["history"])
            axes[0].plot(range(4), values, style, color=color, marker="o", linewidth=1.8,
                         label=f"{'Rehearsal' if arm == 'replay' else 'Current only'} / {seed}")
    repaired = read(ROOT / "runs/SS-rehearsal-2721/result.json")
    axes[0].plot([3, 4], [100 * repaired["before"]["en-US"]["intent_accuracy"],
                        100 * repaired["development"]["en-US"]["intent_accuracy"]],
                 color="#27835d", marker="D", linewidth=2.4, label="Larger rehearsal / 2721")
    axes[0].set(title="English retention during new-language learning", ylabel="Development intent accuracy (%)",
                xticks=range(5), xticklabels=["Parent", "Spanish", "French", "German", "+800\nupdates"], ylim=(0, 100))
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    final = read(OUT / "language-evaluation/summary.json")["SS-rehearsal-2721"]
    for offset, key, label, color in ((-.18, "intent_accuracy", "Intent", "#2463a0"),
                                      (.18, "frame_accuracy", "Complete frame", "#27835d")):
        values = [100 * final[locale][key] for locale in ("es-ES", "fr-FR", "de-DE")]
        bars = axes[1].bar([i + offset for i in range(3)], values, .34, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(title="Admitted successor: reserved final examples", ylabel="Accuracy (%)",
                xticks=range(3), xticklabels=["Spanish", "French", "German"], ylim=(0, 100))
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("SERA continuing request learning · SS-002 / SS-004", fontsize=14)
    fig.savefig(OUT / "results.png", dpi=180)
    fig.savefig(OUT / "results.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
