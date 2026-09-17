"""Summarize sealed predictions without new fitting or final-set decisions."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .common import EXPERIMENT, OUT, RUNS, read, write


def main():
    results = {
        p.stem.removeprefix("final-"): read(p)
        for p in EXPERIMENT.glob("final-*.json")
        if p.name != "final-open.json"
    }
    rows = read(RUNS / "final.json")
    physical = read(RUNS / "evaluation-final-physical.json")
    instant = read(RUNS / "evaluation-final-instant.json")
    errors = []
    for world in range(12):
        indices = [
            i
            for i, r in enumerate(rows)
            if r["family"] == "delayed" and r["condition"] == "clean" and r["world"] == world
        ]
        truth = np.array([rows[i]["truth"][-1] for i in indices])
        errors.append(
            [
                np.mean((np.array([r["predictions"][i][-1] for i in indices]) - truth) ** 2)
                for r in (physical, instant)
            ]
        )
    errors = np.array(errors)
    rng = np.random.default_rng(28444)
    boot = []
    for _ in range(2000):
        sample = errors[rng.integers(0, len(errors), len(errors))].mean(0)
        boot.append(1 - sample[0] / sample[1])
    reduction = 1 - errors[:, 0].mean() / errors[:, 1].mean()
    groups = [
        ("Selected physical", ["physical"]),
        ("Always memory", ["memory"]),
        ("Instantaneous", ["instant"]),
        ("Adapted R1", ["r1_adapted-2801", "r1_adapted-2802"]),
        ("Frozen R1", ["r1_frozen-2801", "r1_frozen-2802"]),
        ("Reset R1", ["r1_reset-2801", "r1_reset-2802"]),
        ("GRU", ["gru-2801", "gru-2802"]),
        ("8-step MLP", ["window-2801", "window-2802"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True, constrained_layout=True)
    for ax, condition in zip(axes, ("clean", "noisy", "missing")):
        for i, (_, arms) in enumerate(groups):
            values = [
                results[arm]["metrics"][f"delayed/{condition}"]["12"]["velocity_rmse"]
                for arm in arms
            ]
            ax.scatter(values, [i] * len(values), color="#126b68" if i < 3 else "#415a91", s=45)
            if len(values) > 1:
                ax.plot(values, [i] * len(values), color="#415a91", alpha=0.5)
        ax.set_title(condition.capitalize() + " observations")
        ax.set_xscale("log")
        ax.set_xlim(0.004, 0.36)
        ax.set_xticks([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.10", "0.30"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Velocity RMSE at 0.6 s (m/s)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(groups)), [g[0] for g in groups])
    axes[0].invert_yaxis()
    fig.suptitle(
        "SERA C01/C02: new delayed-system worlds\nTwo points denote separate training seeds; physical fits use observed subject histories",
        fontsize=12,
    )
    fig.savefig(OUT / "results.png", dpi=170)
    fig.savefig(OUT / "results.svg")
    plt.close(fig)
    choice = {}
    for family in ("simple", "delayed", "omitted"):
        for condition in ("clean", "noisy", "missing"):
            models = [
                m
                for r, m in zip(rows, physical["models"])
                if r["family"] == family and r["condition"] == condition
            ]
            choice[f"{family}/{condition}"] = {
                k: sum(m["kind"] == k for m in models) for k in ("coarse", "instant", "memory")
            }
    write(
        OUT / "analysis.json",
        {
            "delayed_mse_reduction": reduction,
            "world_cluster_bootstrap_95_percent": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_scope": "12 independent parameter-world clusters; descriptive interval, not a new selection gate",
            "models_selected": choice,
            "training": {p.parent.name: read(p) for p in sorted(RUNS.glob("*/fit.json"))},
        },
    )
    print(
        json.dumps(
            {"reduction": reduction, "bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist()}
        )
    )


if __name__ == "__main__":
    main()
