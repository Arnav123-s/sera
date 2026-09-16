"""Export audited tables, transparent reporting corrections and research figures."""

import gzip
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .arrays import ArrayReader
from .study import RELEASE, write


def read(name):
    return json.loads((RELEASE/name).read_text())


def corrections():
    corrected = []
    with gzip.open(RELEASE/"CS-OWNER-PILOT-001/records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            original, updated, fault = [np.asarray(row[name]) for name in ("pristine", "updated", "fault")]
            corrected.append({"seed": row["seed"], "family": row["family"], "count": row["count"], "method": row["method"],
                              "original_reported_coefficient_error": row["coefficient_error"],
                              "corrected_residual_weight_error": float(np.linalg.norm(updated-original)/np.linalg.norm(fault))})
    write(RELEASE/"CS-OWNER-PILOT-001/score-correction.json", {"status": "CORRECTED_FROM_SAVED_COEFFICIENTS", "records": corrected,
            "reason": "Original secondary metric subtracted the fault twice. Source, original score and primary predictions are preserved. No refitting."})
    adjustments = {}
    for name in ("CS-IMAGING-PILOT-001", "CS-IMAGING-PILOT-002", "CS-IMAGING-FINAL-001"):
        rows = read(name+"/summary.json")["rows"]
        reported = sum(r["work"]["fft_calls"] for r in rows)
        missed = sum(r["method"] != "zero_filled" for r in rows)
        adjustments[name] = {"reported_reconstruction_fft_calls": reported,
                             "omitted_final_stationarity_ifft_calls": missed,
                             "corrected_reconstruction_fft_calls": reported+missed,
                             "oracle_ffts_separately": read(name+"/summary.json")["full_oracle_ffts"]}
    write(RELEASE/"cost-counter-corrections.json", {"imaging": adjustments,
           "reason": "One final stationarity inverse FFT was omitted per nonzero-filled reconstruction. Actual wall time always included it; fixed source has a counter-instrumentation test. Historical outputs remain unchanged.",
           "omp": "The synthetic weight study's linear_solves field=36 is a conservative maximum over the 1..8 grid. Early residual convergence can execute fewer solves. Use measured wall time for comparisons; do not interpret that field as an exact count."})


def figures():
    app = read("CS-APPLICATION-FINAL-001/independent-audit.json")["table"]
    owner = read("CS-OWNER-FINAL-001/independent-audit.json")["table"]
    query = read("CS-QUERY-FINAL-001/independent-audit.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.3), constrained_layout=True)
    colors = ["#147c80", "#d68543", "#8593a6"]
    def bars(ax, labels, series, names, title, denominator):
        width = .75/len(series)
        x = np.arange(len(labels))
        for i, (values, name) in enumerate(zip(series, names, strict=True)):
            ax.bar(x+(i-(len(series)-1)/2)*width, np.asarray(values)/denominator, width=width, color=colors[i], label=name)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), ylabel="Fraction recovered")
        ax.set_title(title, fontsize=11, pad=31)
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    labels = ["4 active", "4→12 active", "Dense"]
    series = [[app[f"memory/{family}/32/none/{method}"]["exact_at_1e_6"] for family in ("sparse4", "growing", "dense")]
              for method in ("basis_pursuit", "minimum_norm")]
    bars(axes[0, 0], labels, series, ["Sparse", "Min-norm"], "Memory sketches\n32 measurements · n=24", 24)
    methods = ("basis_pursuit", "omp", "ridge")
    bars(axes[0, 1], ["Sparse update", "Dense update"],
         [[app[f"weights/tanh/{family}/32/{method}"]["within_5_percent"] for family in ("sparse", "dense")] for method in methods],
         ["Sparse", "OMP", "Ridge"], "Fixed tanh network\n32 fit + 16 selection · n=24", 24)
    bars(axes[0, 2], ["32 anchors", "64 anchors"],
         [[owner[f"sparse4/{count}/{method}"]["within_5_percent"] for count in (32, 64)] for method in ("sparse_delta", "omp_delta", "ridge_delta")],
         ["Sparse", "OMP", "Ridge"], "Actual SERA weight repair\n32 selection anchors additionally · n=12", 12)
    bars(axes[1, 0], ["4 faults", "12 faults", "24 faults"],
         [[app[f"corruption/{count}/{method}"]["exact_at_1e_6"] for count in (4, 12, 24)] for method in ("sparse_error", "least_squares")],
         ["Sparse errors", "Least squares"], "Dense payload\n64 coded values · n=24", 24)
    bars(axes[1, 1], ["Equal observations", "Equal work proxy"],
         [[query["comparisons"][regime]["adequate"][method]["solved"] for regime in ("equal_observations", "equal_work_proxy")] for method in ("random", "disagreement")],
         ["Random", "Disagreement"], "Investigation\nSparse/noisy worlds · n=48", 48)
    bars(axes[1, 2], ["8 new phases", "12 new phases"],
         [[query["priors"][f"{count}/{method}"]["within_5_percent"] for count in (8, 12)] for method in ("scratch", "acquired_prior", "wrong_prior")],
         ["Scratch", "Acquired prior", "Wrong prior"], "Reusing earlier observed support\nn=24", 24)
    fig.suptitle("Sparse recovery helps when the structure and measurements support it", fontsize=17)
    fig.supxlabel("Different controlled experiments, not a common benchmark. Memory tolerance: 10⁻⁶; future prediction tolerance: 5%.\nSynthetic studies plus one saved SERA owner; descriptive screening, no universal or statistical-significance claim.", fontsize=10)
    fig.savefig(RELEASE/"performance.png", dpi=160)
    plt.close(fig)
    folder = RELEASE/"CS-IMAGING-FINAL-001"
    reader = ArrayReader(folder/"arrays.zip")
    selected = defaultdict(dict)
    choices = {"sparse_pixels": (1010027, 128, "points_uniform", "pixel"),
               "sparse_haar": (1020034, 256, "points_variable", "haar"),
               "phantom_faint": (1030041, 256, "points_variable", "haar")}
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] not in choices:
                continue
            seed, count, geometry, method = choices[row["family"]]
            if row["seed"] == seed and row["count"] == count and row["geometry"] == geometry and row["noise_sigma"] == 0 and row["method"] in (method, "zero_filled"):
                selected[row["family"]][row["method"]] = row
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for index, (family, (_, count, geometry, method)) in enumerate(choices.items()):
        row, zero = selected[family][method], selected[family]["zero_filled"]
        actual, recovered, filled = [reader.get(value) for value in (row["image"], row["reconstructed"], zero["reconstructed"])]
        low, high = float(actual.min()), float(actual.max())
        for column, (pixels, title) in enumerate(((actual, "Original"), (filled, f"Zero-filled\nNRMSE {zero['relative_error']:.3g}"),
                                                (recovered, f"{method.title()} recovery\nNRMSE {row['relative_error']:.3g}"),
                                                (abs(recovered-actual), "Absolute error"))):
            ax = axes[index, column]
            plotted = ax.imshow(pixels, cmap="magma" if column == 3 else "gray", vmin=0 if column == 3 else low, vmax=None if column == 3 else high, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(f"{family.replace('_', ' ')}\n{count}/1024 complex samples\n{geometry.replace('_', ' ')}", fontsize=10)
            if column == 3:
                fig.colorbar(plotted, ax=ax, shrink=.65)
            if family == "phantom_faint" and column < 3:
                roi = reader.get(row["roi"])
                yy, xx = np.where(roi)
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((xx.min()-.5, yy.min()-.5), 2, 2, fill=False, edgecolor="#08c4b8", linewidth=1.5))
    fig.suptitle("Fourier reconstruction: first final example in each displayed family", fontsize=16)
    fig.supxlabel("Synthetic images only. Sparse pixel/Haar cases are deliberately matched to a basis.\nThe turquoise box marks a 0.04-contrast feature: global image quality does not establish its preservation.", fontsize=10)
    fig.savefig(RELEASE/"imaging-comparison.png", dpi=160)
    plt.close(fig)
    reader.close()


def main():
    corrections()
    figures()
    applications = read("CS-APPLICATION-FINAL-001/independent-audit.json")
    recovery = read("CS-FINAL-001/independent-audit.json")
    queries = read("CS-QUERY-FINAL-001/independent-audit.json")
    imaging = read("CS-IMAGING-FINAL-001/independent-audit.json")
    owner = read("CS-OWNER-FINAL-001/independent-audit.json")
    memory = read("CS-REAL-MEMORY-001/result.json")
    result = {"status": "RESEARCH_CYCLE_COMPLETE", "new_final_records": recovery["records"]+queries["lifetimes"]+queries["prior_reconstructions"]+applications["records"]+imaging["records"]+owner["records"]+len(memory["rows"]),
              "original_packet_replayed": 400,
              "promotion": {"mechanism_supported_answers": recovery["qualification"],
                            "disagreement_investigator": queries["investigator_decision"],
                            "memory_utility": "EXPERIMENTAL_EXACT_BYTES_WITH_TRUSTED_DIGEST",
                            "live_owner": "UNCHANGED"},
              "actual_owner": owner["parent_owner"],
              "final_audits": {name: data["status"] for name, data in (("recovery", recovery), ("queries", queries), ("applications", applications), ("imaging", imaging), ("owner", owner))}}
    write(RELEASE/"results.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
