"""Aggregate complete frozen GG-P0 records without selecting or retraining models."""

import argparse
import collections
import json
import platform
import time
from pathlib import Path

import numpy as np


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(81831)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {"difference": float(values.mean()), "lower95": float(np.quantile(means, .025)),
            "upper95": float(np.quantile(means, .975)), "paired_instances": len(values),
            "interpretation": "Descriptive percentile bootstrap across paired instances"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads((args.run / "state.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    assert state["status"] == "COMPLETED"
    rows, groups = [], collections.defaultdict(list)
    start = time.perf_counter()
    for completed in state["completed"]:
        data = json.loads((args.run / completed["path"]).read_text())
        for record in data["records"]:
            row = {**data["case"], **record}
            rows.append(row)
            groups[(row["family"], row["noise"], row["support"], row["method"])].append(row)
    expected = np.prod([len(protocol[k]) for k in ("seeds", "families", "noise", "support", "methods")])
    assert len(rows) == expected
    decoder_bytes = (args.run / "source/core.py").stat().st_size
    tables = []
    for (family, noise, n, method), values in sorted(groups.items()):
        assert len(values) == len(protocol["seeds"])
        tables.append({"family": family, "noise": noise, "support": n, "method": method,
                       "instances": len(values),
                       "mean_artifact_bytes": float(np.mean([v["artifact_bytes"] for v in values])),
                       "mean_standalone_model_and_decoder_bytes": float(np.mean([v["artifact_bytes"] + decoder_bytes for v in values])),
                       "mean_model_and_decoder_bytes_at_100_objects": float(np.mean([v["artifact_bytes"] + decoder_bytes / 100 for v in values])),
                       "mean_fit_wall_seconds": float(np.mean([v["work"]["wall_seconds"] for v in values])),
                       "mean_inference_wall_seconds": float(np.mean([v["inference_wall_seconds"] for v in values])),
                       "alarm_rate": float(np.mean([v["calibration"]["menu_inadequacy_alarm"] for v in values])),
                       "evaluation": {split: {metric: float(np.mean([v["evaluation"][split][metric] for v in values]))
                                              for metric in values[0]["evaluation"][split]}
                                      for split in values[0]["evaluation"]}})
    comparisons = []
    for family, challenger, baseline in (("circle", "G_program", "B_interpolation"),
                                         ("ellipse", "D_family", "C_circle"),
                                         ("exception", "H_residual", "G_program"),
                                         ("random", "G_program", "A_episodic")):
        for noise in protocol["noise"]:
            for n in protocol["support"]:
                a = {r["seed"]: r for r in groups[(family, noise, n, challenger)]}
                b = {r["seed"]: r for r in groups[(family, noise, n, baseline)]}
                for split in ("interpolation", "withheld_arc", "extrapolation"):
                    diff = [a[s]["evaluation"][split]["mse"] - b[s]["evaluation"][split]["mse"] for s in protocol["seeds"]]
                    comparisons.append({"family": family, "noise": noise, "support": n,
                                        "challenger": challenger, "baseline": baseline, "split": split,
                                        "metric": "MSE: negative difference favors challenger", **paired_interval(diff)})
    source_bytes = sum(p.stat().st_size for p in (args.run / "source").iterdir())
    total_bytes = sum(p.stat().st_size for p in args.run.rglob("*") if p.is_file())
    payload_bytes = sum(row["artifact_bytes"] for row in rows)
    raw_observation_rows = sum(row["work"]["support_observations"] + row["work"]["selection_observations"]
                               + row["work"]["calibration_observations"] for row in rows)
    report = {"experiment_id": protocol["experiment_id"], "source_sha256": state["contract"]["source_sha256"],
              "cases": len(state["completed"]), "models": len(rows), "conditions": len(rows) * 4,
              "tables": tables, "paired_comparisons": comparisons,
              "cost": {"charged_case_wall_seconds": state["charged_seconds"],
                       "sum_fit_cpu_seconds": sum(r["work"]["cpu_seconds"] for r in rows),
                       "sum_fit_wall_seconds": sum(r["work"]["wall_seconds"] for r in rows),
                       "sum_inference_wall_seconds": sum(r["inference_wall_seconds"] for r in rows),
                       "candidate_fits": sum(r["work"]["candidate_fits"] for r in rows),
                       "gradient_steps": sum(r["work"]["gradient_steps"] for r in rows),
                       "model_exposures_to_observation_rows": raw_observation_rows,
                       "all_model_artifact_bytes": payload_bytes, "decoder_source_bytes": decoder_bytes,
                       "frozen_source_bytes": source_bytes, "entire_run_archive_bytes": total_bytes,
                       "memory": state.get("memory"),
                       "boundary": "Object JSON includes numerical parameters, covariance, schema, provenance, guards and residuals; decoder source is separately included. Installed Python/NumPy/Torch runtime, virtual environment, research time and external audit archives are additional costs. No energy or full training optimizer peak breakdown measured."},
              "environment": state["environment"], "analysis_platform": platform.platform(),
              "analysis_wall_seconds": time.perf_counter() - start,
              "status": "exploratory_results_not_capability_promotion"}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# GG-P0-001: compact generative memory results", "",
             "I trained and evaluated eight controls on four supplied-phase task families. This is a component study; it does not establish discovery from unordered points, learned semantic applicability, shared neural transfer or improved investigation.", "",
             f"The complete cohort contains {len(state['completed'])} paired cases and {len(rows)} serialized models. Source identity: `{report['source_sha256']}`. All methods used the same observations within a case. The neural control used 160 gradient steps; symbolic search evaluated 130 supplied expression candidates.", "",
             "The following tables show means over 20 instances at 64 support observations and noise standard deviation 0.02. All other support/noise strata and paired intervals are retained in `summary.json`. MSE is lower-is-better; withheld-arc coverage is a diagnostic for 95% Gaussian-moment marginal intervals, not a guarantee.", ""]
    for family in protocol["families"]:
        lines += [f"## {family}", "", "| Control | In-region MSE | Withheld-arc MSE | Extrapolation MSE | Arc coverage | Artifact bytes | Alarm rate |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for t in tables:
            if (t["family"], t["noise"], t["support"]) == (family, .02, 64):
                e = t["evaluation"]
                lines.append(f"| {t['method']} | {e['interpolation']['mse']:.6g} | {e['withheld_arc']['mse']:.6g} | {e['extrapolation']['mse']:.6g} | {e['withheld_arc']['coverage95']:.1%} | {t['mean_artifact_bytes']:.0f} | {t['alarm_rate']:.0%} |")
        lines.append("")
    cost = report["cost"]
    lines += ["## Cost and claim boundary", "",
              f"Case execution charged {cost['charged_case_wall_seconds']:.2f} seconds, including serialization and scoring. The methods used {cost['candidate_fits']:,} candidate fits and {cost['gradient_steps']:,} neural gradient steps. All model objects total {payload_bytes:,} bytes; the whole preserved run is {total_bytes:,} bytes. The shared decoder source adds {decoder_bytes:,} bytes to a standalone deployment, or {decoder_bytes/100:.2f} bytes per object at an explicitly declared 100-object deployment. Runtime dependencies remain additional to these figures.", "",
              "The circle and other program primitives, phase convention, noise level, expression grammar, fitting algorithms and neural architecture were supplied. Learned quantities are coefficients, class weights or expression choices, network weights, local residuals and calibration variance. A class posterior remains conditional on its menu. A residual alarm is empirical and may miss an unseen exception.", "",
              "A stores exact observations and marks unfamiliar queries unavailable; its reported unseen MSE uses the fixed zero-mean fallback so abstention does not remove hard cases. Random unseen values are independent facts and remain unpredictable. B clamps beyond the observed interval. H's local residual applies only inside its stored support range. E/G select a single candidate using a fixed selection criterion; their uncertainty omits selection uncertainty. D uses coherent model averaging but may still miss an absent class.", "",
              "All intervals are exploratory paired-instance intervals, without multiplicity correction or a claim of confirmed superiority. The complete model records, candidate outcomes, source copies and query vectors remain in the run directory. Fresh-process replay is reported separately.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": report["cases"], "models": report["models"], "cost": cost}, indent=2))


if __name__ == "__main__":
    main()
