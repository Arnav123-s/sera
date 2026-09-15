"""Frozen, episode-disjoint assessor and execution harness for GG-GUARD-001.

Explicit pytest invocation lets the existing bounded supervisor execute this
experiment without modifying the supervisor pinned by older protocols.
SERA_GUARD_MODE is timing, final, or replay; SERA_GUARD_OUTPUT must be fresh.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .acquisition import Action, Posterior, observed_event
from .applicability import (
    FEATURES,
    METHODS,
    LearnedGuard,
    control_scores,
    extract_features,
    fit_guard,
    select_threshold,
)
from .core import CIRCLE, ELLIPSE, LINE, canonical, design, digest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "research-continuation/15_applicability"
TRAIN_FAMILIES = ("circle", "ellipse", "line", "radial_orbit", "local_exception", "random_values")
FUTURE_FAMILIES = ("piecewise_drift", "chirp")
FAMILIES = TRAIN_FAMILIES + FUTURE_FAMILIES
SUPPORT_T = np.linspace(-1.5, 1.5, 25)
# Fixed space filling, declared before observations. All guards use this prefix.
SUPPORT_INDICES = (12, 0, 24, 6, 18, 3, 9, 15, 21, 1)
QUERY_T = np.linspace(-1.8, 1.8, 33)
VARIANCE = .05**2
VALID_RADIUS = .2
PARTITION_BASES = {"timing": 310000, "teach": 410000, "probability_calibration": 510000,
                   "threshold_selection": 610000, "final": 710000, "alias_audit": 810000}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def rng_for(seed, role):
    value = hashlib.sha256(f"GG-GUARD-001/{seed}/{role}".encode()).digest()
    return np.random.default_rng(np.frombuffer(value[:16], dtype=np.uint32))


def seed_for(split, family, index):
    if split not in PARTITION_BASES or family not in FAMILIES or not 0 <= index < 1000:
        raise ValueError("Invalid split, family or bounded instance index")
    if split not in {"timing", "final", "alias_audit"} and family in FUTURE_FAMILIES:
        raise ValueError("Protected future family requested before final evaluation")
    return PARTITION_BASES[split] + 1000*FAMILIES.index(family)+index


def assessor(seed, family):
    """Private immutable world and independent measurement noise streams.

    random_values is a continuous interpolation of independently drawn latent
    knot values. Repeated coordinates denote the same latent world, not new tasks.
    """
    mechanism = rng_for(seed, "mechanism")
    spec = {"circle": CIRCLE, "ellipse": ELLIPSE, "line": LINE,
            "radial_orbit": {"kind": "radial_orbit", "name": "linear-radius-orbit", "frequency": 1}}
    base = spec.get(family, CIRCLE)
    coefficients = mechanism.normal(0, .7, design(base, np.array([0.])).shape[1])
    extra = {"center": float(mechanism.uniform(-.65, .65)),
             "width": float(mechanism.uniform(.35, .75)),
             "amplitude": float(mechanism.uniform(.5, 1.5)),
             "phase": float(mechanism.uniform(-np.pi, np.pi)),
             "drift": mechanism.normal(0, 1, 2).tolist()}
    knots = mechanism.normal(0, 1, (65, 2))

    def clean(t):
        t = np.asarray(t)
        y = (design(base, t) @ coefficients).reshape(-1, 2)
        if family == "local_exception":
            u = (t-extra["center"])/extra["width"]
            window = np.where(np.abs(u) < 1, np.cos(np.pi*u/2)**2, 0)
            y += extra["amplitude"]*window[:, None]*np.array([1., -.7])
        elif family == "piecewise_drift":
            y += np.maximum(t-extra["center"], 0)[:, None]*np.array(extra["drift"])
        elif family == "chirp":
            phase = 3*np.pi*t**2+extra["phase"]
            y += extra["amplitude"]*np.stack([np.sin(phase), np.cos(phase)], axis=1)
        elif family == "random_values":
            y = np.stack([np.interp(t, np.linspace(-2, 2, 65), knots[:, d]) for d in range(2)], axis=1)
        elif family not in spec:
            raise ValueError("Unknown assessor family")
        return y

    truth_support, truth_query = clean(SUPPORT_T), clean(QUERY_T)
    support = truth_support + .05*rng_for(seed, "support_measurement").normal(size=truth_support.shape)
    query = truth_query + .05*rng_for(seed, "query_measurement").normal(size=truth_query.shape)
    return {"family": family, "environment_seed": seed, "coefficients": coefficients.tolist(),
            "extra": extra, "support_observed": support.tolist(), "query_observed": query.tolist(),
            "query_truth": truth_query.tolist()}


def evidence_episode(split, family, index):
    seed = seed_for(split, family, index)
    world = assessor(seed, family)
    # Opaque identities are never numeric input features; names/seeds stay assessor-only.
    identity = hashlib.sha256(f"world/{seed}".encode()).hexdigest()
    actions = tuple(Action(i, float(t), "target", VARIANCE) for i, t in enumerate(SUPPORT_T))
    posterior = Posterior(actions, max_observations=len(SUPPORT_INDICES))
    for n, index in enumerate(SUPPORT_INDICES):
        posterior.observe(observed_event(actions[index], world["support_observed"][index], n,
                                         f"{identity}/observation/{n}", "GG-GUARD-001/observed"))
    features, predictions = extract_features(posterior, QUERY_T, VARIANCE)
    means = np.array([p["mean"] for p in predictions])
    observed_squared_norm = np.sum((means-world["query_observed"])**2, axis=1)
    clean_mse = np.mean((means-world["query_truth"])**2, axis=1)
    valid = (observed_squared_norm <= VALID_RADIUS**2).astype(int)
    witness = {"query_t": QUERY_T.tolist(), "query_observed": world["query_observed"],
               "prediction_means_before_query_observation": means.tolist(),
               "radius": VALID_RADIUS, "evidence_role": "synthetic_observed_outcome"}
    teaching = {"episode_id": identity, "split": split, "features": features.tolist(),
                "observed_valid": valid.tolist(), "witness_sha256": digest(witness)}
    record = {"episode_id": identity, "split": split, "assessor": world,
              "posterior": posterior.snapshot(), "features": features.tolist(),
              "predicted_means": means.tolist(), "observed_valid": valid.tolist(),
              "clean_mse": clean_mse.tolist(), "witness": witness}
    return teaching, record


def assert_partition(records):
    ids, seeds = set(), set()
    for row in records:
        private = row["assessor"]
        if row["episode_id"] in ids or private["environment_seed"] in seeds:
            raise ValueError("Mechanism instance reused across partitions")
        ids.add(row["episode_id"])
        seeds.add(private["environment_seed"])
        if row["split"] != "final" and private["family"] in FUTURE_FAMILIES:
            raise ValueError("Future-family leakage")
    return {"worlds": len(ids), "cross_partition_overlap": 0,
            "protected_families": list(FUTURE_FAMILIES)}


def metrics(valid, mse, accepted):
    valid, accepted = np.asarray(valid, bool), np.asarray(accepted, bool)
    mse = np.asarray(mse)
    wrong, accepted_count = ~valid, int(accepted.sum())
    n = len(valid)
    return {"queries": n, "accepted": accepted_count, "wrong": int(wrong.sum()),
            "wrong_accepted": int((wrong & accepted).sum()), "valid_rejected": int((valid & ~accepted).sum()),
            "coverage": float(accepted.mean()),
            "selective_risk": float(wrong[accepted].mean()) if accepted_count else None,
            "false_acceptance_rate": float(accepted[wrong].mean()) if wrong.any() else None,
            "false_rejection_rate": float((~accepted[valid]).mean()) if valid.any() else None,
            "accepted_clean_mse": float(mse[accepted].mean()) if accepted_count else None,
            "accepted_clean_sse": float(mse[accepted].sum()),
            "utility": float(np.mean(accepted*np.where(valid, 1., -4.)))}


def probability_metrics(valid, probability):
    y, p = np.asarray(valid), np.asarray(probability)
    p_safe = np.clip(p, 1e-12, 1-1e-12)
    bins = []
    for index in range(10):
        selected = (np.minimum((p*10).astype(int), 9) == index)
        bins.append({"lower": index/10, "upper": (index+1)/10, "count": int(selected.sum()),
                     "mean_probability": float(p[selected].mean()) if selected.any() else None,
                     "observed_valid_fraction": float(y[selected].mean()) if selected.any() else None})
    return {"brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(p_safe)+(1-y)*np.log1p(-p_safe))),
            "calibration_bins": bins}


def pool(rows):
    count, wrong, queries = (sum(row[k] for row in rows) for k in ("accepted", "wrong_accepted", "queries"))
    invalid, rejected = sum(row["wrong"] for row in rows), sum(row["valid_rejected"] for row in rows)
    return {"worlds": len(rows), "queries": queries, "accepted": count,
            "coverage": count/queries, "wrong_accepted": wrong,
            "selective_risk": wrong/count if count else None,
            "false_acceptance_rate": wrong/invalid if invalid else None,
            "false_rejection_rate": rejected/(queries-invalid) if queries > invalid else None,
            "accepted_clean_mse": sum(r["accepted_clean_sse"] for r in rows)/count if count else None,
            "mean_utility": float(np.mean([r["utility"] for r in rows]))}


def summarize(records, guard, selections, *, bootstrap_seed=990001, bootstrap_samples=2000):
    result = {"by_family": {}, "paired_utility": {}, "risk_coverage": {}}
    for row in records:
        scores = control_scores(row["features"], guard)
        row["scores"] = {k: v.tolist() for k, v in scores.items()}
        row["methods"] = {}
        for method in METHODS:
            threshold = (0.5 if method == "always_predict" else 1. if method == "always_abstain"
                         else selections[method]["selected"]["threshold"])
            row["methods"][method] = metrics(row["observed_valid"], row["clean_mse"],
                                             scores[method] >= threshold)
    for family in FAMILIES:
        worlds = [r for r in records if r["assessor"]["family"] == family]
        result["by_family"][family] = {method: pool([r["methods"][method] for r in worlds])
                                        for method in METHODS}
        valid = np.concatenate([r["observed_valid"] for r in worlds])
        probabilities = np.concatenate([r["scores"]["learned"] for r in worlds])
        result["by_family"][family]["probability"] = probability_metrics(valid, probabilities)
        prior = guard.artifact()["payload"]["training"]["training_valid_fraction"]
        result["by_family"][family]["constant_probability"] = probability_metrics(valid, np.full(len(valid), prior))
        # Curves only describe the test ranking; their thresholds never select a deployed guard.
        curve = []
        for threshold in np.linspace(0, 1, 51):
            group = [metrics(r["observed_valid"], r["clean_mse"],
                             np.array(r["scores"]["learned"]) >= threshold) for r in worlds]
            curve.append({"threshold": float(threshold), **pool(group)})
        result["risk_coverage"][family] = curve
    rng = np.random.default_rng(bootstrap_seed)
    for baseline in ("always_predict", "residual_uncertainty", "support_distance"):
        deltas_by_family = [np.array([r["methods"]["learned"]["utility"]-r["methods"][baseline]["utility"]
                                     for r in records if r["assessor"]["family"] == family]) for family in FAMILIES]
        means = np.stack([rng.choice(d, size=(bootstrap_samples, len(d)), replace=True).mean(axis=1)
                          for d in deltas_by_family]).mean(axis=0)
        result["paired_utility"][baseline] = {
            "mean_difference": float(np.mean([d.mean() for d in deltas_by_family])),
            "world_stratified_bootstrap_95": np.quantile(means, [.025, .975]).tolist()}
    base_coverage = all(result["by_family"][f]["learned"]["coverage"] >= .8 for f in TRAIN_FAMILIES[:4])
    risks = all(result["by_family"][f]["learned"]["selective_risk"] is not None
                and result["by_family"][f]["learned"]["selective_risk"] <= .10
                for f in FAMILIES if result["by_family"][f]["learned"]["coverage"] >= .05)
    future = pool([r["methods"]["learned"] for r in records if r["assessor"]["family"] in FUTURE_FAMILIES])
    control_gain = all(result["paired_utility"][b]["world_stratified_bootstrap_95"][0] > 0
                       for b in ("residual_uncertainty", "support_distance"))
    result["gate"] = {"in_menu_coverage_at_least_80_percent_each": base_coverage,
                      "risk_at_most_10_percent_each_family_with_5_percent_coverage": risks,
                      "future_pooled_coverage_at_least_10_percent": future["coverage"] >= .1,
                      "paired_utility_beats_both_fixed_controls": control_gain,
                      "component_promotion": base_coverage and risks and future["coverage"] >= .1 and control_gain,
                      "interpretation": "An empirical component gate, not an unseen-family guarantee or M2."}
    result["future_pooled"] = future
    return result


def alias_audit(guard, threshold):
    """Two worlds have identical observations but incompatible unmeasured answers.

    This diagnostic is not an extra training family. It witnesses the access limit:
    no learned guard can distinguish these worlds from this shared prefix alone.
    """
    _, row = evidence_episode("alias_audit", "circle", 0)
    support = SUPPORT_T[list(SUPPORT_INDICES)]
    t = QUERY_T[int(np.argmax(np.min(np.abs(QUERY_T[:, None]-support), axis=1)))]
    posterior = Posterior.restore(row["posterior"])
    features, prediction = extract_features(posterior, [t], VARIANCE)
    p = float(guard.predict(features)[0])
    query_index = int(np.flatnonzero(QUERY_T == t)[0])
    ordinary = np.asarray(row["assessor"]["query_truth"][query_index])
    # A compact bump supported around this unobserved query is zero at all support points.
    radius = float(np.min(np.abs(support-t))*.49)
    return {"same_observed_prefix_sha256": row["posterior"]["sha256"],
            "query_t": float(t), "bump_support_radius": radius,
            "bump_at_all_observed_positions": [0.]*len(support),
            "same_feature_sha256": digest(features.tolist()), "identical_validity_probability": p,
            "identical_accept_decision": p >= threshold,
            "identical_predicted_mean": prediction[0]["mean"],
            "compatible_query_world_A": ordinary.tolist(),
            "compatible_query_world_B": (ordinary+np.array([2., -2.])).tolist(),
            "interpretation": "Finite observations cannot rule out an unobserved local exception; "
                              "the same answer is necessarily wrong in at least one compatible world."}


def source_inventory():
    names = ["experiments/generative_memory/"+x for x in
             ("applicability.py", "applicability_study.py", "applicability_audit.py", "acquisition.py", "core.py")]
    names += ["scripts/run_bounded.py", "scripts/windows_job.py", "tests/test_applicability.py"]
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"src/sera").glob("*.py"))]
    return {n: {"sha256": hashlib.sha256((ROOT/n).read_bytes()).hexdigest(), "bytes": (ROOT/n).stat().st_size}
            for n in sorted(names)}


def validate_protocol(protocol):
    payload = protocol["payload"]
    if digest(payload) != protocol["sha256"] or payload["sources"] != source_inventory():
        raise ValueError("Frozen protocol or executable source changed")
    if (payload["train_families"] != list(TRAIN_FAMILIES) or payload["future_families"] != list(FUTURE_FAMILIES)
            or payload["support_indices"] != list(SUPPORT_INDICES) or payload["features"] != list(FEATURES)):
        raise ValueError("Frozen family/access/feature contract differs")
    if payload["runtime"] != runtime():
        raise ValueError("Frozen numerical runtime changed")
    return payload


def execute(output, mode):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started, cpu = time.perf_counter(), time.process_time()
    if mode == "timing":
        records, teaching = [], []
        for family in TRAIN_FAMILIES:
            for i in range(2):
                train, record = evidence_episode("timing", family, i)
                records.append(record)
                # Timing models are isolated and permanently ineligible for final training.
                train["split"] = "teach" if i == 0 else "probability_calibration"
                teaching.append(train)
        model = fit_guard([r for r in teaching if r["split"] == "teach"],
                          [r for r in teaching if r["split"] == "probability_calibration"],
                          optimizer_seed=390001)
        LearnedGuard(model).predict(records[0]["features"])
        write(output/"timing.json", {"status": "PASS", "worlds": len(records), "queries_per_world": len(QUERY_T),
                                    "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                                    "model": model, "sources": source_inventory(),
                                    "scope": "Disjoint timing fixture only. No final or future family instantiated."})
        return
    if mode != "final":
        raise ValueError("Unknown execution mode")
    protocol = json.loads((RELEASE/"protocol.json").read_text())
    config = validate_protocol(protocol)
    write(output/"protocol.json", protocol)
    records, teaching = [], {}
    phase_seconds = {}
    # Generate final worlds only AFTER model fitting and threshold freezing below.
    for split in ("teach", "probability_calibration", "threshold_selection"):
        phase_start = time.perf_counter()
        teaching[split] = []
        for family in TRAIN_FAMILIES:
            for index in range(config["worlds_per_family"][split]):
                train, row = evidence_episode(split, family, index)
                teaching[split].append(train)
                records.append(row)
        phase_seconds[split+"_evidence"] = time.perf_counter()-phase_start
        print(f"Completed {split}: {len(teaching[split])} worlds", flush=True)
    phase_start = time.perf_counter()
    model = fit_guard(teaching["teach"], teaching["probability_calibration"],
                      optimizer_seed=config["optimizer_seed"], steps=config["training_steps"])
    write(output/"guard-model.json", model)
    guard = LearnedGuard(model)
    phase_seconds["guard_fit_and_probability_calibration"] = time.perf_counter()-phase_start
    threshold_ids = {r["episode_id"] for r in teaching["threshold_selection"]}
    if threshold_ids & set(model["payload"]["training"]["train_episode_ids"]+model["payload"]["training"]["calibration_episode_ids"]):
        raise ValueError("Threshold selection reused model-fitting worlds")
    scores = [control_scores(r["features"], guard) for r in teaching["threshold_selection"]]
    y = np.array([r["observed_valid"] for r in teaching["threshold_selection"]])
    selections = {method: select_threshold(np.array([s[method] for s in scores]), y)
                  for method in ("residual_uncertainty", "support_distance", "learned")}
    write(output/"thresholds.json", {"selections": selections, "world_ids": sorted(threshold_ids),
                                     "model_sha256": model["sha256"]})
    frozen_before_final = {"model_sha256": model["sha256"], "selections_sha256": digest(selections)}
    write(output/"before-final.json", frozen_before_final)
    final = []
    phase_start = time.perf_counter()
    for family in FAMILIES:
        for index in range(config["worlds_per_family"]["final"]):
            _, row = evidence_episode("final", family, index)
            final.append(row)
    phase_seconds["final_evidence"] = time.perf_counter()-phase_start
    partition = assert_partition(records+final)
    phase_start = time.perf_counter()
    summary = summarize(final, guard, selections)
    phase_seconds["final_scoring_and_bootstrap"] = time.perf_counter()-phase_start
    summary["partition"] = partition
    summary["alias_audit"] = alias_audit(guard, selections["learned"]["selected"]["threshold"])
    summary["model_sha256"] = model["sha256"]
    summary["protocol_sha256"] = protocol["sha256"]
    summary["phase_seconds"] = phase_seconds
    summary["costs"] = {"worlds_by_split": {s: len(teaching[s]) for s in teaching} | {"final": len(final)},
                        "observations_per_world": len(SUPPORT_INDICES), "outcome_pairs_per_world": len(QUERY_T),
                        "support_observations": len(records+final)*len(SUPPORT_INDICES),
                        "teaching_and_calibration_outcomes": len(records)*len(QUERY_T),
                        "final_assessment_outcomes": len(final)*len(QUERY_T),
                        "additional_guard_specific_inquiries": 0,
                        "learned_parameters": model["payload"]["training"]["learned_parameters"],
                        "model_file_bytes": (output/"guard-model.json").stat().st_size,
                        "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                        "boundary": "Includes prefix reconstruction, fitting, calibration, scoring and bootstrap. "
                                    "Excludes process startup, later replay/audit and research. "
                                    "Final outcomes never train any control; teaching outcomes are extra learned-guard cost."}
    if frozen_before_final != {"model_sha256": guard.artifact()["sha256"], "selections_sha256": digest(selections)}:
        raise ValueError("Model or thresholds changed during final evaluation")
    payload = {"records": records+final, "teaching": teaching}
    with (output/"records.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(canonical(payload).encode())
    summary["records_sha256"] = hashlib.sha256((output/"records.json.gz").read_bytes()).hexdigest()
    summary["records_archive_bytes"] = (output/"records.json.gz").stat().st_size
    write(output/"summary.json", summary)
    print(json.dumps({"status": "COMPLETE", "costs": summary["costs"], "gate": summary["gate"]}), flush=True)


def replay(run, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = json.loads((run/"protocol.json").read_text())
    config = validate_protocol(protocol)
    archived = json.loads(gzip.decompress((run/"records.json.gz").read_bytes()))
    expected = sum(config["worlds_per_family"][s]*len(TRAIN_FAMILIES) for s in
                   ("teach", "probability_calibration", "threshold_selection")) + config["worlds_per_family"]["final"]*len(FAMILIES)
    if len(archived["records"]) != expected:
        raise ValueError("Missing or unexpected world records")
    model = json.loads((run/"guard-model.json").read_text())
    guard = LearnedGuard(model)
    selections = json.loads((run/"thresholds.json").read_text())["selections"]
    for row in archived["records"]:
        posterior = Posterior.restore(row["posterior"])
        features, predictions = extract_features(posterior, QUERY_T, VARIANCE)
        if features.tolist() != row["features"] or [p["mean"] for p in predictions] != row["predicted_means"]:
            raise ValueError("Evidence/prediction/feature replay differs")
        if row["split"] == "final":
            if {k: v.tolist() for k, v in control_scores(features, guard).items()} != row["scores"]:
                raise ValueError("Guard or control prediction replay differs")
    refit = fit_guard(archived["teaching"]["teach"], archived["teaching"]["probability_calibration"],
                      optimizer_seed=config["optimizer_seed"], steps=config["training_steps"])
    if refit != model:
        raise ValueError("Exact guard retraining differs")
    threshold_rows = archived["teaching"]["threshold_selection"]
    scores = [control_scores(r["features"], guard) for r in threshold_rows]
    y = np.array([r["observed_valid"] for r in threshold_rows])
    replayed_selections = {method: select_threshold(np.array([s[method] for s in scores]), y)
                           for method in selections}
    if replayed_selections != selections:
        raise ValueError("Threshold calibration replay differs")
    final = [r for r in archived["records"] if r["split"] == "final"]
    summary = summarize(final, guard, selections)
    old = json.loads((run/"summary.json").read_text())
    if any(summary[k] != old[k] for k in summary):
        raise ValueError("Final metrics/curves/gates replay differs")
    write(output/"replay.json", {"status": "PASS", "worlds_replayed": expected,
                                 "query_predictions_replayed": expected*len(QUERY_T),
                                 "exact_model_refit": True, "exact_thresholds": True,
                                 "exact_summary_and_curves": True, "wall_seconds": time.perf_counter()-started,
                                 "source_protocol_sha256": protocol["sha256"]})


def test_execute_explicit_study():
    mode = os.environ.get("SERA_GUARD_MODE")
    if mode not in {"timing", "final", "replay"}:
        raise ValueError("Set explicit SERA_GUARD_MODE for this bounded experiment")
    output = Path(os.environ["SERA_GUARD_OUTPUT"]).resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Use a fresh run directory inside this repository")
    if mode == "replay":
        replay(Path(os.environ["SERA_GUARD_RUN"]).resolve(), output)
    else:
        execute(output, mode)


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
