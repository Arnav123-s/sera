"""GG-GUARD-002: actual frozen SERA guard, independent-world policy calibration."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.generative_memory.acquisition import Action, Posterior, observed_event
from experiments.generative_memory.applicability import LearnedGuard, extract_features
from experiments.generative_memory.applicability_study import (
    QUERY_T,
    SUPPORT_INDICES,
    SUPPORT_T,
    TRAIN_FAMILIES,
    VARIANCE,
    assessor,
    rng_for,
)
from experiments.generative_memory.core import canonical, digest

from .risk import PolicyContract, WorldCertificate, WorldOutcome, calibrate, upper_bound

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/16_v3"
GUARD = ROOT/"research-continuation/15_applicability/guard-model.json"
GROUPS = ("consistent_near", "consistent_edge", "contradicted_near", "contradicted_edge")
THRESHOLDS = (.9, .925, .95, .975, .99)
PANELS = ("matched", "mechanism_mixture_shift", "abrupt_jump", "decaying_spiral")
METHODS = ("always", "abstain", "old_empirical", "minimum_90", "fixed_residual", "global_certificate", "group_certificate")
BASES = {"timing": 32000000, "calibration": 42000000, "matched": 52000000,
         "mechanism_mixture_shift": 62000000, "abrupt_jump": 72000000, "decaying_spiral": 82000000}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def sources():
    files = ["experiments/world_calibration/"+n for n in ("__init__.py", "risk.py", "study.py", "audit.py")]
    files += ["experiments/generative_memory/"+n for n in ("core.py", "acquisition.py", "applicability.py", "applicability_study.py")]
    files += ["experiments/generative_memory/applicability_audit.py"]
    files += ["scripts/run_v3_bounded.py", "scripts/windows_job_v3.py", "tests/test_world_calibration.py"]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/"src/sera").glob("*.py")]
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}


def feature_group(x):
    if np.asarray(x).shape != (16,) or not np.isfinite(x).all():
        raise ValueError("Complete finite guard features required")
    residual = "consistent" if x[9] <= np.log1p(4.) else "contradicted"
    support = "near" if x[2] <= .18 and x[3] == 0 else "edge"
    return residual+"_"+support


def policy_contract(model, grouped):
    ids = {
        "predictor": model["sha256"], "features": digest(model["payload"]["contract"]),
        "groups": digest({"rule": "qmean<=4; distance<=.18 and outside==0" if grouped else "all"}),
        "query_selector": digest({"rule": "one uniformly sampled QUERY_T index per independent world", "grid": QUERY_T.tolist()}),
        "acquisition": digest({"support": SUPPORT_T.tolist(), "indices": list(SUPPORT_INDICES), "variance": VARIANCE, "extra": 0}),
        "loss": digest({"metric": "observed Euclidean error", "radius": .2}),
        "interpreter": digest({"sources": sources(), "runtime": runtime()}),
        "population": digest({"family_distribution": {f: 1/6 for f in TRAIN_FAMILIES}, "assessor": "original generator instance distribution"}),
    }
    # Attempt 0 reserves .025 of a .05 sequential lifetime budget. Two prespecified
    # certificate policies share it; each further splits across groups/thresholds.
    return PolicyContract(tuple(sorted(ids.items())), GROUPS if grouped else ("all",), THRESHOLDS, alpha=.0125)


def private_world(seed, family):
    if family in TRAIN_FAMILIES:
        return assessor(seed, family)
    rng = rng_for(seed, "new-mechanism-v3")
    center = rng.normal(0, .5, 2)
    phase = float(rng.uniform(-np.pi, np.pi))
    amplitude = float(rng.uniform(.5, 1.5))
    rate = float(rng.uniform(.25, .8))
    switch = float(rng.uniform(-.65, .65))
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)

    def clean(t):
        t = np.asarray(t)
        angle = np.pi*t+phase
        y = center+amplitude*np.stack([np.cos(angle), np.sin(angle)], axis=1)
        if family == "abrupt_jump":
            y += (t >= switch)[:, None]*amplitude*direction
        elif family == "decaying_spiral":
            y = center + (amplitude*np.exp(-rate*(t+1.5)))[:, None]*np.stack([np.cos(angle), np.sin(angle)], axis=1)
        else:
            raise ValueError("Unknown protected mechanism")
        return y

    a, b = clean(SUPPORT_T), clean(QUERY_T)
    return {"family": family, "environment_seed": seed,
            "parameters": {"center": center.tolist(), "phase": phase, "amplitude": amplitude,
                           "rate": rate, "switch": switch, "direction": direction.tolist()},
            "support_observed": (a+.05*rng_for(seed, "support_measurement").normal(size=a.shape)).tolist(),
            "query_truth": b.tolist(),
            "query_observed": (b+.05*rng_for(seed, "query_measurement").normal(size=b.shape)).tolist()}


def episode(panel, index, guard):
    if panel not in BASES or not 0 <= index < 1000000:
        raise ValueError("Invalid bounded world identity")
    seed = BASES[panel]+index
    # Choose the query before receiving its answer. The RNG stream is separate
    # from mechanism, measurement noise and family selection.
    query_index = int(rng_for(seed, "query-selector-v3").integers(len(QUERY_T)))
    if panel in {"abrupt_jump", "decaying_spiral"}:
        family = panel
    elif panel == "mechanism_mixture_shift":
        family = str(rng_for(seed, "family-v3").choice(TRAIN_FAMILIES, p=[.0625]*4+[.5, .25]))
    else:
        family = str(rng_for(seed, "family-v3").choice(TRAIN_FAMILIES))
    world = private_world(seed, family)
    identity = digest({"experiment": "GG-GUARD-002", "seed": seed})
    actions = tuple(Action(i, float(t), "target", VARIANCE) for i, t in enumerate(SUPPORT_T))
    posterior = Posterior(actions, max_observations=10)
    for j, action_id in enumerate(SUPPORT_INDICES):
        posterior.observe(observed_event(actions[action_id], world["support_observed"][action_id], j,
                                         f"{identity}/support/{j}", "GG-GUARD-002/sensor"))
    x, predictions = extract_features(posterior, [QUERY_T[query_index]], VARIANCE)
    score = float(guard.predict(x)[0])
    mean = np.array(predictions[0]["mean"])
    truth, observed = world["query_truth"][query_index], world["query_observed"][query_index]
    error = int(np.linalg.norm(mean-observed) > .2)
    witness = {"world_id": identity, "query_index": query_index,
               "prediction_before_outcome": mean.tolist(), "observed_outcome": observed,
               "prefix_sha256": posterior.snapshot()["sha256"], "origin": "simulator_observation"}
    return {"world_id": identity, "panel": panel, "seed": seed, "private_family": family,
            "query_index": query_index, "query_t": float(QUERY_T[query_index]),
            "features": x[0].tolist(), "group": feature_group(x[0]), "score": score,
            "error": error, "clean_mse": float(np.mean((mean-truth)**2)),
            "clean_truth": truth, "posterior": posterior.snapshot(), "witness": witness,
            "witness_sha256": digest(witness)}


def outcome(row, grouped):
    if row["panel"] != "calibration":
        raise ValueError("Assessment worlds cannot become calibration observations")
    return WorldOutcome(row["world_id"], row["group"] if grouped else "all", row["score"],
                        row["error"], row["witness_sha256"])


def decisions(row, certificates, contracts, empirical_threshold):
    x = np.array(row["features"])
    group_answer = certificates["group"].decide(row["group"], row["score"], contracts["group"])
    global_answer = certificates["global"].decide("all", row["score"], contracts["global"])
    return {"always": True, "abstain": False, "old_empirical": row["score"] >= empirical_threshold,
            "minimum_90": row["score"] >= .9,
            "fixed_residual": bool(np.expm1(x[10]) <= 9.210340372 and np.expm1(x[4]) <= .04),
            "global_certificate": global_answer["accepted"], "group_certificate": group_answer["accepted"]}


def metrics(rows, method):
    accepted = [r for r in rows if r["decisions"][method]]
    n, k = len(accepted), sum(r["error"] for r in accepted)
    return {"worlds": len(rows), "accepted": n, "errors": k,
            "coverage": n/len(rows) if rows else None,
            "selective_risk": k/n if n else None,
            "risk_upper_95": upper_bound(k, n, .05),
            "accepted_clean_mse": float(np.mean([r["clean_mse"] for r in accepted])) if n else None,
            "utility": float(np.mean([int(r["decisions"][method])*(1-5*r["error"]) for r in rows])) if rows else None}


def summarize(rows):
    result = {"panels": {}, "groups": {}, "families": {}}
    for panel in PANELS:
        subset = [r for r in rows if r["panel"] == panel]
        result["panels"][panel] = {m: metrics(subset, m) for m in METHODS}
        result["groups"][panel] = {g: metrics([r for r in subset if r["group"] == g], "group_certificate") for g in GROUPS}
        result["families"][panel] = {f: metrics([r for r in subset if r["private_family"] == f], "group_certificate")
                                      for f in sorted({r["private_family"] for r in subset})}
    matched = [r for r in rows if r["panel"] == "matched"]
    ordinary = [r for r in matched if r["private_family"] in TRAIN_FAMILIES[:4]]
    score = np.array([(int(r["decisions"]["group_certificate"])-int(r["decisions"]["minimum_90"]))*(1-5*r["error"]) for r in matched])
    bootstrap = np.random.default_rng(991204).choice(score, size=(2000, len(score)), replace=True).mean(axis=1)
    noninferiority = float(np.quantile(bootstrap, .025)) >= -.05
    active_groups = [r for r in result["groups"]["matched"].values() if r["accepted"]]
    matched_bound = bool(active_groups) and all(upper_bound(r["errors"], r["accepted"], .05/4) <= .1 for r in active_groups)
    future_bound = all(upper_bound(result["panels"][p]["group_certificate"]["errors"],
                                   result["panels"][p]["group_certificate"]["accepted"], .025) <= .1
                       for p in ("abrupt_jump", "decaying_spiral")
                       if result["panels"][p]["group_certificate"]["accepted"])
    coverage = metrics(ordinary, "group_certificate")["coverage"]
    result["paired_utility_vs_minimum90"] = {"difference": float(score.mean()), "bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist()}
    result["gate"] = {"matched_active_group_risk_upper_at_most_10pct": matched_bound,
                      "in_menu_coverage_at_least_50pct": coverage >= .5,
                      "matched_utility_noninferiority_margin_005": noninferiority,
                      "new_family_accepted_risk_bounds_at_most_10pct": future_bound,
                      "prospectively_qualified": matched_bound and coverage >= .5 and noninferiority and future_bound,
                      "scope": "A rejected new-family bound prevents operational promotion. No-answer families do not establish knowledge."}
    return result


def load_protocol():
    record = json.loads((RELEASE/"GG-GUARD-002/protocol.json").read_text())
    p = record["payload"]
    if digest(p) != record["sha256"] or p["sources"] != sources() or p["runtime"] != runtime():
        raise ValueError("Frozen protocol, source or runtime changed")
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != p["guard_file_sha256"]:
        raise ValueError("Frozen predecessor changed")
    return record


def run(output, timing=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start, cpu = time.perf_counter(), time.process_time()
    model = json.loads(GUARD.read_text())
    guard = LearnedGuard(model)
    if timing:
        rows = [episode("timing", i, guard) for i in range(12)]
        # No calibration or protected-family outcomes are drawn by this fixture.
        write(output/"timing.json", {"worlds": len(rows), "wall_seconds": time.perf_counter()-start,
                                    "cpu_seconds": time.process_time()-cpu, "source_hashes": sources(), "runtime": runtime()})
        return
    protocol = load_protocol()
    write(output/"protocol.json", protocol)
    p = protocol["payload"]
    contracts = {key: policy_contract(model, grouped) for key, grouped in (("global", False), ("group", True))}
    calibration = [episode("calibration", i, guard) for i in range(p["counts"]["calibration"])]
    forbidden = set(model["payload"]["training"]["train_episode_ids"]+model["payload"]["training"]["calibration_episode_ids"])
    artifacts = {key: calibrate([outcome(r, key == "group") for r in calibration], contract, forbidden_worlds=forbidden)
                 for key, contract in contracts.items()}
    certificates = {key: WorldCertificate(artifacts[key], contract) for key, contract in contracts.items()}
    write(output/"certificates.json", artifacts)
    before = {"guard": model["sha256"], "certificates": {k: v["sha256"] for k, v in artifacts.items()}}
    write(output/"before-final.json", before)
    print(f"Calibration complete: {len(calibration)} independent worlds", flush=True)
    threshold = json.loads((ROOT/"research-continuation/15_applicability/thresholds.json").read_text())["selections"]["learned"]["selected"]["threshold"]
    rows = []
    for panel in PANELS:
        for i in range(p["counts"][panel]):
            row = episode(panel, i, guard)
            row["decisions"] = decisions(row, certificates, contracts, threshold)
            rows.append(row)
        print(f"Assessment complete: {panel}", flush=True)
    all_rows = calibration+rows
    if len({r["world_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("World partitions overlap")
    summary = summarize(rows)
    summary["protocol_sha256"] = protocol["sha256"]
    summary["cost"] = {"worlds": len(all_rows), "calibration_worlds": len(calibration), "assessment_worlds": len(rows),
                       "acquired_prefix_observations": len(all_rows)*10, "observed_query_outcomes": len(all_rows),
                       "new_neural_training_steps": 0, "extra_inquiries": 0,
                       "wall_seconds": time.perf_counter()-start, "cpu_seconds": time.process_time()-cpu,
                       "boundary": "Includes actual guard inference, all prefix replays, certificate fitting and scoring; archive serialization/startup are in supervisor time. Predecessor training is inherited cost, not rerun."}
    if before != {"guard": guard.artifact()["sha256"], "certificates": {k: v.record()["sha256"] for k, v in certificates.items()}}:
        raise ValueError("Frozen model/certificates changed during assessment")
    raw = canonical({"calibration": calibration, "assessment": rows}).encode()
    (output/"worlds.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    summary["worlds_sha256"] = hashlib.sha256((output/"worlds.json.gz").read_bytes()).hexdigest()
    write(output/"summary.json", summary)
    print(json.dumps({"cost": summary["cost"], "gate": summary["gate"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    run(args.output, args.timing)
