"""Fresh paid-query policies and partial-support reuse, separate from recovery finals."""

import argparse
import gzip
import json
import time
import zipfile

import numpy as np
from scipy.optimize import linprog

from .core import dictionary, fit, guard, l1, matrix, observations, predict, relative
from .study import RELEASE, ROOT, source_files, truth, world, write

POLICIES = ("random", "space_filling", "disagreement")
REGIMES = ("equal_observations", "equal_work_proxy", "extended_challenge")
FAMILIES = ("sparse", "noisy", "local_exception", "off_grid")


def observe(spec, t, rng):
    return truth(spec, np.atleast_1d(t))+rng.uniform(-spec["nominal_noise_bound"], spec["nominal_noise_bound"], (len(np.atleast_1d(t)), 2))


def alternatives(t, y, epsilon, seed):
    """A fixed ensemble of feasible weighted-L1, smaller Fourier and cubic models."""
    terms = dictionary(33)
    a, pool = matrix(terms, t), (np.arange(129)+.271)/129
    b = matrix(terms, pool)
    rng, candidates, work = np.random.default_rng(seed), [], 0
    ordinary, _ = l1(a, y, epsilon)
    work += 2
    candidates.append(b@ordinary)
    limit = float(abs(ordinary).sum())*1.1+1e-6
    for _ in range(6):
        penalty = rng.uniform(.97, 1.03, 33)
        coefficient = []
        for column in y.T:
            aa = np.c_[a, -a]
            constraints = {"A_eq": aa, "b_eq": column} if epsilon == 0 else {"A_ub": np.r_[aa, -aa], "b_ub": np.r_[column+epsilon, -column+epsilon]}
            solution = linprog(np.r_[penalty, penalty], bounds=(0, None), method="highs", options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}, **constraints)
            work += 1
            if not solution.success:
                raise ValueError("Alternative LP failed: "+solution.message)
            coefficient.append(solution.x[:33]-solution.x[33:])
        coefficient = np.column_stack(coefficient)
        if abs(coefficient).sum() <= limit:
            candidates.append(b@coefficient)
    # Two independently specified representations can join only when compatible
    # with the same paid observations. They do not observe hidden trajectories.
    for aa, bb in ((a[:, :17], b[:, :17]),
                   (np.column_stack([np.asarray(t)**k for k in range(4)]), np.column_stack([pool**k for k in range(4)]))):
        coefficient = np.linalg.lstsq(aa, y, rcond=None)[0]
        work += 1
        if np.max(abs(aa@coefficient-y)) <= epsilon+1e-7:
            candidates.append(bb@coefficient)
    return pool, np.asarray(candidates), work


def lifetime(spec, policy, regime):
    start = time.perf_counter()
    rng = np.random.default_rng(spec["seed"]+700001)
    phase = list(np.arange(12)/12)
    values = list(observe(spec, phase, rng))
    queries, fits, movement = [], 0, 0.
    challenges = 32 if regime == "extended_challenge" else 12
    # Proxy: one scalar observation=1; one linear/LP solve=.25. Actual time is
    # also measured. This is a declared work proxy, not FLOPs or equal runtime.
    reserve = 2*(8+challenges)+.5
    cap = 84. if regime == "equal_work_proxy" else float("inf")
    for step in range(8):
        policy_fits = 16 if policy == "disagreement" else 0
        if 2*(len(phase)+1)+.25*(fits+policy_fits)+reserve > cap:
            break
        candidates = 0
        if policy == "random":
            coordinate = float(rng.uniform(0, 1))
        else:
            pool = (np.arange(129)+.271)/129
            distance = abs(pool[:, None]-np.asarray(phase)[None]).min(axis=1)
            if policy == "space_filling":
                score = distance
            else:
                pool, ensemble, used = alternatives(phase, np.asarray(values), spec["nominal_noise_bound"], spec["seed"]+step*97)
                assert used == policy_fits
                fits += used
                candidates = len(ensemble)
                score = np.var(ensemble, axis=0).sum(axis=1)
                if float(score.max()) < 1e-12:
                    score = distance
            score[distance < 1e-8] = -1
            coordinate = float(pool[score.argmax()])
        measured = observe(spec, [coordinate], rng)[0]
        movement += abs(coordinate-phase[-1])
        phase.append(coordinate)
        values.append(measured)
        queries.append({"phase": coordinate, "value": measured.tolist(), "compatible_candidates": candidates})
    # Selection and challenge coordinates are paired across policies and never
    # available to the query selector, even as unlabeled future targets.
    held = np.random.default_rng(spec["seed"]+900001)
    select_t = held.uniform(0, 1, 8)
    cal_t = (np.arange(32)+.5)/32 if challenges == 32 else held.uniform(0, 1, 12)
    selection = observations(select_t, observe(spec, select_t, held), "selection")
    calibration = observations(cal_t, observe(spec, cal_t, held), "calibration")
    artifact = fit("basis_pursuit", observations(phase, values, "support"), selection, dictionary(33), spec["nominal_noise_bound"])
    fits += 2
    checked = guard(artifact, calibration)
    q = (np.arange(256)+.413)/256
    predicted, actual = predict(artifact, q), truth(spec, q)
    error = relative(predicted, actual)
    paid = 2*(len(phase)+8+challenges)
    cost = paid+.25*fits
    assert cost <= cap+1e-9
    return {"world": spec, "policy": policy, "regime": regime, "queries": queries, "artifact": artifact,
            "calibration": {"t": calibration.t.tolist(), "y": calibration.y.tolist()}, "guard": checked,
            "prediction_error": error, "capability": error <= .05, "scalar_observations": paid,
            "linear_solves": fits, "work_proxy": cost, "phase_movement": movement,
            "worker_seconds": time.perf_counter()-start,
            "query": {"t": q.tolist(), "prediction": predicted.tolist(), "truth": actual.tolist()}}


def support_reuse(seed):
    """Actual observed old support, a changed successor and a wrong-prior control."""
    rng = np.random.default_rng(seed)
    first = world(33, "sparse", seed)
    terms = dictionary(33)
    t = rng.uniform(0, 1, 24)
    y = truth(first, t)
    recovered, _ = l1(matrix(terms, t), y, 0)
    acquired = np.flatnonzero(np.max(abs(recovered), axis=1) > 1e-5).tolist()
    wrong = sorted(set((i+13) % 33 for i in acquired))
    second = dict(first)
    weights = np.asarray(first["weights"]).copy()
    previous = np.flatnonzero(abs(weights).sum(axis=1) > 0)
    replacement = int(rng.choice(np.flatnonzero(abs(weights).sum(axis=1) == 0)))
    weights[replacement] = weights[previous[0]]*rng.uniform(.8, 1.2, 2)
    weights[previous[0]] = 0
    second["weights"] = weights.tolist()
    records = []
    q = (np.arange(256)+.193)/256
    for count in (8, 12):
        phase = rng.uniform(0, 1, count)
        values = truth(second, phase)
        for method, prior in (("scratch", []), ("acquired_prior", acquired), ("wrong_prior", wrong)):
            start = time.perf_counter()
            coefficient, cert = l1(matrix(terms, phase), values, 0, prior=prior)
            prediction, actual = matrix(terms, q)@coefficient, truth(second, q)
            records.append({"seed": seed, "count": count, "method": method, "prior": prior,
                            "original_world": first, "world": second,
                            "original_observations": {"t": t.tolist(), "y": y.tolist()}, "acquired_coefficients": recovered.tolist(),
                            "support": {"t": phase.tolist(), "y": values.tolist()}, "coefficients": coefficient.tolist(),
                            "coefficients_error": relative(coefficient, weights), "prediction_error": relative(prediction, actual),
                            "certificates": cert, "previous_scalar_cost": 48, "new_scalar_cost": 2*count,
                            "worker_seconds": time.perf_counter()-start,
                            "query": {"t": q.tolist(), "prediction": prediction.tolist(), "truth": actual.tolist()}})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    args = parser.parse_args()
    folder = RELEASE/args.name
    folder.mkdir(exist_ok=False)
    protocol = {"id": args.name, "config": vars(args), "sources": source_files(),
                "policies": POLICIES, "families": FAMILIES, "regimes": REGIMES,
                "initial_phases": 12, "maximum_new_queries": 8, "selection_phases": 8,
                "ordinary_challenges": 12, "extended_challenges": 32,
                "work_proxy": "One scalar value costs 1; each LP or least-squares solve costs .25. Equal-work arm cap=84, reserved validation included; runtime reported separately.",
                "primary": "Prediction capability at relative error<=.05 and accepted error/coverage, paired by underlying world. No final-dependent thresholds or query changes.",
                "investigator_gate": "Disagreement must reduce mean relative error at least 10% versus BOTH random and space filling in sparse/noisy lifetimes under BOTH equal-observation and equal-work regimes; accepted error <=5% in each regime across all four families. Otherwise reject investigator promotion.",
                "prior_reuse": "Each prior comes from 24 observed phases of an actual earlier sparse situation. Change one active component; compare scratch, acquired support and an index-shifted wrong prior at 8/12 new phases. Equal old acquisition cost charged to all arms. No free true support.",
                "boundary": "Fresh instances, supplied coordinates/dictionaries/sensor bound and fixed policies. Alternative compatibility does not establish whole-dictionary adequacy. No learned eta, causal-control or shared-core benefit claim."}
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["sources"]:
            archive.write(ROOT/name, name)
    rows, start = [], time.perf_counter()
    with gzip.open(folder/"lifetimes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            for family_index, family in enumerate(FAMILIES):
                spec = world(33, family, args.start_seed+index*101+family_index*10007)
                for regime in REGIMES:
                    for policy in POLICIES:
                        row = lifetime(spec, policy, regime)
                        stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                        rows.append({k: row[k] for k in ("prediction_error", "capability", "guard", "scalar_observations", "work_proxy", "worker_seconds", "policy", "regime")} | {"family": family, "seed": spec["seed"]})
                stream.flush()
                write(folder/"progress.json", {"completed_lifetimes": len(rows), "last_world": spec["seed"]})
    priors = []
    with gzip.open(folder/"prior-reuse.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            for row in support_reuse(args.start_seed+800003+index*101):
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                priors.append({k: row[k] for k in ("seed", "count", "method", "coefficients_error", "prediction_error", "previous_scalar_cost", "new_scalar_cost", "worker_seconds")})
    write(folder/"summary.json", {"status": "COMPLETE", "lifetimes": len(rows), "prior_reconstructions": len(priors),
                                  "rows": rows, "priors": priors, "worker_seconds": time.perf_counter()-start})
    print(json.dumps({"status": "COMPLETE", "lifetimes": len(rows), "prior_reconstructions": len(priors), "seconds": time.perf_counter()-start}))


if __name__ == "__main__":
    main()
