"""Audit informative queries and learned support reuse from preserved evidence."""

import argparse
import gzip
import json
from collections import defaultdict

import numpy as np

from .audit import independent_matrix, independent_truth
from .audit_variants import check_lp, error, source_check
from .study import RELEASE, write


def statistics(rows):
    accepted = [r for r in rows if r["guard"]["accepted"]]
    return {"instances": len(rows), "solved": sum(r["capability"] for r in rows),
            "mean_error": float(np.mean([r["prediction_error"] for r in rows])),
            "accepted": len(accepted), "accepted_wrong": sum(not r["capability"] for r in accepted),
            "accepted_risk": sum(not r["capability"] for r in accepted)/max(1, len(accepted)),
            "mean_paid_scalars": float(np.mean([r["scalar_observations"] for r in rows])),
            "mean_work_proxy": float(np.mean([r["work_proxy"] for r in rows])),
            "worker_seconds": float(sum(r["worker_seconds"] for r in rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    folder = RELEASE/parser.parse_args().name
    protocol = source_check(folder)
    grouped, focus, paired = defaultdict(list), defaultdict(list), {}
    count, certificates = 0, 0
    with gzip.open(folder/"lifetimes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            artifact, spec = row["artifact"], row["world"]
            terms = artifact["terms"]
            assert terms == [["one"]]+[[op, 2*k] for k in range(1, 17) for op in ("cos", "sin")]
            support, select, cal, query = artifact["support"], artifact["selection"], row["calibration"], row["query"]
            sets = [set(item["t"]) for item in (support, select, cal, query)]
            assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
            np.testing.assert_allclose(support["t"][:12], np.arange(12)/12, atol=0, rtol=0)
            for i, item in enumerate(row["queries"]):
                assert support["t"][12+i] == item["phase"]
                assert support["y"][12+i] == item["value"]
            for item in (support, select, cal):
                expected = independent_truth(spec, terms, item["t"])
                assert np.max(abs(np.asarray(item["y"])-expected)) <= spec["nominal_noise_bound"]+1e-11
            weights = np.array(artifact["coefficients"])
            predicted = independent_matrix(terms, query["t"])@weights
            actual = independent_truth(spec, terms, query["t"])
            np.testing.assert_allclose(actual, query["truth"], atol=1e-11, rtol=0)
            np.testing.assert_allclose(predicted, query["prediction"], atol=1e-11, rtol=0)
            measured_error = error(predicted, actual)
            assert abs(measured_error-row["prediction_error"]) < 1e-10
            assert row["capability"] == (measured_error <= .05)
            a = independent_matrix(terms, support["t"])
            cal_error = error(independent_matrix(terms, cal["t"])@weights, np.asarray(cal["y"]))
            norm = np.linalg.norm(a, axis=0)
            normalized = a[:, norm > 1e-10]/norm[norm > 1e-10]
            corr = np.abs(np.triu(normalized.T@normalized, 1))
            ambiguous = bool((norm <= 1e-10).any() or (corr > 1-1e-9).any())
            assert row["guard"]["accepted"] == (cal_error <= .05 and not ambiguous)
            assert abs(cal_error-row["guard"]["relative_calibration_error"]) < 1e-10
            paid = 2*(len(support["t"])+len(select["t"])+len(cal["t"]))
            solves = 2+(16*len(row["queries"]) if row["policy"] == "disagreement" else 0)
            assert paid == row["scalar_observations"] and solves == row["linear_solves"]
            assert paid+.25*solves == row["work_proxy"]
            if row["regime"] == "equal_work_proxy":
                assert row["work_proxy"] <= 84
            if row["regime"] == "extended_challenge":
                assert len(cal["t"]) == 32
            certificates += check_lp(a, np.asarray(support["y"]), artifact["certificates"])
            pair = (spec["seed"], row["regime"])
            serialized = json.dumps([select, cal], sort_keys=True)
            if pair in paired:
                assert paired[pair] == serialized
            paired[pair] = serialized
            grouped[f"{row['regime']}/{spec['family']}/{row['policy']}"].append(row)
            focus[(row["regime"], row["policy"], "all")].append(row)
            if spec["family"] in ("sparse", "noisy"):
                focus[(row["regime"], row["policy"], "adequate")].append(row)
            count += 1
    gates = {}
    comparisons = {}
    for regime in ("equal_observations", "equal_work_proxy"):
        stats = {policy: statistics(focus[(regime, policy, "adequate")]) for policy in ("random", "space_filling", "disagreement")}
        challengers = stats["disagreement"]["mean_error"]
        gains = {policy: 1-challengers/max(stats[policy]["mean_error"], 1e-12) for policy in ("random", "space_filling")}
        comparisons[regime] = {"adequate": stats, "relative_improvement": gains}
        gates[regime+"_improvement"] = all(gain >= .1 for gain in gains.values())
        gates[regime+"_risk"] = statistics(focus[(regime, "disagreement", "all")])["accepted_risk"] <= .05
    prior_count, prior_groups = 0, defaultdict(list)
    terms = [["one"]]+[[op, 2*k] for k in range(1, 17) for op in ("cos", "sin")]
    with gzip.open(folder/"prior-reuse.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            old, current = row["original_observations"], row["support"]
            np.testing.assert_allclose(independent_truth(row["original_world"], terms, old["t"]), old["y"], atol=1e-11, rtol=0)
            recovered = np.array(row["acquired_coefficients"])
            np.testing.assert_allclose(independent_matrix(terms, old["t"])@recovered, old["y"], atol=1e-7, rtol=0)
            acquired = np.flatnonzero(abs(recovered).max(axis=1) > 1e-5).tolist()
            expected_prior = [] if row["method"] == "scratch" else (acquired if row["method"] == "acquired_prior" else sorted(set((i+13) % 33 for i in acquired)))
            assert row["prior"] == expected_prior
            assert row["previous_scalar_cost"] == 2*len(old["t"]) == 48
            assert row["new_scalar_cost"] == 2*len(current["t"]) == 2*row["count"]
            np.testing.assert_allclose(independent_truth(row["world"], terms, current["t"]), current["y"], atol=1e-11, rtol=0)
            a = independent_matrix(terms, current["t"])
            certificates += check_lp(a, np.array(current["y"]), row["certificates"], expected_prior)
            prediction = independent_matrix(terms, row["query"]["t"])@row["coefficients"]
            actual = independent_truth(row["world"], terms, row["query"]["t"])
            np.testing.assert_allclose(prediction, row["query"]["prediction"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual, row["query"]["truth"], atol=1e-10, rtol=0)
            assert abs(error(prediction, actual)-row["prediction_error"]) < 1e-10
            prior_groups[f"{row['count']}/{row['method']}"].append(row)
            prior_count += 1
    assert count == 36*protocol["config"]["seeds"] and prior_count == 6*protocol["config"]["seeds"]
    result = {"status": "PASS", "lifetimes": count, "prior_reconstructions": prior_count, "lp_certificates": certificates,
              "table": {key: statistics(rows) for key, rows in sorted(grouped.items())}, "comparisons": comparisons,
              "gates": gates, "investigator_decision": "QUALIFIED_CONDITIONAL" if all(gates.values()) else "REJECTED_FOR_PROMOTION",
              "priors": {key: {"instances": len(rows), "within_5_percent": sum(r["prediction_error"] <= .05 for r in rows),
                                "exact_at_1e_6": sum(r["prediction_error"] <= 1e-6 for r in rows),
                                "mean_error": float(np.mean([r["prediction_error"] for r in rows])),
                                "old_scalar_cost_each": 48, "new_scalar_cost_each": rows[0]["new_scalar_cost"]} for key, rows in prior_groups.items()},
              "extended_challenge_boundary": "32 space-filling challenges vs 12 random changes both observation count and design; cannot isolate their individual effects.",
              "scope": "Independently checked measurements, evidence separation, outputs, LP optimality, charged observations/work and acquired-prior provenance. Finite policies are engineered, not learned eta. No re-fit or adaptive final tuning."}
    write(folder/"independent-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("table", "comparisons", "priors")}))


if __name__ == "__main__":
    main()
