"""Independent evidence/weight/output audit of actual-owner maintenance trials."""

import argparse
import gzip
import json
from collections import defaultdict

import numpy as np

from .arrays import ArrayReader
from .audit_variants import check_lp, error, source_check
from .study import RELEASE, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    folder = RELEASE/parser.parse_args().name
    protocol = source_check(folder)
    evidence = json.loads((folder/"evidence.json").read_text())
    arrays = ArrayReader(folder/"arrays.zip")
    features = {k: arrays.get(v).astype(float) for k, v in evidence["features"].items()}
    logits = {k: [arrays.get(v) for v in items] for k, items in evidence["logits"].items()}
    labels = np.array([r["labels"] for r in evidence["prompts"]["evaluation"]])
    parent_labels = np.column_stack([value.argmax(axis=1) for value in logits["evaluation"]])
    allowed = {"support": {2, 3, 4}, "selection": {1}, "evaluation": {0}}
    for split, rows in evidence["prompts"].items():
        for row in rows:
            _, a, b, c = row["labels"]
            assert (121*a+11*b+c) % 5 in allowed[split]
    geometry = {}
    for count in protocol["counts"]:
        a = features["support"][:count]
        norms = np.linalg.norm(a, axis=0)
        normalized = a/np.maximum(norms, 1e-12)
        corr = np.abs(normalized.T@normalized)
        np.fill_diagonal(corr, 0)
        geometry[str(count)] = {"rank": int(np.linalg.matrix_rank(a)), "maximum_column_correlation": float(corr.max()),
                                "zero_columns": int((norms < 1e-12).sum()), "rip_certificate": False}
    groups, paired = defaultdict(list), {}
    certificates, records = 0, 0
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            a = features["support"][:row["count"]]
            q = features["evaluation"]
            pristine, fault, damaged, fitted, updated = [np.array(row[k]) for k in ("pristine", "fault", "damaged", "fitted_delta", "updated")]
            np.testing.assert_allclose(pristine+fault, damaged, atol=1e-13, rtol=0)
            np.testing.assert_array_equal((damaged+fitted).astype(np.float32).astype(float), updated)
            np.testing.assert_allclose(a@pristine+row["bias"], row["retained_responses"], atol=1e-11, rtol=0)
            np.testing.assert_allclose(-a@fault, row["target"], atol=1e-11, rtol=0)
            if row["family"] == "sparse4":
                assert np.count_nonzero(fault) == 4
            assert row["paid_anchor_scalars"] == row["count"]+32
            certificates += check_lp(a, np.array(row["target"])[:, None], row["certificates"])
            measured_error = error(q@(updated-damaged), -q@fault)
            assert abs(measured_error-row["prediction_error"]) < 1e-9
            residual_weight_error = float(np.sqrt(np.sum((updated-pristine)**2))/max(np.sqrt(np.sum(fault**2)), 1e-12))
            assert abs(residual_weight_error-row["coefficient_error"]) < 1e-9
            actual = arrays.get(row["evaluation_logits"])
            neuron = row["neuron"]
            np.testing.assert_allclose(actual[:, neuron], q@updated+row["bias"], atol=2e-5, rtol=0)
            others = [i for i in range(11) if i != neuron]
            np.testing.assert_allclose(actual[:, others], logits["evaluation"][1][:, others], atol=2e-5, rtol=0)
            predicted_labels = parent_labels.copy()
            predicted_labels[:, 1] = actual.argmax(axis=1)
            assert row["exact_slot_accuracy"] == float(np.mean((predicted_labels == labels).all(axis=1)))
            assert row["parent_agreement"] == float(np.mean((predicted_labels == parent_labels).all(axis=1)))
            assert row["parent_exact_slot_accuracy"] == float(np.mean((parent_labels == labels).all(axis=1)))
            pair = (row["seed"], row["family"], row["count"])
            shared = json.dumps([row[k] for k in ("retained_responses", "target", "damaged")])
            if pair in paired:
                assert paired[pair] == shared
            paired[pair] = shared
            groups[f"{row['family']}/{row['count']}/{row['method']}"].append(row)
            records += 1
    arrays.close()
    assert records == protocol["config"]["seeds"]*30
    table = {key: {"trials": len(rows), "within_5_percent": sum(r["prediction_error"] <= .05 for r in rows),
                   "mean_prediction_error": float(np.mean([r["prediction_error"] for r in rows])),
                   "mean_coefficient_error": float(np.mean([r["coefficient_error"] for r in rows])),
                   "mean_parent_agreement": float(np.mean([r["parent_agreement"] for r in rows])),
                   "mean_exact_slot_accuracy": float(np.mean([r["exact_slot_accuracy"] for r in rows])),
                   "parent_exact_slot_accuracy": rows[0]["parent_exact_slot_accuracy"],
                   "decoder_seconds": float(sum(r["decoder_seconds"] for r in rows))} for key, rows in sorted(groups.items())}
    result = {"status": "PASS", "records": records, "lp_certificates": certificates, "paired_settings": len(paired),
              "geometry": geometry, "table": table, "parent_owner": protocol["parent_owner"],
              "boundary": "Checks actual stored network-layer outputs and known fault ground truth independently. Trusted pre-fault logits are maintenance evidence, not new factual learning or a new language benchmark. Full recurrent execution checks are in the study receipt."}
    write(folder/"independent-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "table"}))


if __name__ == "__main__":
    main()
