import gzip
import json
from pathlib import Path

import pytest

from experiments.continuing_control.study import parent
from experiments.cross_route_transfer.study import evaluate_retention
from workbench.model import Learner
from workbench.storage import Store, encoded

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/"runs/sera-workbench/current.json").exists(), reason="Requires the trained local live workspace")
def test_live_owner_retains_neural_routes_on_fresh_cases():
    original = parent()
    record = Store(ROOT/"runs/sera-workbench").read()
    successor = Learner(record).solver
    identifiers = original.identity(), successor.identity()
    config = {"retention_samples": 128, "typed_samples": 128}
    before = evaluate_retention(original, 1200239, config)
    after = evaluate_retention(successor, 1200239, config)
    assert before["groups"] == after["groups"] and before["scores"] == after["scores"]
    assert identifiers == (original.identity(), successor.identity())
    output = ROOT/"research-continuation/20_live_workbench/retention"
    if not output.exists():
        output.mkdir()
        for name, data in (("before", before), ("after", after)):
            (output/(name+".json.gz")).write_bytes(gzip.compress(encoded(data), mtime=0))
        (output/"result.json").write_text(json.dumps({"status": "PASS", "seed": 1200239,
                    "groups": len(before["groups"]), "exact_group_and_case_scores": True,
                    "original_solver": identifiers[0], "successor_solver": identifiers[1],
                    "record_revision": record["revision"], "live_contexts": len(record["contexts"]),
                    "interaction_observations": len(record["interaction"]["events"]),
                    "scope": "Finite fresh bank; no claim of general transfer or plasticity."}, indent=2)+"\n")
