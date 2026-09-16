from pathlib import Path

import numpy as np
import pytest

from experiments.update_learning.study import Eta, future_block
from workbench.model import Learner, transact
from workbench.storage import Store, digest
from workbench.task_learning import learn


def test_source_retrieval_fits_new_coefficients_with_separate_test_examples():
    result = learn("Convert Celsius to Fahrenheit", [0, 20, 37, 100])
    assert result["status"] == "ACCEPTED"
    assert np.allclose(result["coefficients"], [0, 1.8, 32])
    a, b, c = map(set, result["split"].values())
    assert not (a & b or a & c or b & c)
    assert result["material"]["evidence"].startswith("Examples computed")
    with pytest.raises(ValueError, match="No unambiguous"):
        learn("repair my computer", [1, 2])


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/"runs/A06-integrated-001/solver/versions/v0.pt").exists(), reason="Requires the preserved local trained parent")
def test_task_executes_from_saved_owner_and_is_recoverable(tmp_path):
    content = "x,y\n"+"\n".join(f"{x},{2*x*x-3*x+1}" for x in range(-12, 13))
    result = transact(tmp_path, {"request_id": "new-calibration", "operation": "learn_task", "task": "Calibrate sensor",
                                 "examples": content, "queries": [-1, 0, 4.5], "tolerance": 1e-8})
    assert np.allclose(result["result"]["outputs"], [6, 1, 28])
    record = Store(tmp_path).read()
    restored = Learner(record)
    context = result["result"]["context"]
    assert np.allclose(restored.session.owner.transform_context(context, [-1, 0, 4.5]), [6, 1, 28])
    assert restored.solve(4, 5, 6)["answer"] == 0


def test_unsupported_mapping_and_extrapolation_are_withheld():
    nonlinear = "x,y\n"+"\n".join(f"{x},{np.sin(x)}" for x in range(25))
    result = learn("unknown relation", [3, 4], nonlinear, .001)
    assert result["status"] == "WITHHELD" and result["outputs"] == []
    assert learn("celsius to fahrenheit", [999])["status"] == "WITHHELD"


def test_successor_inherits_actual_knowledge_and_independently_updated_eta():
    first, successor = future_block(list(np.linspace(0, 2, 24)), list(np.linspace(2, 4, 24)), Eta())
    second, _ = future_block(first["K_after"], list(np.linspace(4, 6, 24)), successor)
    assert second["K_before_sha256"] == digest(first["K_after"])
    assert second["eta_before"] == first["eta_after"]
    assert successor.examples == 24*6
    assert len(second["K_after"]) == 72
