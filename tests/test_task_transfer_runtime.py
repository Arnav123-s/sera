import json

import pytest
import torch

from experiments.task_transfer.runtime import (
    TaskReadout,
    load,
    lock,
    name_check,
    read_csv,
    require_fresh_calibration,
)


def test_task_readout_stores_actual_owned_parameter():
    layer = TaskReadout([2.]+[0.]*19)
    assert "weight" in dict(layer.named_parameters())
    assert layer([[0, 0, 0]]).item() == 2
    with torch.no_grad():
        layer.weight[0] = 3
    assert layer([[0, 0, 0]]).item() == 3
    with pytest.raises(ValueError):
        TaskReadout([float("nan")]*20)


def test_exclusive_lock_preserves_other_owners_and_cleans_own(tmp_path):
    with lock(tmp_path):
        with pytest.raises(FileExistsError):
            with lock(tmp_path):
                pass
        assert (tmp_path/"task.lock").exists()
    assert not (tmp_path/"task.lock").exists()


def test_csv_roles_shape_and_pathlike_names_rejected(tmp_path):
    path = tmp_path/"examples.csv"
    path.write_text("x1,x2,x3,y,role\n0,0,0,1,fit\n.1,0,0,2,selection\n.2,0,0,3,calibration\n")
    batches = read_csv(path, "teach")
    assert set(batches) == {"fit", "selection", "calibration"}
    path.write_text("x1,x2,x3\n0,0,0,extra\n")
    with pytest.raises(ValueError):
        read_csv(path, "predict")
    with pytest.raises(ValueError):
        name_check("../escape")
    with pytest.raises(ValueError):
        name_check(None)
    assert json.dumps(batches)


def test_checkpoint_rejects_corruption_before_loading_parent(tmp_path):
    (tmp_path/"revisions").mkdir()
    name = "000001-"+"a"*12+".json"
    (tmp_path/"revisions"/name).write_text("{}")
    (tmp_path/"current.json").write_text(json.dumps({"revision": name, "sha256": "b"*64}))
    with pytest.raises(ValueError, match="integrity mismatch"):
        load(tmp_path)
    (tmp_path/"current.json").write_text(json.dumps({"revision": "../parent.json"}))
    with pytest.raises(ValueError, match="revision name"):
        load(tmp_path)


def test_renaming_task_or_reusing_observation_does_not_make_calibration_fresh():
    calibration = {"x": [[.1, .2, .3]]}
    for operation in ("teach", "observe"):
        event = {"operation": operation, "task": "different_name", "used_inputs": [[.1, .2, .3]]}
        with pytest.raises(ValueError, match="whole session"):
            require_fresh_calibration([event], calibration)
    require_fresh_calibration([{"used_inputs": [[0, 0, 0]]}], calibration)
