import numpy as np
import torch

from experiments.acquisition_dependence.study import load_before
from experiments.cross_route_transfer.data import admit_training, teacher_identity
from experiments.cross_route_transfer.study import parent
from experiments.instance_transfer.study import bank


def test_raw_coordinate_teacher_keeps_the_actual_acquisition_intervention():
    torch.set_num_threads(1)
    after = parent().components["r1"]
    before, _ = load_before()
    data, effect = bank(after, before, 281, cfg={"observed": 8, "dreams": 16, "development": 8}, final=False)
    assert effect > .01
    assert not np.allclose([r.target for r in data["corrected"]], [r.target for r in data["uncorrected"]])
    admit_training(data["corrected"], allow_conditional=True, teacher=teacher_identity(after))
    admit_training(data["uncorrected"], allow_conditional=True, teacher=teacher_identity(before))
    assert all(len(r.observations) == 3 and all(o.position in (0, 1, 2) for o in r.observations) for r in data["observed"])
