import json

import numpy as np

from experiments.audit_compatibility.transfer import native_booleans


def test_numpy_gate_booleans_keep_their_exact_meaning_in_json():
    original = {"pass": np.bool_(True), "fail": [np.bool_(False)], "score": .0125}
    converted = json.loads(json.dumps(native_booleans(original)))
    assert converted == {"pass": True, "fail": [False], "score": .0125}
    assert isinstance(original["pass"], np.bool_)
