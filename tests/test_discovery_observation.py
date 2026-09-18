import csv
import io

import numpy as np
import pytest

from experiments.discovery_observation import fit_and_check, measurement_rows, point


def acceleration_route():
    return {"terms": [[["x", 1], ["t", -2]], [["x0", 1], ["t", -2]], [["v0", 1], ["t", -1]]],
            "coefficients": ["2", "-2", "-2"]}


def test_generic_rearrangement_uses_discovered_coefficients():
    assert abs(point(acceleration_route(), 2., [1., 3., -2.])-3.) < 1e-12


def test_observation_fixture_keeps_heldout_measurements_out_of_fit():
    # This synthetic fixture tests plumbing only. It is not learner data or
    # evidence for the separately attributed physical observation experiment.
    def raw(changed=False):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Time (s)", "Position (m)"])
        for i in range(93):
            t = .05*(i+1)
            writer.writerow([t, 1+3*t-t*t+(1 if changed and 26 <= i < 36 else 0)])
        return output.getvalue().encode()
    first = fit_and_check(acceleration_route(), raw())
    changed = fit_and_check(acceleration_route(), raw(True))
    assert first["accepted"]
    assert first["nuisance_parameters"] == changed["nuisance_parameters"]
    assert not changed["accepted"]
    assert np.isclose(changed["held_out_rmse_metres"], 1.)


def test_unlabeled_sequential_row_index_does_not_shift_measured_columns():
    rows, indexed = measurement_rows(b'"Time (s)","Position (m)"\n0,0.05,1.478452\n1,0.1,1.478452\n')
    assert indexed
    assert rows == [{"Time (s)": "0.05", "Position (m)": "1.478452"},
                    {"Time (s)": "0.1", "Position (m)": "1.478452"}]
    with pytest.raises(ValueError, match="sequential"):
        measurement_rows(b'"Time (s)","Position (m)"\n9,0.05,1.478452\n')
