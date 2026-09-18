from experiments.solution_credit import known_execution
from experiments.solution_search import canonical


def test_native_forward_and_constant_force_routes_are_not_new():
    for domain, target, names in (
            ("motion_0", "f", ["a0", "m"]), ("motion_0", "a0", ["f", "m"]),
            ("motion_0", "v", ["f", "m", "t", "v0"]),
            ("motion_1", "f", ["a0", "a1", "m", "t"]),
            ("motion_2", "v", ["a0", "a1", "a2", "v0", "t"]),
            ("polynomials", "i", ["c0", "c1", "c2", "t"])):
        assert known_execution({"domain": domain, "target": target, "requires": names})


def test_new_inverse_routes_are_preserved():
    for domain, target, names in (
            ("motion_1", "m", ["a1", "f", "t", "v", "v0"]),
            ("polynomials", "c1", ["c0", "i", "p", "t"]),
            ("motion_2", "a0", ["f", "m"])):
        assert not known_execution({"domain": domain, "target": target, "requires": names})


def test_additional_inputs_do_not_farm_coverage_credit():
    from experiments.solution_credit import dominates
    q = {"domain": "polynomials", "target": "c2", "missing": []}
    a = canonical(q, {"terms": [[["p", 1]], [["c0", 1]], [["c1", 1], ["t", 1]]], "weights": ["1", "-1", "-1"]},
                  {"terms": [[["t", 2]]], "weights": ["1"]}, "a", 3)
    # The relation can use an additional integral input without gaining input coverage.
    b = canonical(q, {"terms": [[["p", 1], ["t", 1]], [["i", 1]], [["c1", 1], ["t", 2]]], "weights": ["3/2", "-3/2", "-3/4"]},
                  {"terms": [[["t", 3]]], "weights": ["1"]}, "b", 4)
    assert not dominates(a, b)  # Different required evidence: c0 versus i.
    bigger = {**a, "id": "different", "requires": sorted(a["requires"] + ["i"])}
    assert dominates(a, bigger)


def test_distinct_denominator_assumptions_are_preserved():
    from experiments.solution_credit import dominates
    q = {"domain": "motion_0", "target": "m", "missing": []}
    a = canonical(q, {"terms": [[["f", 1]]], "weights": ["1"]},
                  {"terms": [[["a0", 1]]], "weights": ["1"]}, "a", 2)
    other = {**a, "id": "other", "requires": a["requires"] + ["v"],
             "denominator": {"terms": [[["v", 1]]], "weights": ["1"]}}
    assert not dominates(a, other)
