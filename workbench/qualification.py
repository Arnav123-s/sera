"""A fresh finite qualification of on-request fitting and deliberate non-coverage."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .storage import encoded
from .task_learning import learn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"research-continuation/20_live_workbench/ONDEMAND-001"


def main():
    OUTPUT.mkdir(exist_ok=False)
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), Path(__file__).with_name("task_learning.py"), Path(__file__).with_name("owner.py")]}
    protocol = {"seeds": list(range(145001, 145031)), "families": ["affine", "quadratic", "omitted_sinusoid"],
                "examples": 40, "queries": 9, "tolerance": 1e-6, "sources": sources,
                "gate": "All exact affine/quadratic task queries accurate to 1e-6; omitted sinusoid withheld; out-of-domain queries withheld.",
                "scope": "Numerical transformations with supplied polynomial family. Assessor coefficients never enter fitting. Separate finite mathematical tasks, not broad tutorial comprehension."}
    (OUTPUT/"protocol.json").write_bytes(encoded(protocol))
    results = []
    for seed in protocol["seeds"]:
        rng = np.random.default_rng(seed)
        a, b, c = rng.uniform(-2, 2, 3)
        x = np.linspace(-4, 4, 40)
        q = rng.uniform(-3.5, 3.5, 9)
        for family in protocol["families"]:
            function = (lambda z: b*z+c) if family == "affine" else ((lambda z: a*z*z+b*z+c) if family == "quadratic" else (lambda z: np.sin(3*z)+b*z))
            examples = "x,y\n"+"\n".join(f"{xx:.15g},{yy:.15g}" for xx, yy in zip(x, function(x)))
            result = learn("Fresh numerical task", q.tolist(), examples, 1e-6)
            prediction = np.polyval(result["coefficients"], q)
            error = float(np.max(abs(prediction-function(q))))
            expected = "WITHHELD" if family == "omitted_sinusoid" else "ACCEPTED"
            passed = result["status"] == expected and (expected == "WITHHELD" or error <= 1e-6)
            outside = learn("Out-of-domain query", [5.], examples, 1e-6)
            results.append({"seed": seed, "family": family, "expected_status": expected,
                            "result": result, "independent_query_max_error": error,
                            "out_of_domain_withheld": outside["status"] == "WITHHELD", "passed": passed})
    (OUTPUT/"results.json").write_bytes(encoded(results))
    summary = {"status": "PASS" if all(r["passed"] and r["out_of_domain_withheld"] for r in results) else "FAIL",
               "tasks": len(results), "accepted": sum(r["result"]["status"] == "ACCEPTED" for r in results),
               "withheld": sum(r["result"]["status"] == "WITHHELD" for r in results),
               "independently_checked_new_queries": 60*9, "out_of_domain_withheld": sum(r["out_of_domain_withheld"] for r in results),
               "maximum_accepted_query_error": max(r["independent_query_max_error"] for r in results if r["result"]["status"] == "ACCEPTED")}
    (OUTPUT/"summary.json").write_bytes(encoded(summary))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
