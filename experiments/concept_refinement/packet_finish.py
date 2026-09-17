"""Complete unexecuted checks; preserve strict portability failures as failures."""

import json
import os
import subprocess
import sys
import time

from .packet_verify import INTAKE, OUT


def differences(a, b, path=""):
    if type(a) is not type(b):
        return [{"path": path, "expected": a, "actual": b}]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [{"path": path, "keys_differ": True}]
        return [r for k in a for r in differences(a[k], b[k], path + "/" + k)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": path, "lengths_differ": True}]
        return [r for i, (x, y) in enumerate(zip(a, b)) for r in differences(x, y, path + f"/{i}")]
    if a != b:
        return [
            {
                "path": path,
                "expected": a,
                "actual": b,
                "absolute_difference": abs(a - b) if isinstance(a, (float, int)) else None,
            }
        ]
    return []


def main():
    folder = OUT / "packet-finish"
    folder.mkdir(exist_ok=False)
    sys.path.insert(0, str(INTAKE / "SERA_v13/code"))
    from conceptlab import physics, production

    result = {}
    for name, module in (("physics", physics), ("production", production)):
        actual = module.execute(save=False)
        expected = json.loads((INTAKE / "SERA_v13/results" / f"{name}.json").read_text())
        diff = differences(expected, actual)
        result[name] = {"original_exact_json_equality_pass": not diff, "differences": diff}
        (folder / f"{name}-local.json").write_text(json.dumps(actual, indent=2) + "\n")
    p = json.loads((folder / "physics-local.json").read_text())
    result["independent_physics_invariants"] = {
        "same_present_max": max(r["same_present_max_difference"] for r in p["bath_pairs"]),
        "future_distance_min": min(r["future_trace_distance"] for r in p["bath_pairs"]),
        "energy_balance_max": max(r["integrated_balance_error"] for r in p["energy_balance"]),
        "compression_error_max": max(
            r["maximum_recurrence_error"] for r in p["memory_compression"]
        ),
    }
    packet = INTAKE / "SERA_v12"
    receipts = []
    for name, args in (
        ("manifest-data", ["VERIFY.py", "--output", str(folder / "v12-data"), "--max-jobs", "1"]),
        ("body", ["code/verify_science.py", "body", "--output", str(folder / "v12-body.json")]),
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ):
        start = time.perf_counter()
        with (folder / f"v12-{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", *args],
                cwd=packet,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
        receipts.append(
            {"job": name, "returncode": proc.returncode, "seconds": time.perf_counter() - start}
        )
    result["v12"] = receipts
    (folder / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "v13": {
                    k: {
                        "strict_pass": v["original_exact_json_equality_pass"],
                        "differences": len(v["differences"]),
                    }
                    for k, v in result.items()
                    if "differences" in v
                },
                "invariants": result["independent_physics_invariants"],
                "v12": receipts,
            }
        )
    )
    if any(r["returncode"] for r in receipts):
        raise RuntimeError("Preserved v12 failure requires review")


if __name__ == "__main__":
    main()
