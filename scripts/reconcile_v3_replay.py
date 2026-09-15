"""Separate compatibility diagnosis; never changes the original strict replay.

The strict calibration hash failed previously at float rounding. Only that check
is substituted here: all non-bound payload values must remain byte-equivalent,
each bound must differ by <1e-12, and both original hashes remain verified.
Alias replay is first attempted strictly; an affected-algebra diagnosis is used
only if exact NumPy equality fails. All decisions/counts still require equality.
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/"research/intake/v3-update/SERA_Kavi_Audit_v3"
sys.path.insert(0, str(PACK/"code"))
import verify_new_results as original  # noqa: E402

differences = []


def compatible_certificate(actual, expected):
    for value in (actual, expected):
        if original.digest(value["payload"]) != value["sha256"]:
            raise ValueError("Invalid certificate hash")
    new, old = copy.deepcopy(actual["payload"]), expected["payload"]
    for i, (a, b) in enumerate(zip(new["table"], old["table"])):
        delta = abs(a["upper_risk"]-b["upper_risk"])
        if delta >= 1e-12:
            raise ValueError("Unexpected certificate risk difference")
        if delta:
            differences.append({"old_certificate": expected["sha256"], "row": i,
                                "old": b["upper_risk"], "new": a["upper_risk"], "difference": delta})
        a["upper_risk"] = b["upper_risk"]
    if original.digest(new) != original.digest(old):
        raise ValueError("A nonnumeric certificate field changed")
    return True


started = time.perf_counter()
source = (PACK/"code/verify_new_results.py").read_text()
needle = "digest(cert)==digest(json.loads((root/f'certificate-{seed}.json').read_text()))"
if source.count(needle) != 1:
    raise ValueError("Unexpected strict verifier source")
namespace = {"__name__": "v3_compatibility_diagnosis", "compatible_certificate": compatible_certificate}
exec(compile(source.replace(needle, "compatible_certificate(cert,json.loads((root/f'certificate-{seed}.json').read_text()))"), "v3-compatibility-only", "exec"), namespace)
calibration = namespace["calibration"](PACK/"results/calibration")
try:
    alias = original.alias(PACK/"results/alias")
    alias_strict = {"status": "PASS"}
except ValueError as error:
    alias_strict = {"status": "FAIL", "error": str(error)}
    compatible = source.replace("np.array_equal(theta,r['learned_circle_parameters'])", "np.allclose(theta,r['learned_circle_parameters'],atol=1e-11,rtol=0)")
    compatible = compatible.replace("np.array_equal(cov,r['parameter_covariance'])", "np.allclose(cov,r['parameter_covariance'],atol=1e-11,rtol=0)")
    ns = {"__name__": "alias_compatibility_diagnosis"}
    exec(compile(compatible, "alias-compatibility-only", "exec"), ns)
    alias = ns["alias"](PACK/"results/alias")
result = {"calibration_original_strict": "FAIL: certificate canonical serialization; preserved",
          "calibration_compatibility": calibration, "bound_differences": differences,
          "alias_original_strict": alias_strict, "alias_compatibility": alias,
          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
          "wall_seconds": time.perf_counter()-started,
          "scope": "Saved-data diagnosis, no regeneration or training; compatibility does not replace strict provenance."}
target = ROOT/"research-continuation/16_v3/intake-replay-diagnosis.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "bound_differences"}, indent=2))
