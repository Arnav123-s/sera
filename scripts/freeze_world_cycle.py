"""Freeze the first prospective world-calibration attempt before drawing outcomes."""

import hashlib
import json
import zipfile

from experiments.generative_memory.core import digest
from experiments.world_calibration.study import (
    BASES,
    GROUPS,
    GUARD,
    QUERY_T,
    RELEASE,
    ROOT,
    SUPPORT_INDICES,
    THRESHOLDS,
    runtime,
    sources,
    write,
)

target = RELEASE/"GG-GUARD-002"
target.mkdir(parents=True, exist_ok=False)
timing = json.loads((ROOT/"runs/v3-batch-001/timing/data/timing.json").read_text())
payload = {
    "experiment": "GG-GUARD-002", "attempt": 0, "comparison_head": "9104839400d5053152ca423be6da1efe355aa71a",
    "hypotheses": ["Independent-world finite-grid risk calibration improves reliability at useful coverage",
                   "The pretrained score cannot separate omitted mechanisms; valid matched bounds do not fix this"],
    "counts": {"calibration": 1800, "matched": 1000, "mechanism_mixture_shift": 500,
               "abrupt_jump": 256, "decaying_spiral": 256},
    "seed_bases": BASES, "query_grid": QUERY_T.tolist(), "support_indices": list(SUPPORT_INDICES),
    "groups": list(GROUPS), "thresholds": list(THRESHOLDS), "risk_target": .1,
    "alpha_allocation": {"lifetime": .05, "attempt_0": .025, "global": .0125, "group": .0125,
                         "each_future_attempt_j": ".05 / 2**(j+1); new independent worlds required"},
    "gate": "Simultaneous matched active-group CP upper <=.1; in-menu coverage>=.5; paired utility 95% lower vs minimum90>=-.05; both accepted protected-family CP upper<=.1 at alpha .025 each. No-answer future groups establish no understanding.",
    "utility": {"correct": 1, "wrong": -4, "abstain": 0},
    "matched_sampling": "IID equal probability across six original teaching families; family identity never enters learner/group",
    "shift_scope": "Mechanism-mixture shift may alter risk within observable groups; no guarantee under arbitrary shift",
    "query_sampling": "One uniformly sampled query index from a seed-separated stream before outcome; one world is one independent unit",
    "protected_mechanisms": "Abrupt translated jump and exponentially decaying spiral; source frozen before any generated outcomes",
    "guard_path": GUARD.relative_to(ROOT).as_posix(), "guard_file_sha256": hashlib.sha256(GUARD.read_bytes()).hexdigest(),
    "sources": sources(), "runtime": runtime(), "timing_worlds": timing["worlds"],
    "timing_wall_seconds": timing["wall_seconds"], "job_cap_seconds": 600,
    "memory_limit_bytes": 2147483648, "new_neural_training_steps": 0,
    "audit_tolerances": {"independent_features_and_means": 2e-8, "independent_network": 1e-12, "binomial_inversion": 1e-11},
    "prohibition": "No tuning on final outcomes. Implementation bugs are retained and repaired explicitly; a changed statistical candidate needs a new attempt, alpha and unseen cohorts.",
}
record = {"payload": payload, "sha256": digest(payload)}
write(target/"protocol.json", record)
with zipfile.ZipFile(target/"frozen-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in payload["sources"]:
        archive.write(ROOT/name, name)
print(record["sha256"])
