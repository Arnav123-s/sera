"""Read-only numerical/structural parity audit of archived and fresh v2b records.

No fitting, score selection, artifact editing, or tolerance relaxation occurs.
Every floating leaf is compared by the verifier's atol=rtol=1e-12 criterion.
Hash/serialization-size/platform/timing differences are retained separately.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("D:/ai/projects/sera/research-continuation/12_reproductions/bundle2")
SOURCE = ROOT / "source"
FRESH = ROOT / "reproduction"
DEST = ROOT / "independent-audit"
HASH = re.compile(r"^[0-9a-f]{64}$")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Comparator:
    def __init__(self):
        self.counts = Counter()
        self.max_numeric_error = 0.0
        self.max_numeric_location = None
        self.failures = []
        self.non_numeric_differences = []

    def differing(self, category, path, a, b):
        self.counts[category] += 1
        self.non_numeric_differences.append({"category": category, "path": path, "archived": a, "fresh": b})

    def compare(self, a, b, path="root"):
        key = path.rsplit("/", 1)[-1]
        if key == "environment":
            self.counts["environment_records"] += 1
            if a != b:
                self.differing("environment_difference", path, a, b)
            return
        if key.endswith("seconds") and isinstance(a, (int, float)):
            self.counts["timing_fields"] += 1
            if a != b:
                self.differing("timing_difference", path, a, b)
            return
        if key.endswith("bytes") and isinstance(a, int):
            self.counts["serialized_size_fields"] += 1
            if a != b:
                self.differing("serialized_size_difference", path, a, b)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                # Program-library keys are hashes of float-bearing definitions.
                # Align only uniquely numerically equivalent definitions, retaining
                # all changed key identities as evidence rather than dropping them.
                if a and b and all(HASH.fullmatch(k) for k in (*a, *b)):
                    remaining = set(b)
                    for old_key, value in a.items():
                        matches = [new_key for new_key in remaining if value.get("support") == b[new_key].get("support") and np.allclose(value.get("coefficients"), b[new_key].get("coefficients"), atol=1e-12, rtol=1e-12)]
                        if len(matches) != 1:
                            self.failures.append({"path": path, "category": "ambiguous_or_missing_library_definition", "old_key": old_key, "matches": matches})
                            continue
                        new_key = matches[0]
                        remaining.remove(new_key)
                        if old_key != new_key:
                            self.differing("program_key_hash_difference", path, old_key, new_key)
                        self.compare(value, b[new_key], path + "/program_definition")
                    if remaining:
                        self.failures.append({"path": path, "category": "extra_library_definitions", "keys": sorted(remaining)})
                    return
                self.failures.append({"path": path, "category": "dictionary_keys", "archived_only": sorted(set(a) - set(b)), "fresh_only": sorted(set(b) - set(a))})
            for key in sorted(set(a) & set(b)):
                self.compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.failures.append({"path": path, "category": "list_length", "archived": len(a), "fresh": len(b)})
            for i, (x, y) in enumerate(zip(a, b)):
                self.compare(x, y, path + f"/{i}")
        elif isinstance(a, float) and isinstance(b, (float, int)):
            self.counts["numeric_float_leaves"] += 1
            error = abs(a - b)
            if a != b:
                self.counts["numeric_float_last_bit_differences"] += 1
            if error > self.max_numeric_error:
                self.max_numeric_error, self.max_numeric_location = error, path
            if not np.isclose(a, b, atol=1e-12, rtol=1e-12):
                self.failures.append({"path": path, "category": "numeric_tolerance", "archived": a, "fresh": b, "absolute_error": error, "allowed_error": 1e-12 + 1e-12 * abs(b)})
        elif isinstance(a, str) and isinstance(b, str) and HASH.fullmatch(a) and HASH.fullmatch(b):
            self.counts["hash_leaves"] += 1
            if a != b:
                self.differing("hash_difference", path, a, b)
        else:
            self.counts["exact_nonfloat_leaves"] += 1
            if a != b:
                self.failures.append({"path": path, "category": "exact_value", "archived": a, "fresh": b})

    def report(self):
        return {"counts": dict(self.counts), "max_numeric_absolute_error": self.max_numeric_error, "max_numeric_location": self.max_numeric_location, "failures": self.failures, "separately_recorded_differences": self.non_numeric_differences}


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((SOURCE / "protocols/experiment_v2.json").read_text())
    repair_protocol = json.loads((SOURCE / "protocols/repair_v2.json").read_text())
    expected = {
        "main": {f"main_{s}_{f}_{m}.json" for s, f, m in itertools.product(protocol["seeds"], protocol["families"], protocol["methods"])},
        "meta": {f"meta_{s}_meta_lifetime.json" for s in protocol["seeds"]},
        "repair": {f"{s}_{f}.json" for s, f in itertools.product(repair_protocol["seeds"], repair_protocol["families"])},
    }
    comparisons = {}
    records = []
    integrity_failures = []
    coverage = {}
    fresh_values = defaultdict(list)
    for cohort in ("main", "meta", "repair"):
        old_files = {p.name: p for p in (SOURCE / "results" / cohort).glob("*.json")}
        new_files = {p.name: p for p in (FRESH / "results" / cohort).glob("*.json")}
        coverage[cohort] = {"expected": len(expected[cohort]), "archived": len(old_files), "fresh": len(new_files), "archived_missing": sorted(expected[cohort] - set(old_files)), "fresh_missing": sorted(expected[cohort] - set(new_files)), "archived_extra": sorted(set(old_files) - expected[cohort]), "fresh_extra": sorted(set(new_files) - expected[cohort])}
        comparator = Comparator()
        for name in sorted(set(old_files) & set(new_files)):
            old = json.loads(old_files[name].read_text())
            new = json.loads(new_files[name].read_text())
            for role, value in (("archived", old), ("fresh", new)):
                copy = dict(value)
                recorded = copy.pop("record_sha256")
                if digest(copy) != recorded:
                    integrity_failures.append({"cohort": cohort, "file": name, "role": role, "issue": "outer_record_digest"})
                expected_protocol = digest(repair_protocol if cohort == "repair" else protocol)
                if value["protocol_sha256"] != expected_protocol:
                    integrity_failures.append({"cohort": cohort, "file": name, "role": role, "issue": "protocol_digest"})
            before = len(comparator.failures)
            comparator.compare(old, new, f"{cohort}/{name}")
            records.append({"cohort": cohort, "file": name, "archived_sha256": sha(old_files[name]), "fresh_sha256": sha(new_files[name]), "numerical_or_structural_failures": len(comparator.failures) - before})
            fresh_values[cohort].append(new["result"])
        comparisons[cohort] = comparator.report()
    summaries = {}
    main_rows = fresh_values["main"]
    for scope in ("in_basis", "misspecified"):
        summaries[scope] = []
        for method in protocol["methods"]:
            rows = [r for r in main_rows if r["method"] == method and ((r["family"] != "misspecified") == (scope == "in_basis"))]
            summaries[scope].append({"method": method, "trials": len(rows), "interpolation_mse": float(np.mean([r["evaluation"]["interpolation"]["mse"] for r in rows])), "extrapolation_mse": float(np.mean([r["evaluation"]["extrapolation"]["mse"] for r in rows])), "rollout_mse": float(np.mean([r["evaluation"]["rollout"]["mse"] for r in rows])), "interpolation_95_coverage": float(np.mean([r["evaluation"]["interpolation"]["gaussian_95_coverage"] for r in rows])), "retained_terms": float(np.mean([len(r["artifact"]["program"]["support"]) for r in rows])), "validation_acceptances": sum(r["evaluation"]["validation_accepts"] for r in rows)})
    summaries["meta_generation4"] = []
    for knowledge_current, policy_current in itertools.product((False, True), repeat=2):
        rows = [r for life in fresh_values["meta"] for h in life["history"] if h["generation"] == 4 for r in h["evaluations"] if r["knowledge_current"] == knowledge_current and r["policy_current"] == policy_current]
        summaries["meta_generation4"].append({"knowledge_current": knowledge_current, "policy_current": policy_current, "tasks": len(rows), "interpolation_mse": float(np.mean([r["evaluation"]["interpolation"]["mse"] for r in rows]))})
    summaries["repair"] = []
    for family in repair_protocol["families"]:
        rows = [r for r in fresh_values["repair"] if r["family"] == family]
        summaries["repair"].append({"family": family, "trials": len(rows), "triggered": sum(r["triggered"] for r in rows), "admitted": sum(r["admitted"] for r in rows), "frequency_identified": sum(r["frequency_identified"] for r in rows), "selected_frequencies": dict(Counter(str(r["selected_frequency"]) for r in rows)), **{split + "_" + arm + "_mse": float(np.mean([r["evaluation"][split][arm + "_mse"] for r in rows])) for split, arm in itertools.product(("interpolation", "extrapolation"), ("fixed", "repair"))}})
    summary_comparisons = {}
    for name in ("paired_comparisons.json", "meta_paired_comparisons.json", "experiment_totals.json"):
        comparator = Comparator()
        comparator.compare(json.loads((SOURCE / "results" / name).read_text()), json.loads((FRESH / "results" / name).read_text()), name)
        summary_comparisons[name] = comparator.report()
    manifest = []
    for line in (SOURCE / "MANIFEST.sha256").read_text().splitlines():
        expected_hash, relative = line.split("  ", 1)
        actual_hash = sha(SOURCE / relative)
        manifest.append({"path": relative, "expected": expected_hash, "actual": actual_hash, "matches": expected_hash == actual_hash})
    code_files = sorted((SOURCE / "code").rglob("*.py"))
    code_map = {p.relative_to(SOURCE / "code").as_posix(): sha(p) for p in code_files}
    experiment_map = {k: v for k, v in code_map.items() if k.startswith("sklab/") or k == "run_study.py"}
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "archived_root": SOURCE.as_posix(), "fresh_root": FRESH.as_posix(),
        "numeric_atol": 1e-12, "numeric_rtol": 1e-12, "coverage": coverage, "integrity_failures": integrity_failures,
        "comparisons": comparisons, "records": records, "recomputed_fresh_summaries": summaries, "summary_comparisons": summary_comparisons,
        "manifest_count": len(manifest), "manifest_all_match": all(m["matches"] for m in manifest), "manifest": manifest,
        "all_code_posix_map": code_map, "all_code_posix_digest": digest(code_map), "experimental_posix_map": experiment_map, "experimental_posix_digest": digest(experiment_map),
        "experimental_native_digest": digest({k.replace("/", "\\"): v for k, v in experiment_map.items()}),
        "protocol_digests": {"P1": digest(protocol), "P2": digest(repair_protocol)},
        "audit_script_sha256": sha(Path(__file__)), "argv": sys.argv,
        "boundary": "Read-only comparison of all archived and fresh raw records; no new independent seeds, training or verification tolerance changes. Hash and byte-length differences remain failures for the original strict archive-repair verifier.",
    }
    (DEST / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"coverage": coverage, "integrity_failures": integrity_failures, "manifest_count": len(manifest), "manifest_all_match": result["manifest_all_match"], "comparisons": {k: {"counts": v["counts"], "max_numeric_absolute_error": v["max_numeric_absolute_error"], "max_numeric_location": v["max_numeric_location"], "failures": v["failures"][:10], "failure_count": len(v["failures"])} for k, v in comparisons.items()}, "summaries": summaries, "experimental_posix_digest": result["experimental_posix_digest"], "all_code_posix_digest": result["all_code_posix_digest"]}, indent=2))


if __name__ == "__main__":
    main()
