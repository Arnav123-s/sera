"""Assemble the claim-to-evidence record for OLA-001 from the retained artefacts.

Nothing is computed here. Every number is copied from a file that a measurement
wrote, and each claim carries the path and SHA-256 of the artefact it came from,
so a reader can check any line without rerunning anything.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
STAGE = LAB_ROOT / "research-continuation/48_owner_language_acquisition"
ARMS = ("connected", "blocked", "disconnected")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    path = Path(path)
    if not path.exists():
        return {"path": None, "present": False}
    return {"path": path.relative_to(LAB_ROOT).as_posix(), "present": True,
            "sha256": sha256(path), "bytes": path.stat().st_size}


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def family_row(measurement, family):
    value = measurement.get(family, {}) if measurement else {}
    return {"status": value.get("status"), "mean_nll": value.get("mean_nll"),
            "perplexity": value.get("perplexity"), "accuracy": value.get("accuracy"),
            "chance": value.get("chance"), "questions": value.get("questions"),
            "predicted_tokens": value.get("predicted_tokens"), "qualifies": value.get("qualifies")}


def curve(training):
    if not training:
        return []
    return [{"label": row.get("label"), "step": row.get("step"),
             **{family: family_row(row, family) for family in ("next_token", "cloze", "definition")}}
            for row in training.get("evaluations", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="OLA-001")
    parser.add_argument("--evaluation-attempt", default=None)
    parser.add_argument("--correction-attempt", default=None)
    parser.add_argument("--replay-attempt", default=None)
    parser.add_argument("--out", default="RESULTS.json")
    options = parser.parse_args()

    report = {"schema": "sera.owner-language.results.1", "run": options.run,
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "protocol": source(STAGE / "protocol.json"),
              "corpus": source(STAGE / "corpus-manifest.json"),
              "pre_training_gate": source(STAGE / "pre-training-gate.json"),
              "calibration": source(STAGE / "calibration.json"),
              "baseline_verification": source(STAGE / "baseline-verification.json"),
              "arms": {}, "artefacts": {}}

    gate = load(STAGE / "pre-training-gate.json")
    if gate:
        report["pre_training_gate_result"] = {
            "passed": gate["passed"],
            "checks": [{"name": item["name"], "status": item["status"]} for item in gate["checks"]]}

    evaluation_attempt = Path(options.evaluation_attempt) if options.evaluation_attempt else None
    for arm in ARMS:
        training_path = RUNS / f"attempts/{options.run}-{arm}/training-{arm}.json"
        training = load(training_path)
        entry = {"training_record": source(training_path)}
        if training:
            entry["schedule"] = training["schedule"]
            entry["stages"] = [{key: stage.get(key) for key in
                                ("stage", "status", "steps", "cumulative_steps", "sources", "seconds",
                                 "mean_loss_first_quarter", "mean_loss_last_quarter", "exposure")}
                               for stage in training["stages"]]
            entry["final_step"] = training.get("final_step")
            entry["descendant"] = training.get("descendant_identity")
            entry["parent"] = training.get("parent_identity")
            entry["trainable"] = {key: training["trainable"][key] for key in
                                  ("trainable_tensors", "trainable_elements", "frozen_tensors",
                                   "frozen_elements", "connection")}
            entry["control"] = training.get("control")
            entry["development_curve"] = curve(training)
            entry["checkpoints"] = len(training.get("checkpoints", []))
        supervisor = load(RUNS / f"attempts/{options.run}-{arm}/state.json")
        if supervisor:
            entry["supervision"] = {key: supervisor.get(key) for key in
                                    ("status", "charged_seconds", "peak_job_committed_bytes",
                                     "memory_limit_bytes", "max_concurrent_processes",
                                     "verified_job_configuration", "lease")}
        if evaluation_attempt:
            path = evaluation_attempt / f"evaluation-{arm}.json"
            measured = load(path)
            entry["evaluation_record"] = source(path)
            if measured:
                entry["held_out"] = {split: {family: family_row(value, family)
                                             for family in ("next_token", "cloze", "definition")}
                                     for split, value in measured["measurements"].items()}
                entry["evaluation_sets"] = {split: value.get("set_sha256")
                                            for split, value in measured["measurements"].items()}
                entry["final_opened"] = measured.get("final_opened")
                if "retention" in measured:
                    entry["retention"] = {key: measured["retention"][key] for key in
                                          ("declared", "answered", "failed", "classification_counts",
                                           "kinds_changed_through_the_adapter", "kinds_unchanged",
                                           "anomalies", "protection_holds")}
                if "practical_examples" in measured:
                    entry["practical_examples"] = {key: measured["practical_examples"][key] for key in
                                                   ("declared", "answered", "failed", "classification_counts",
                                                    "anomalies", "protection_holds")}
                if "demonstration" in measured:
                    entry["demonstration"] = measured["demonstration"]
        report["arms"][arm] = entry

    if options.correction_attempt:
        path = Path(options.correction_attempt) / "correction.json"
        correction = load(path)
        report["correction"] = {"record": source(path)}
        if correction:
            report["correction"].update({key: correction[key] for key in
                                         ("identified_weakness", "practice", "independent_check",
                                          "retention_of_other_sources", "transfer_after_correction",
                                          "return_to_the_original_task", "passed")})
    if options.replay_attempt:
        path = Path(options.replay_attempt) / "independent-replay.json"
        replay = load(path)
        report["independent_replay"] = {"record": source(path)}
        if replay:
            report["independent_replay"].update({
                "passed": replay["passed"], "sources": replay["sources"],
                "parent_equivalence": replay["parent_equivalence"],
                "arms": {arm: {"checkpoints": entry["checkpoints"]["revisions"],
                               "all_weight_hashes_match": entry["checkpoints"]["all_weight_hashes_match"],
                               "revision_chain_monotonic": entry["checkpoints"]["revision_chain_monotonic"],
                               "agreement_with_claims": entry.get("agreement_with_claims")}
                         for arm, entry in replay["arms"].items()}})

    costs = load(RUNS / "costs.json")
    if costs:
        report["costs"] = {"ledger": source(RUNS / "costs.json"),
                           "charged_seconds": costs["charged_seconds"],
                           "attempts": len(costs["attempts"]),
                           "failed_attempts": [a["label"] for a in costs["attempts"] if a["status"] != "PASS"],
                           "memory_limit_bytes": costs["memory_limit_bytes"],
                           "paid_services": costs["paid_services"],
                           "peak_committed_bytes": max(a["peak_job_committed_bytes"] for a in costs["attempts"]),
                           "by_attempt": [{key: a[key] for key in
                                           ("label", "status", "charged_seconds", "peak_job_committed_bytes",
                                            "lease_released")} for a in costs["attempts"]]}

    out = STAGE / options.out
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"written": out.relative_to(LAB_ROOT).as_posix(),
                      "arms": {arm: {"final_step": entry.get("final_step"),
                                     "descendant": entry.get("descendant"),
                                     "dev": (entry.get("held_out") or {}).get("dev")}
                               for arm, entry in report["arms"].items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
