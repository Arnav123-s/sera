"""Detect a measured weakness, practise against it, and check the repair honestly.

The campaign asks for one task in which the learner finds a specific missing
ability, obtains permitted human material, practises, passes an independent
check, retains what it already had, transfers, and returns to the original task.

The weakness is not chosen by hand. The development set is cut by group into a
*diagnosis* half and a *check* half; the weakest source on the diagnosis half is
the one repaired. Practice draws only from that source's training split. The
check half was never used to choose the weakness and never trained on, so it is
an independent check of the repair rather than a restatement of it.

Retention is measured on the other sources at the same moment, so a repair that
works by forgetting is visible.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate, identity

activate()

import torch  # noqa: E402

from experiments.owner_language import training as tr  # noqa: E402
from experiments.owner_language import usage  # noqa: E402
from experiments.owner_language.corpus import load_items  # noqa: E402
from experiments.owner_language.tasks import load_evaluation  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"


def halve(rows):
    """Split by group so a group is wholly in diagnosis or wholly in check."""
    diagnosis, check = [], []
    for row in rows:
        residue = int(hashlib.sha256(("OLA-48-correct:" + row["group"]).encode()).hexdigest(), 16) % 2
        (diagnosis if residue == 0 else check).append(row)
    return diagnosis, check


def measure_by_source(owner, tokens, rows, family):
    sources = sorted({row["source"] for row in rows})
    return {source: tr.measure_choice(owner, tokens, [row for row in rows if row["source"] == source], family)
            for source in sources}


def summary(measurements):
    return {source: {"status": value["status"], "questions": value.get("questions"),
                     "accuracy": value.get("accuracy"), "chance": value.get("chance")}
            for source, value in measurements.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="OLA-001")
    parser.add_argument("--arm", default="connected")
    parser.add_argument("--practice-steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(options.output or os.environ.get("SERA_LAB_ATTEMPT", RUNS / "attempts/correct"))
    output.mkdir(parents=True, exist_ok=True)

    context = usage.restore_descendant(options.run, options.arm)
    owner, tokens, metadata = context["owner"], context["tokens"], context["metadata"]
    cloze = load_evaluation("dev")["families"]["cloze"]
    diagnosis, check = halve(cloze)
    transfer_rows = load_evaluation("transfer")["families"]["cloze"]

    record = {"schema": "sera.owner-language.correction.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.usage",)),
              "run": options.run, "arm": options.arm,
              "start_descendant": owner.identity(), "start_step": metadata["step"],
              "split_rule": "development cloze cut by group: sha256('OLA-48-correct:<group>') mod 2",
              "diagnosis_questions": len(diagnosis), "check_questions": len(check)}

    before_diagnosis = measure_by_source(owner, tokens, diagnosis, "cloze")
    record["diagnosis"] = summary(before_diagnosis)
    eligible = {source: value for source, value in before_diagnosis.items() if value["status"] == "MEASURED"}
    if not eligible:
        record["status"] = "NO_MEASURABLE_DIAGNOSIS"
        (output / "correction.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        raise SystemExit(1)
    weakest = min(eligible, key=lambda source: eligible[source]["accuracy"])
    record["identified_weakness"] = {
        "source": weakest, "accuracy": eligible[weakest]["accuracy"],
        "questions": eligible[weakest]["questions"],
        "others": {source: value["accuracy"] for source, value in eligible.items() if source != weakest},
        "chosen_by": "lowest development-diagnosis cloze accuracy, not by hand"}

    before_check = measure_by_source(owner, tokens, check, "cloze")
    before_transfer = tr.measure_choice(owner, tokens, transfer_rows, "cloze")
    before_next_token = tr.measure_next_token(owner, load_evaluation("dev")["families"]["next_token"])
    record["before"] = {"check": summary(before_check), "transfer": before_transfer,
                        "dev_next_token": before_next_token}

    schedule = tr.validate_schedule([{"name": "correction-" + weakest, "sources": [weakest],
                                      "steps": options.practice_steps,
                                      "objective": f"bounded corrective practice on {weakest} training material"}],
                                    [weakest])
    store = RUNS / "descendants" / options.run / (options.arm + "-corrected")
    trainer = tr.Trainer(owner, tokens, schedule, arm=options.arm, learning_rate=options.learning_rate,
                         batch=options.batch, length=options.length, seed=4901, store=store,
                         baseline_contract=metadata["baseline_contract"])
    trainer.parent_identity = metadata["baseline_contract"]["baseline_owner"]
    trainer.step = 0
    practice_items = [row for row in load_items(split="train") if row["source"] == weakest]
    record["practice"] = {"source": weakest, "material": "training split only, human-authored",
                          "available_items": len(practice_items), "steps": options.practice_steps,
                          "store": str(store)}
    started = time.perf_counter()
    stage = trainer.run_stage(schedule[0], report_every=100, checkpoint_every=300)
    trainer.save(note="correction-complete")
    record["practice"]["stage"] = stage
    record["practice"]["seconds"] = time.perf_counter() - started
    record["corrected_descendant"] = owner.identity()

    after_check = measure_by_source(owner, tokens, check, "cloze")
    after_transfer = tr.measure_choice(owner, tokens, transfer_rows, "cloze")
    after_next_token = tr.measure_next_token(owner, load_evaluation("dev")["families"]["next_token"])
    record["after"] = {"check": summary(after_check), "transfer": after_transfer,
                       "dev_next_token": after_next_token}

    def accuracy(measurements, source):
        value = measurements.get(source, {})
        return value.get("accuracy") if value.get("status") == "MEASURED" else None

    improved = None
    if accuracy(before_check, weakest) is not None and accuracy(after_check, weakest) is not None:
        improved = accuracy(after_check, weakest) - accuracy(before_check, weakest)
    retained = {}
    for source in before_check:
        if source == weakest:
            continue
        start, end = accuracy(before_check, source), accuracy(after_check, source)
        retained[source] = {"before": start, "after": end,
                            "change": None if start is None or end is None else end - start,
                            "within_tolerance": None if start is None or end is None
                            else end >= start - options.retention_tolerance}
    record["independent_check"] = {
        "source": weakest, "before": accuracy(before_check, weakest), "after": accuracy(after_check, weakest),
        "change": improved, "passed": bool(improved is not None and improved > 0),
        "meaning": "the check half was never used to choose the weakness and was never trained on"}
    record["retention_of_other_sources"] = {
        "tolerance": options.retention_tolerance, "by_source": retained,
        "passed": all(value["within_tolerance"] for value in retained.values()) if retained else None}
    record["transfer_after_correction"] = {
        "source": "descartes", "before": before_transfer.get("accuracy"),
        "after": after_transfer.get("accuracy"),
        "change": None if before_transfer["status"] != "MEASURED" or after_transfer["status"] != "MEASURED"
        else after_transfer["accuracy"] - before_transfer["accuracy"]}

    retained_suite = json.loads(
        (LAB_ROOT / "research-continuation/47_intervention_understanding/retention-tasks.json").read_text())
    failures, answered = [], 0
    for request in retained_suite:
        try:
            usage.perform(context, dict(request))
            answered += 1
        except Exception as error:
            failures.append({"id": request.get("id"), "error": type(error).__name__ + ": " + str(error)})
    record["return_to_the_original_task"] = {
        "suite": "declared 33 retained tasks", "answered": answered, "failed": failures,
        "passed": not failures}

    record["passed"] = bool(record["independent_check"]["passed"]
                            and record["retention_of_other_sources"]["passed"]
                            and record["return_to_the_original_task"]["passed"])
    (output / "correction.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("identified_weakness", "independent_check",
                                             "retention_of_other_sources", "transfer_after_correction",
                                             "return_to_the_original_task", "passed")},
                     indent=2, default=str))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
