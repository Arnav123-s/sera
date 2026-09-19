"""Measure a trained descendant: held-out language, transfer, retention, use.

Retention is not a parameter comparison. The declared 33-task retained suite and
the 39 practical examples are executed through the rebuilt descendant and each
case is compared, field by field, against the baseline record produced before
any teaching. Cases that are identical and cases that changed are both reported;
the protocol says which ones must be identical.

The final split is sealed. It is opened only when ``--open-final`` is passed,
and the run records that it was opened.
"""

import argparse
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
from experiments.owner_language.tasks import load_evaluation  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
STAGE47 = LAB_ROOT / "research-continuation/47_intervention_understanding"
BASELINE_ATTEMPT = RUNS / "attempts/baseline-verification-002"

PROTECTED_KINDS = {
    "field_what_if", "field_trajectory", "field_plan", "field_findings", "field_explain",
    "intervention_what_if", "intervention_explain", "intervention_plan", "intervention_findings",
    "what_if", "solve_solution", "apply_inquiry_rule", "inquiry_findings", "inquiry_next",
    "gap_findings", "gap_predict", "gap_next_observation", "gap_consequences",
    "solution_findings", "curiosity", "solve_discovery", "discovered_route",
    "observed_motion", "imagine", "read", "request", "status",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def compare_suite(context, tasks, baseline_cases, label, output):
    """Run the retained suite through the descendant and diff every case."""
    baseline = {case["id"]: case for case in baseline_cases}
    rows, cases = [], []
    for request in tasks:
        identifier = request.get("id")
        try:
            answer = usage.perform(context, dict(request))
            status = "ANSWERED"
            error = None
        except Exception as failure:
            answer, status, error = None, "FAILED", type(failure).__name__ + ": " + str(failure)
        cases.append({"id": identifier, "kind": request.get("kind"), "status": status,
                      "error": error, "result": None if answer is None else answer.get("result")})
        reference = baseline.get(identifier)
        same = None
        if reference is not None and reference["status"] == "ANSWERED" and status == "ANSWERED":
            same = canonical(reference["result"]["result"]) == canonical(answer["result"])
        rows.append({"id": identifier, "kind": request.get("kind"), "status": status, "error": error,
                     "baseline_status": None if reference is None else reference["status"],
                     "identical_to_baseline": same,
                     "protected": request.get("kind") in PROTECTED_KINDS})
    (output / f"{label}-cases.json").write_text(json.dumps(cases, indent=1, default=str) + "\n", encoding="utf-8")
    protected = [row for row in rows if row["protected"]]
    changed_protected = [row for row in protected if row["identical_to_baseline"] is not True]
    return {"suite": label, "declared": len(tasks),
            "answered": sum(1 for row in rows if row["status"] == "ANSWERED"),
            "failed": [row["id"] for row in rows if row["status"] != "ANSWERED"],
            "protected_cases": len(protected),
            "protected_identical": sum(1 for row in protected if row["identical_to_baseline"] is True),
            "protected_changed": [row["id"] for row in changed_protected],
            "unprotected_changed": [row["id"] for row in rows
                                    if not row["protected"] and row["identical_to_baseline"] is False],
            "protection_holds": not changed_protected,
            "rows": rows, "case_file": f"{label}-cases.json"}


def demonstrate(context, output):
    """Use the learned language to complete real questions from the held-out text."""
    transfer = load_evaluation("transfer")
    dev = load_evaluation("dev")
    shown = []
    for row in dev["families"]["definition"][:5]:
        answer = usage.perform(context, {"kind": "language_define", "id": "define:" + row["id"],
                                         "definition": " ".join(row["definition"]),
                                         "candidates": row["candidates"]})
        shown.append({"task": "definition->headword", "split": "dev", "question_id": row["id"],
                      "source": row["source"], "locator": row["locator"],
                      "true_answer": row["answer"], "chosen": answer["result"]["selected"],
                      "correct": answer["result"]["selected"] == row["answer"],
                      "margin": answer["result"]["margin_log_probability"]})
    for row in transfer["families"]["cloze"][:5]:
        answer = usage.perform(context, {"kind": "language_cloze", "id": "cloze:" + row["id"],
                                         "prefix": " ".join(row["prefix"]),
                                         "candidates": row["candidates"]})
        shown.append({"task": "unseen-author cloze", "split": "transfer", "question_id": row["id"],
                      "source": row["source"], "locator": row["locator"],
                      "true_answer": row["answer"], "chosen": answer["result"]["selected"],
                      "correct": answer["result"]["selected"] == row["answer"],
                      "margin": answer["result"]["margin_log_probability"]})
    completions = [usage.perform(context, {"kind": "language_complete", "text": text, "top": 5})["result"]
                   for text in ("the first letter of the english", "a triangle has three")]
    (output / "demonstration.json").write_text(
        json.dumps({"forced_choices": shown, "completions": completions}, indent=2, default=str) + "\n",
        encoding="utf-8")
    return {"forced_choices": shown, "completions": completions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="OLA-001")
    parser.add_argument("--arm", default="connected")
    parser.add_argument("--revision", type=int, default=None)
    parser.add_argument("--splits", default="dev,transfer")
    parser.add_argument("--open-final", action="store_true")
    parser.add_argument("--retention", action="store_true")
    parser.add_argument("--demonstrate", action="store_true")
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(options.output or os.environ.get("SERA_LAB_ATTEMPT", RUNS / "attempts/evaluate"))
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    context = usage.restore_descendant(options.run, options.arm, revision=options.revision)
    owner, tokens, metadata = context["owner"], context["tokens"], context["metadata"]
    record = {"schema": "sera.owner-language.evaluation-run.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.usage",)),
              "run": options.run, "arm": options.arm,
              "restore_seconds": time.perf_counter() - started,
              "descendant": owner.identity(), "checkpoint_revision": metadata["revision"],
              "trained_steps": metadata["step"],
              "parent_owner": metadata["baseline_contract"]["baseline_owner"],
              "baseline_contract_preserved": metadata["baseline_contract"],
              "vocabulary_digest": tokens.digest,
              "measurements": {}, "final_opened": False}

    splits = [name for name in options.splits.split(",") if name]
    if options.open_final and "final" not in splits:
        splits.append("final")
    for split in splits:
        if split == "final" and not options.open_final:
            continue
        evaluation = load_evaluation(split)
        record["measurements"][split] = tr.evaluate(owner, tokens, evaluation)
        record["measurements"][split]["set_sha256"] = hashlib.sha256(
            (RUNS / f"corpus/evaluation/{split}.json").read_bytes()).hexdigest()
        if split == "final":
            record["final_opened"] = True

    if options.retention:
        baseline = json.loads((BASELINE_ATTEMPT / "retention-cases.json").read_text())
        tasks = json.loads((STAGE47 / "retention-tasks.json").read_text())
        record["retention"] = compare_suite(context, tasks, baseline, "retention", output)
        baseline_examples = json.loads((BASELINE_ATTEMPT / "practical-examples-cases.json").read_text())
        examples = json.loads((STAGE47 / "example-tasks.json").read_text())
        if isinstance(examples, dict):
            examples = examples.get("tasks", [])
        record["practical_examples"] = compare_suite(context, examples, baseline_examples,
                                                     "practical-examples", output)

    if options.demonstrate:
        record["demonstration"] = demonstrate(context, output)

    (output / f"evaluation-{options.arm}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    summary = {"arm": options.arm, "steps": record["trained_steps"], "descendant": record["descendant"],
               "measurements": {split: {family: {k: value[family].get(k) for k in
                                                 ("status", "mean_nll", "perplexity", "accuracy", "questions",
                                                  "predicted_tokens")}
                                        for family in ("next_token", "cloze", "definition")}
                                for split, value in record["measurements"].items()}}
    if options.retention:
        summary["retention"] = {k: record["retention"][k] for k in
                                ("declared", "answered", "failed", "protected_cases", "protected_identical",
                                 "protected_changed", "unprotected_changed", "protection_holds")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
