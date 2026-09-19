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
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate, identity

activate()

import torch  # noqa: E402

from experiments.owner_language import training as tr  # noqa: E402
from experiments.owner_language import usage  # noqa: E402
from experiments.owner_language.descendant import adapter_disabled  # noqa: E402
from experiments.owner_language.tasks import load_evaluation  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
STAGE47 = LAB_ROOT / "research-continuation/47_intervention_understanding"
BASELINE_ATTEMPT = RUNS / "attempts/baseline-verification-002"


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def answer_case(context, request):
    try:
        answer = usage.perform(context, dict(request))
        return {"status": "ANSWERED", "result": answer.get("result"), "error": None}
    except Exception as failure:
        return {"status": "FAILED", "result": None,
                "error": type(failure).__name__ + ": " + str(failure)}


def compare_suite(context, tasks, baseline_cases, label, output):
    """Run the retained suite twice and let each route classify itself.

    The adapter is the only channel by which teaching can reach an inherited
    route. Running every case with the adapter live and again with it zeroed
    therefore separates three outcomes without any hand-written list:

    * unchanged either way — the route does not read the shared core, so it is
      protected by construction and the measurement confirms it;
    * changed with the adapter live, identical to the baseline with it zeroed —
      the route does read the shared core, and the change is exactly the
      learning arriving through the declared connection;
    * changed even with the adapter zeroed — something other than the adapter
      moved, which must not happen and is reported as an anomaly.
    """
    baseline = {case["id"]: case for case in baseline_cases}
    rows, cases = [], []
    owner = context["owner"]
    for request in tasks:
        identifier = request.get("id")
        live = answer_case(context, request)
        with adapter_disabled(owner) as available:
            zeroed = answer_case(context, request) if available else dict(live)
        reference = baseline.get(identifier)
        same_live = same_zeroed = None
        if reference is not None and reference["status"] == "ANSWERED":
            expected = canonical(reference["result"]["result"])
            if live["status"] == "ANSWERED":
                same_live = expected == canonical(live["result"])
            if zeroed["status"] == "ANSWERED":
                same_zeroed = expected == canonical(zeroed["result"])
        if same_live is True and same_zeroed is True:
            classification = "unchanged"
        elif same_live is False and same_zeroed is True:
            classification = "changed_through_the_adapter"
        elif same_zeroed is False:
            classification = "changed_without_the_adapter"
        else:
            classification = "undetermined"
        cases.append({"id": identifier, "kind": request.get("kind"), "status": live["status"],
                      "error": live["error"], "result": live["result"],
                      "adapter_zeroed_result": zeroed["result"], "classification": classification})
        rows.append({"id": identifier, "kind": request.get("kind"), "status": live["status"],
                     "error": live["error"],
                     "baseline_status": None if reference is None else reference["status"],
                     "identical_to_baseline": same_live,
                     "identical_with_adapter_zeroed": same_zeroed,
                     "classification": classification})
    (output / f"{label}-cases.json").write_text(json.dumps(cases, indent=1, default=str) + "\n", encoding="utf-8")
    counts = Counter(row["classification"] for row in rows)
    anomalies = [row["id"] for row in rows if row["classification"] == "changed_without_the_adapter"]
    return {"suite": label, "declared": len(tasks),
            "answered": sum(1 for row in rows if row["status"] == "ANSWERED"),
            "failed": [row["id"] for row in rows if row["status"] != "ANSWERED"],
            "classification_counts": dict(counts),
            "unchanged": [row["id"] for row in rows if row["classification"] == "unchanged"],
            "changed_through_the_adapter": [row["id"] for row in rows
                                            if row["classification"] == "changed_through_the_adapter"],
            "kinds_changed_through_the_adapter": sorted(
                {row["kind"] for row in rows if row["classification"] == "changed_through_the_adapter"}),
            "kinds_unchanged": sorted({row["kind"] for row in rows if row["classification"] == "unchanged"}),
            "anomalies": anomalies,
            "protection_holds": not anomalies and not [row["id"] for row in rows if row["status"] != "ANSWERED"],
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
                                ("declared", "answered", "failed", "classification_counts",
                                 "kinds_changed_through_the_adapter", "kinds_unchanged",
                                 "anomalies", "protection_holds")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
