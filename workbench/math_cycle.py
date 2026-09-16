"""One continuing learner: request, acquire, verify, forget and reacquire."""

import argparse
import copy
import json
import time

import torch

from experiments.grounded_audit import semantic, solutions
from experiments.grounded_language.acquisition import shape
from experiments.grounded_language.data import examples
from experiments.grounded_language.study import (
    BASE,
    RELEASE,
    ROOT,
    evaluate,
    packed,
    read,
    sha,
    write,
)

from .model import Learner, transact
from .storage import Store

TASKS = [
    ("subtract three from x then multiply by two to get four modulo eleven", "known", False),
    ("scale x by three then subtract two to get four modulo eleven", "known", True),
    ("the product of x and three minus two equals four modulo eleven", "novel", False),
    ("subtract three from x then multiply by two to get four modulo eleven", "known", False),
    ("multiply the difference of x and two by three equals four modulo eleven", "novel", False),
    ("take x minus two scale by three and obtain four modulo eleven", "known", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="L11-ONREQUEST-001")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output, workspace = RELEASE/args.name, ROOT/"runs"/args.name.lower()
    output.mkdir(parents=True, exist_ok=False)
    if workspace.exists():
        raise FileExistsError("Do not restart an existing continuing learner")
    protocol = {"name": args.name, "tasks": TASKS, "seed": 190001,
                "qualification_sha256": sha(RELEASE/"qualification.json"),
                "base_sha256": sha(BASE), "evaluation_per_step": 512,
                "policy": "The fixed runtime chooses a supplied lesson from missing wording, trains on partition 2/3/4, validates on 1, and updates the same owner. Test scores never enter the policy.",
                "gate": "All six requested examples independently correct when accepted; fresh cases of the requested sentence shape have exact translation >=.95. Broader family scores separately expose forgetting of unrequested wording.",
                "unit": "One continuing six-request learner, not six independent learners.",
                "sources": {name: sha(ROOT/name) for name in ("workbench/model.py", "workbench/language_runtime.py", "workbench/math_cycle.py")}}
    write(output/"protocol.json", protocol)
    store = Store(workspace)
    store.commit(copy.deepcopy(read(BASE)), None)
    started = time.perf_counter()
    records, raw = [], {}
    for index, (text, wording, lexical) in enumerate(TASKS):
        prior = store.read()
        answer = transact(workspace, {"operation": "learn_language", "request_id": f"fixed-request-{index}", "text": text})
        current = store.read()
        learner = Learner(current)
        metric, detail = evaluate(learner.session.owner, examples(190001+index*101, 512, "final", wording=wording, lesson=lexical),
                                  threshold=learner.language["threshold"])
        result = answer["result"]
        relevant = [row for row in detail if shape(row["text"]) == shape(text)]
        requested_accuracy = sum(row["translation_correct"] for row in relevant)/len(relevant)
        independently_correct = result.get("labels") == semantic(text) and result.get("solutions") == solutions(semantic(text))
        assert len(learner.session.events) == len(prior["interaction"]["events"])
        assert learner.solver.neural.owner is learner.solver.components["typed"].owner is learner.session.owner
        record = {"index": index, "request": text, "answer": result, "fresh_family": metric,
                  "requested_shape_examples": len(relevant), "requested_shape_exact_translation": requested_accuracy,
                  "independently_correct": independently_correct, "revision": current["revision"],
                  "owner_sha256": current["owner_sha256"], "lesson_count": learner.language["lessons"],
                  "cumulative_optimizer_steps": learner.language["optimizer_steps"],
                  "cumulative_teaching_examples": learner.language["paid_teaching_examples"],
                  "original_contexts_preserved": current["contexts"] == read(BASE)["contexts"],
                  "exact_restored_owner": learner.view()["owner"] == current["owner_sha256"]}
        records.append(record)
        raw[str(index)] = detail
        write(output/f"step-{index}.json", record)
        print(json.dumps({"step": index, "status": result["status"], "lesson": result["learned_on_request"],
                          "correct": independently_correct, "fresh_exact_translation": metric["exact_translation"],
                          "retired_shapes": [] if result["lesson"] is None else result["lesson"]["withdrawn_shapes"]}), flush=True)
    packed(output/"evaluation.json.gz", raw)
    passed = all(r["answer"]["status"] == "ACCEPTED" and r["independently_correct"] and r["requested_shape_exact_translation"] >= .95 for r in records)
    summary = {"status": "PASS" if passed else "REJECTED", "requests": len(records), "records": records,
               "verified_revisions": store.verify_history(), "workspace": workspace.relative_to(ROOT).as_posix(),
               "worker_seconds": time.perf_counter()-started, "protocol_sha256": sha(output/"protocol.json")}
    write(output/"result.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
