"""Fresh-process proof that the laboratory baseline owner restores and behaves.

This replaces two defective checks preserved from the earlier draft:

* ``verify_owner_baseline.py`` named the laboratory checkpoint but restored the
  production default and only tested ``Path.exists()`` (review finding 2);
* ``verify_retention()`` compared one subject's parameters against copies of
  themselves and returned three route names without calling them, then the
  report read that as "zero catastrophic forgetting" (review finding 3).

Here the store is named explicitly, the restored identity is checked against the
recorded owner, every parameter and buffer of the shared owner is inventoried,
and the declared 33-task retention suite is executed through the restored owner
with per-case results and errors preserved.  A case that raises is a failure,
never a pass.
"""

import argparse
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

from experiments.owner_language.owner import (  # noqa: E402
    BASELINE_STORE,
    inventory_summary,
    parameter_inventory,
    restore_owner,
)

LAB_ROOT = Path(__file__).resolve().parents[1]
RETENTION = LAB_ROOT / "research-continuation/47_intervention_understanding/retention-tasks.json"
EXAMPLES = LAB_ROOT / "research-continuation/47_intervention_understanding/example-tasks.json"


def run_suite(session, growth, tasks, label):
    """Execute real requests through the restored owner and keep every outcome."""
    from scripts.intervention_use import perform

    before = session.state()
    cases, failures = [], 0
    for request in tasks:
        started = time.perf_counter()
        try:
            result = perform(session, growth, dict(request))
            cases.append({"id": request.get("id"), "kind": request.get("kind"), "status": "ANSWERED",
                          "seconds": time.perf_counter() - started, "result": result})
        except Exception as error:  # a failed retained task is a failure, not a pass
            failures += 1
            cases.append({"id": request.get("id"), "kind": request.get("kind"), "status": "FAILED",
                          "seconds": time.perf_counter() - started,
                          "error": type(error).__name__ + ": " + str(error)})
    unchanged = session.state() == before
    return {"suite": label, "declared": len(tasks), "answered": len(tasks) - failures, "failed": failures,
            "read_only": unchanged, "passed": failures == 0 and unchanged and len(tasks) > 0, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=BASELINE_STORE)
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)

    output = options.output or Path(os.environ.get("SERA_LAB_ATTEMPT", LAB_ROOT / "runs/owner-learning-001/baseline-check"))
    output.mkdir(parents=True, exist_ok=True)

    record = {"schema": "sera.owner-language.baseline-verification.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "supervised_token_present": bool(os.environ.get("SERA_LAB_SUPERVISED")),
              "process": identity(("experiments.owner_language.owner",))}

    started = time.perf_counter()
    session, growth, restore_record = restore_owner(options.store)
    record["restore"] = restore_record
    record["restore_seconds"] = time.perf_counter() - started

    entries = parameter_inventory(session.owner)
    record["owner_inventory_summary"] = inventory_summary(entries)
    record["owner_inventory_summary"]["intervention_models"] = len(session.owner.intervention_models)
    record["owner_inventory_summary"]["physical_fields"] = len(session.owner.physical_fields)
    record["owner_config"] = session.owner.export_config()
    (output / "owner-inventory.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")

    tasks = json.loads(RETENTION.read_text())
    record["retention"] = run_suite(session, growth, tasks, "declared-retained-tasks")
    examples = json.loads(EXAMPLES.read_text())
    if isinstance(examples, dict):
        examples = examples.get("tasks", [])
    record["practical_examples"] = run_suite(session, growth, examples, "declared-practical-examples")

    for key in ("retention", "practical_examples"):
        cases = record[key].pop("cases")
        (output / (key.replace("_", "-") + "-cases.json")).write_text(json.dumps(cases, indent=1, default=str) + "\n",
                                                                      encoding="utf-8")
        record[key]["case_file"] = (output / (key.replace("_", "-") + "-cases.json")).name
        record[key]["failed_ids"] = [c["id"] for c in cases if c["status"] == "FAILED"]

    record["passed"] = bool(record["retention"]["passed"] and record["practical_examples"]["passed"])
    (output / "baseline-verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in {"owner_config", "process"}}, indent=2))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
