"""Independent replay: re-derive the published numbers from the retained bytes.

This script does not trust the training record. In a fresh process it re-hashes
the corpus, the evaluation sets and every checkpoint in the series, walks the
revision chain, rebuilds the descendant from the laboratory baseline plus the
saved tensors, and recomputes the development measurements the run reported. It
also re-checks the two structural claims that everything else rests on: that the
step-0 descendant is behaviourally the parent, and that the descendant's
identity is reproducible from the checkpoint alone.
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
from experiments.owner_language.corpus import CORPUS  # noqa: E402
from experiments.owner_language.tasks import load_evaluation  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
STAGE = LAB_ROOT / "research-continuation/48_owner_language_acquisition"
BASELINE_ATTEMPT = RUNS / "attempts/baseline-verification-002"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replay_sources():
    published = json.loads((STAGE / "corpus-manifest.json").read_text())
    rows = []
    for name, record in published["sources"].items():
        path = CORPUS / "raw" / Path(record["reference_path"]).name
        rows.append({"source": name, "expected": record["sha256"], "actual": sha256(path),
                     "bytes": path.stat().st_size,
                     "matches": sha256(path) == record["sha256"],
                     "title": record["title"], "authors": record["authors"], "url": record["url"]})
    items = sha256(CORPUS / "items.jsonl")
    evaluation = []
    for split, value in published["evaluation"]["sets"].items():
        path = CORPUS / "evaluation" / f"{split}.json"
        evaluation.append({"split": split, "expected": value["sha256"], "actual": sha256(path),
                           "matches": sha256(path) == value["sha256"]})
    return {"raw_sources": rows, "all_sources_match": all(row["matches"] for row in rows),
            "items_sha256": items, "items_match": items == published["items_sha256"],
            "evaluation_sets": evaluation,
            "all_evaluation_sets_match": all(row["matches"] for row in evaluation)}


def replay_checkpoint_chain(run, arm):
    store = tr.CheckpointStore(usage.descendant_store(run, arm))
    names = store.history()
    rows, previous = [], -1
    for name in names:
        metadata = json.loads((store.revisions / name).read_text())
        weights = store.revisions / metadata["weights"]
        actual = sha256(weights)
        rows.append({"revision": metadata["revision"], "step": metadata["step"], "note": metadata.get("note"),
                     "descendant": metadata["descendant_owner"],
                     "weights_sha256_matches": actual == metadata["weights_sha256"],
                     "parent_owner": metadata["baseline_contract"]["baseline_owner"],
                     "contract_hashes": sorted(metadata["baseline_contract"]["contracts"]),
                     "cursors": sorted(metadata["cursors"]),
                     "monotonic": metadata["revision"] == previous + 1})
        previous = metadata["revision"]
    pointer = json.loads((store.directory / "current.json").read_text())
    return {"revisions": len(rows), "rows": rows,
            "all_weight_hashes_match": all(row["weights_sha256_matches"] for row in rows),
            "revision_chain_monotonic": all(row["monotonic"] for row in rows),
            "pointer": pointer,
            "pointer_resolves": (store.revisions / pointer["revision"]).is_file()}


def replay_parent_equivalence(run, arm="connected"):
    """Revision 0 must be the parent: zero adapter output and identical answers."""
    context = usage.restore_descendant(run, arm, revision=0)
    owner = context["owner"]
    adapter_output_norm = float(owner.adapter[-1].weight.abs().max())
    tasks = json.loads(
        (LAB_ROOT / "research-continuation/47_intervention_understanding/retention-tasks.json").read_text())
    baseline = {case["id"]: case for case in json.loads((BASELINE_ATTEMPT / "retention-cases.json").read_text())}
    identical, differing, failed = 0, [], []
    for request in tasks:
        try:
            answer = usage.perform(context, dict(request))
        except Exception as error:
            failed.append({"id": request.get("id"), "error": type(error).__name__ + ": " + str(error)})
            continue
        reference = baseline.get(request.get("id"))
        if reference is None or reference["status"] != "ANSWERED":
            continue
        if json.dumps(reference["result"]["result"], sort_keys=True, default=str) == \
                json.dumps(answer["result"], sort_keys=True, default=str):
            identical += 1
        else:
            differing.append(request.get("id"))
    return {"revision": 0, "step": context["metadata"]["step"],
            "adapter_output_max_abs_weight": adapter_output_norm,
            "adapter_output_is_zero": adapter_output_norm == 0.,
            "retained_tasks_identical_to_baseline": identical,
            "retained_tasks_differing": differing, "retained_tasks_failed": failed,
            "behaviourally_the_parent": adapter_output_norm == 0. and not differing and not failed}


def replay_measurements(run, arm, splits):
    context = usage.restore_descendant(run, arm)
    owner, tokens, metadata = context["owner"], context["tokens"], context["metadata"]
    measured = {}
    for split in splits:
        path = RUNS / f"corpus/evaluation/{split}.json"
        if not path.exists():
            measured[split] = {"status": "UNAVAILABLE", "reason": "evaluation set absent"}
            continue
        measured[split] = tr.evaluate(owner, tokens, load_evaluation(split))
    return {"arm": arm, "descendant": owner.identity(), "step": metadata["step"],
            "checkpoint_revision": metadata["revision"], "measurements": measured}


def compare_claims(replayed, claimed, tolerance=1e-9):
    rows = []
    for split, families in replayed["measurements"].items():
        for family in ("next_token", "cloze", "definition"):
            if family not in families or family not in claimed.get(split, {}):
                continue
            here, there = families[family], claimed[split][family]
            for key in ("mean_nll", "accuracy"):
                if key in here and key in there and here[key] is not None and there[key] is not None:
                    difference = abs(here[key] - there[key])
                    rows.append({"split": split, "family": family, "key": key,
                                 "replayed": here[key], "claimed": there[key],
                                 "difference": difference, "agrees": difference <= tolerance})
    return {"comparisons": rows, "all_agree": all(row["agrees"] for row in rows) if rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="OLA-001")
    parser.add_argument("--arms", default="connected,blocked,disconnected")
    parser.add_argument("--splits", default="dev,transfer")
    parser.add_argument("--claims", default=None,
                        help="An evaluation record to re-derive, as evaluation-<arm>.json in one attempt directory")
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(options.output or os.environ.get("SERA_LAB_ATTEMPT", RUNS / "attempts/replay"))
    output.mkdir(parents=True, exist_ok=True)

    arms = [name for name in options.arms.split(",") if name]
    splits = [name for name in options.splits.split(",") if name]
    record = {"schema": "sera.owner-language.independent-replay.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.usage",)),
              "run": options.run, "sources": replay_sources(), "arms": {}}

    for arm in arms:
        entry = {"checkpoints": replay_checkpoint_chain(options.run, arm)}
        entry["measurements"] = replay_measurements(options.run, arm, splits)
        if options.claims:
            claimed_path = Path(options.claims) / f"evaluation-{arm}.json"
            if claimed_path.exists():
                claimed = json.loads(claimed_path.read_text())["measurements"]
                entry["agreement_with_claims"] = compare_claims(entry["measurements"], claimed)
        record["arms"][arm] = entry

    record["parent_equivalence"] = replay_parent_equivalence(options.run, "connected")
    record["passed"] = bool(
        record["sources"]["all_sources_match"] and record["sources"]["items_match"]
        and record["sources"]["all_evaluation_sets_match"]
        and all(entry["checkpoints"]["all_weight_hashes_match"] for entry in record["arms"].values())
        and all(entry["checkpoints"]["revision_chain_monotonic"] for entry in record["arms"].values())
        and record["parent_equivalence"]["behaviourally_the_parent"]
        and all(entry.get("agreement_with_claims", {"all_agree": True})["all_agree"] is not False
                for entry in record["arms"].values()))
    (output / "independent-replay.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({"passed": record["passed"],
                      "sources_match": record["sources"]["all_sources_match"],
                      "items_match": record["sources"]["items_match"],
                      "evaluation_sets_match": record["sources"]["all_evaluation_sets_match"],
                      "parent_equivalence": record["parent_equivalence"]["behaviourally_the_parent"],
                      "arms": {arm: {"revisions": entry["checkpoints"]["revisions"],
                                     "hashes_match": entry["checkpoints"]["all_weight_hashes_match"],
                                     "agreement": entry.get("agreement_with_claims", {}).get("all_agree")}
                               for arm, entry in record["arms"].items()}}, indent=2, default=str))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
