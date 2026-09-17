"""Fixed full-source evaluation; final labels only after identities are frozen."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .audit import count
from .data import PREPARED, ROOT, SOURCE, parse, read, sha, write
from .runtime import RequestSession
from .study import measure

OUT = ROOT / "research-continuation/26_stream_curriculum"


def checkpoints(folders):
    result = []
    for folder in folders:
        summary = read(ROOT / folder / "summary.json")
        if (summary["steps"] not in (1800, 5400) or summary["status"] != "planned_updates_complete"
                or summary["unique_training_examples"] != 11514
                or summary["old_tensors_unchanged"] != 145
                or summary["retained_language_requests"] != 96):
            raise ValueError("Incomplete declared full-data acquisition or retention")
        path = summary["checkpoint"]
        if sha(ROOT / path) != summary["checkpoint_sha256"]:
            raise ValueError("Changed full-data checkpoint")
        result.append({"path": path, "sha256": summary["checkpoint_sha256"], "fit": folder})
    return result


def partition(name):
    if name not in ("dev", "test"):
        raise ValueError("An evaluation partition is required")
    manifest = read(SOURCE / "manifest.json")
    path = SOURCE / f"{name}.jsonl"
    if sha(path) != manifest["files"][f"{name}.jsonl"]["sha256"]:
        raise ValueError("Changed evaluation data")
    rows, rejected = [], []
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            if raw["partition"] != name or raw["locale"] != "en-US":
                raise ValueError("Wrong evaluation partition")
            try:
                rows.append(parse(raw))
            except ValueError as error:
                rejected.append({"id": raw["id"], "reason": str(error)})
    return rows, rejected


def summarize(records, training_texts):
    result = count(records)
    novel = [r for r in records if r["text"].casefold() not in training_texts]
    result["novel_text"] = count(novel) if novel else {"examples": 0}
    result["per_domain"] = {domain: count([r for r in records if r["scenario"] == domain])
                            for domain in sorted({r["scenario"] for r in records})}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("dev", "test"))
    parser.add_argument("--fits", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--additional-fits", nargs="*", default=[])
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Fresh evaluator output must be within runs")
    output.mkdir(parents=True, exist_ok=False)
    models = checkpoints(args.fits)
    if args.additional_fits and args.partition != "test":
        raise ValueError("Supplementary sequential models are assessed without reselection")
    supplementary = checkpoints(args.additional_fits)
    full_train = [json.loads(line) for line in (PREPARED / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    training_texts = {row["text"].casefold() for row in full_train}
    selected = None
    if args.partition == "test":
        if args.development is None:
            raise ValueError("Freeze development selection before any official test access")
        development = read(args.development)
        if development["models"] != models:
            raise ValueError("Models changed after development")
        selected = development["selected"]
        # Exclusive cohort receipt prevents a second automatic final run elsewhere.
        final_lock = OUT / "official-test-access.json"
        with final_lock.open("x", encoding="utf-8") as receipt:
            json.dump({"models": models, "supplementary": supplementary, "selected": selected, "output": str(output),
                       "development_sha256": sha(args.development), "source": sha(Path(__file__)),
                       "protocol": sha(OUT / "evaluation-protocol.md")}, receipt, indent=2)
    write(output / "freeze.json", {"models": models, "supplementary": supplementary, "partition": args.partition,
                                    "selected_before_test": selected,
                                    "source": sha(Path(__file__)), "protocol": sha(OUT / "evaluation-protocol.md")})
    rows, rejected = partition(args.partition)
    scores = []
    reusable = {}
    if args.reuse is not None:
        if args.partition != "dev":
            raise ValueError("Only unchanged development predictions may be reused")
        previous = read(args.reuse)
        if previous["partition"] != "dev":
            raise ValueError("Cannot reuse a final assessment as development")
        for i, item in enumerate(previous["scores"]):
            reusable[item["checkpoint"]["path"]] = (item, args.reuse.parent / f"records-{i}.json")
    for index, checkpoint in enumerate([*models, *supplementary]):
        reused = False
        if checkpoint["path"] in reusable:
            prior, path = reusable[checkpoint["path"]]
            if prior["checkpoint"] != checkpoint:
                raise ValueError("Changed model in a reused development measurement")
            records = read(path)
            if len(records) != len(rows) or any((r["id"], r["text"], r["target_intent"], r["target_tags"])
                                               != (v["id"], v["text"], v["intent"], v["tags"])
                                               for r, v in zip(records, rows)):
                raise ValueError("Changed source rows in a reused measurement")
            result = {**count(records), "records": records}
            reused = True
        else:
            session = RequestSession({k: checkpoint[k] for k in ("path", "sha256")})
            result = measure(session.owner, rows, session.owner.stream_config["vocabulary"])
            del session
        write(output / f"records-{index}.json", result["records"])
        actual = summarize(result["records"], training_texts)
        assert all(actual[k] == v for k, v in result.items() if k != "records")
        scores.append({"checkpoint": checkpoint, **actual})
        print(json.dumps({"fit": checkpoint["fit"], "partition": args.partition,
                          "intent": actual["intent_accuracy"], "slots": actual["slot_span_f1"],
                          "frames": actual["frame_accuracy"], "reused": reused}), flush=True)
    if args.partition == "dev":
        best = sorted(scores, key=lambda r: (-r["frame_accuracy"], -r["intent_accuracy"], r["checkpoint"]["path"]))[0]
        selected = {**best["checkpoint"], "qualified_annotation_aid": best["intent_accuracy"] >= .70 and best["slot_span_f1"] >= .50,
                    "criterion": "development exact frames; then intent accuracy; then path"}
    majority = Counter(row["intent"] for row in full_train).most_common(1)[0][0]
    majority_correct = sum(r["intent"] == majority for r in rows)
    summary = {"schema": "sera.full-request-evaluation.1", "partition": args.partition,
               "models": models, "supplementary": supplementary, "selected": selected, "scores": scores,
               "source_rows": len(rows) + len(rejected), "eligible_rows": len(rows), "rejected": rejected,
               "fixed_control": {"intent": majority, "intent_correct": majority_correct,
                                 "intent_accuracy": majority_correct / len(rows), "slot_span_f1": 0.0},
               "evaluation_source": sha(Path(__file__)), "official_test_used_for_training": False}
    write(output / "summary.json", summary)


if __name__ == "__main__":
    main()
