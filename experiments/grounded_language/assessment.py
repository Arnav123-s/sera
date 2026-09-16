"""Evaluate forgetting, targeted acquisition and declared finite guards."""

import argparse
import copy
import json
import time
import zipfile
from pathlib import Path

import torch

from experiments.cross_route_transfer.study import evaluate_retention
from sera.session_state import model_identity
from workbench.model import Learner

from .acquisition import guarded_interpret, lesson, validate_shapes
from .data import examples
from .model import fingerprint
from .study import BASE, RELEASE, ROOT, evaluate, load_selected, packed, read, sha, write


def assess(args):
    torch.set_num_threads(1)
    destination = RELEASE/args.name
    destination.mkdir(parents=True, exist_ok=False)
    source = RELEASE/args.source
    source_result = read(source/"result.json")
    final = not source_result["config"]["pilot"]
    protocol = {"source": args.source, "source_result_sha256": sha(source/"result.json"),
                "final": final, "seed": args.seed, "lesson_steps": args.lesson_steps,
                "retention_samples": 128, "novel_word_examples": 48, "novel_wording_examples": 192,
                "lesson_selection": "Lowest development cross-entropy; gate at .99 development exact translation; final is read only after learning.",
                "guard": "Supplied sentence shape support + explicit vocabulary admission + previously selected confidence filter.",
                "sources": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("acquisition.py"))},
                "claims": "Neither lexical admission nor the supplied curriculum demonstrates general English or learned eta."}
    write(destination/"protocol.json", protocol)
    with zipfile.ZipFile(destination/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["sources"]:
            archive.write(Path(__file__).with_name(name), name)
    started = time.perf_counter()
    learner, result = load_selected(source)
    owner = learner.session.owner
    threshold = result["calibration"]["threshold"]
    original_identity = model_identity(owner)
    split = "final" if final else "development"
    reports, raw = {}, {}
    if not args.no_retention:
        predecessor = Learner(read(BASE))
        cfg = {"retention_samples": 128, "typed_samples": 128}
        parent_scores = evaluate_retention(predecessor.solver, args.seed+1, cfg)
        successor_scores = evaluate_retention(learner.solver, args.seed+1, cfg)
        drops = {name: value-successor_scores["groups"][name] for name, value in parent_scores["groups"].items()}
        reports["retention"] = {"groups": len(drops), "drops": drops, "maximum_drop": max(drops.values()),
                                "exact_scores_equal": parent_scores["scores"] == successor_scores["scores"],
                                "seed": args.seed+1}
        packed(destination/"retention.json.gz", {"predecessor": parent_scores, "successor": successor_scores})
    checkpoints = ROOT/"runs/grounded-language"/args.name
    checkpoints.mkdir(parents=True, exist_ok=False)
    # A branch per kind prevents one lesson from silently teaching the other.
    for kind in ("word", "wording"):
        candidate = copy.deepcopy(owner)
        teaching = examples(args.seed+10, 48 if kind == "word" else 192, "train",
                            lesson=kind == "word", wording="novel" if kind == "wording" else "known")
        development = examples(args.seed+20, 128, "development", lesson=kind == "word",
                               wording="novel" if kind == "wording" else "known")
        packed(destination/(kind+"-teaching.json.gz"), {"teaching": teaching, "development": development})
        receipts = []
        def save(state, lesson_kind=kind):
            path = checkpoints/f"{lesson_kind}-step-{state['step']:05d}.pt"
            if path.exists():
                raise FileExistsError("Lesson checkpoints are immutable")
            torch.save({**state, "schema": "sera.language-lesson.resume.1", "kind": lesson_kind,
                        "seed": args.seed+40, "source_result_sha256": sha(source/"result.json"),
                        "protocol_sha256": sha(destination/"protocol.json"), "language_source": fingerprint(),
                        "teaching": teaching, "development": development, "total_steps": args.lesson_steps}, path)
            receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
            write(destination/"status.json", {"status": "LESSON", "kind": lesson_kind, "latest": receipts[-1]})
        acquired = lesson(candidate, teaching, development, kind=kind, steps=args.lesson_steps, seed=args.seed+40, checkpoint=save)
        # Final cases first materialize after all training and selection are complete.
        evaluation = examples(args.seed+30, 512, split, lesson=kind == "word",
                              wording="novel" if kind == "wording" else "known")
        before, before_raw = evaluate(owner, evaluation, threshold=threshold)
        after, after_raw = evaluate(candidate, evaluation, threshold=threshold)
        retained, retained_raw = evaluate(candidate, examples(args.seed+50, 512, split), threshold=threshold)
        validation = validate_shapes(candidate, development+examples(args.seed+25, 384, "calibration"))
        shapes = validation["shapes"]
        decisions = [guarded_interpret(candidate, row["text"], threshold=threshold, shapes=shapes) for row in evaluation]
        accepted = [i for i, row in enumerate(decisions) if row["status"] == "ACCEPTED"]
        guarded = {"accepted": len(accepted), "coverage": len(accepted)/len(evaluation),
                   "translation_accuracy": sum(after_raw[i]["translation_correct"] for i in accepted)/len(accepted) if accepted else None}
        path = checkpoints/(kind+"-selected.pt")
        torch.save({"schema": "sera.language-lesson.selected.1", "state": candidate.state_dict(),
                    "owner_config": candidate.export_config(), "owner_sha256": model_identity(candidate),
                    "source_result_sha256": sha(source/"result.json"), "protocol_sha256": sha(destination/"protocol.json"),
                    "base_snapshot_sha256": sha(BASE), "source": args.source, "kind": kind,
                    "shapes": shapes, "threshold": threshold}, path)
        reports[kind] = {"before": before, "after": after, "known_retention": retained, "acquisition": acquired,
                         "shape_validation": validation,
                         "guarded": guarded, "checkpoints": receipts,
                         "selected_checkpoint": {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)},
                         "selected_owner": model_identity(candidate)}
        raw[kind] = {"before": before_raw, "after": after_raw, "known_retention": retained_raw, "decisions": decisions}
    if model_identity(owner) != original_identity:
        raise ValueError("Assessment changed the selected predecessor")
    packed(destination/"raw.json.gz", raw)
    reports.update(status="COMPLETE", source=args.source, protocol_sha256=sha(destination/"protocol.json"),
                   worker_seconds=time.perf_counter()-started, selected_predecessor_unchanged=True)
    write(destination/"result.json", reports)
    write(destination/"status.json", {"status": "COMPLETE", "result_sha256": sha(destination/"result.json")})
    print(json.dumps({"status": "COMPLETE", "source": args.source, "seconds": reports["worker_seconds"],
                      "maximum_retention_drop": reports.get("retention", {}).get("maximum_drop"),
                      "lessons": {k: {"before": reports[k]["before"]["exact_translation"],
                                       "after": reports[k]["after"]["exact_translation"],
                                       "known": reports[k]["known_retention"]["exact_translation"],
                                       "guard": reports[k]["guarded"]} for k in ("word", "wording")}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=160001)
    parser.add_argument("--lesson-steps", type=int, default=120)
    parser.add_argument("--no-retention", action="store_true")
    assess(parser.parse_args())


if __name__ == "__main__":
    main()
