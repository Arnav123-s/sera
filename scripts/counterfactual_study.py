"""Prepare and run a prospective, owner-driven counterfactual investigation."""

import argparse
import json
import time

import numpy as np
import torch

from experiments.counterfactual_core import generate, propose
from experiments.gap_inquiry import ROOT, digest, read, sha, write
from scripts.solution_use import restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/45_counterfactual_inquiry"
RUN = ROOT / "runs/CI-study-001"
STORE = ROOT / "runs/sera-counterfactual-live"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("The prospective experiment is already frozen")
    paths = sorted([*ROOT.glob("experiments/counterfactual_*.py"), *ROOT.glob("scripts/counterfactual_*.py"),
                    OUT / "PROTOCOL.md", OUT / "inventory.json", RUN / "frontier.json"])
    write(OUT / "freeze.json", {"files": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                                "parent_current": sha(ROOT / "runs/sera-solution-progress-live/current.json"),
                                "split": "id mod 8: 0 train, 4 validation, every other value final"})


def verify_freeze():
    record = read(OUT / "freeze.json")
    for path, expected in record["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Changed frozen experiment: {path}")
    if sha(ROOT / "runs/sera-solution-progress-live/current.json") != record["parent_current"]:
        raise ValueError("Parent advanced; reconcile before continuing")


def restore_inquiry(store=STORE):
    from experiments.counterfactual_owner import InquirySession
    base, growth = restore()
    saved = Store(store).read()
    if saved and (saved["parent_current"] != sha(ROOT / "runs/sera-solution-progress-live/current.json")
                  or saved["freeze"] != sha(OUT / "freeze.json")):
        raise ValueError("Changed saved owner lineage or frozen experiment")
    session = InquirySession(base, saved["state"] if saved else None)
    if saved and session.identity() != saved["owner"]:
        raise ValueError("Saved owner wrapper changed")
    return session, growth


def persist(session, previous):
    store = Store(STORE)
    if store.read() != previous:
        raise ValueError("Another session changed the shared inquiry owner")
    return store.commit({"schema": "sera.counterfactual.store.1", "state": session.state(),
                         "owner": session.identity(), "parent_current": sha(ROOT / "runs/sera-solution-progress-live/current.json"),
                         "freeze": sha(OUT / "freeze.json")}, previous)


def proof_event(event, directory):
    from experiments.counterfactual_check import validate_event
    path = directory / (event["commitment"] + ".json")
    if path.exists():
        receipt = read(path)
    else:
        receipt = validate_event(event, event["predictor"])
        write(path, receipt)
    if not receipt["accepted"]:
        raise ValueError("Independent conditional proof failed; preserve and diagnose")
    return receipt


def summarize(rows):
    result = {}
    for policy in sorted({r["policy"] for r in rows}):
        values = [r for r in rows if r["policy"] == policy]
        errors = [e for r in values for e in r["errors"]]
        result[policy] = {"questions": len(values), **{k: sum(r[k] for r in values) for k in
                          ("cases", "covered", "correct_consensus", "wrong_consensus", "observations", "expanded", "unresolved")},
                          "mean_normalized_error": float(np.mean(errors)) if errors else None}
    return result


def run_phase(phase):
    from experiments.counterfactual_check import (
        Simulation,
        evaluate_investigation,
        replay_investigation,
    )
    from experiments.counterfactual_loop import POLICIES, commit, investigate, next_question
    verify_freeze()
    if (OUT / (phase + ".json")).exists():
        raise FileExistsError("Do not repeat completed experiments")
    if phase != "train" and not (OUT / "train.json").exists():
        raise ValueError("Finish prospective procedure practice first")
    if phase == "final" and not (OUT / "selection.json").exists():
        raise ValueError("Freeze the validation-selected procedure first")
    frontier = read(RUN / "frontier.json")
    session, _ = restore_inquiry()
    previous = Store(STORE).read()
    partition = {"train": lambda n: n == 0, "validation": lambda n: n == 4, "final": lambda n: n not in (0, 4)}[phase]
    questions = [q for q in frontier["questions"] if partition(int(q["id"][:8], 16) % 8)]
    directory = RUN / phase
    done = {p.stem: read(p) for p in (directory / "done").glob("*.json")}
    while (question := next_question(questions, done)) is not None:
        start = time.perf_counter()
        folder = directory / question["id"]
        procedure_path = folder / "procedure.json"
        if procedure_path.exists():
            weights = read(procedure_path)["weights"]
        else:
            weights = session.policy_weights()
            write(procedure_path, {"weights": weights, "owner": session.identity()})
        initial_path = folder / "initial.json"
        if initial_path.exists():
            event = read(initial_path)
        else:
            event = commit(propose(question, frontier["matrices"]), session.identity())
            write(initial_path, event)
        if question["id"] not in session.records and event["predictor"] != session.identity():
            # A saved pending predictor is still valid if only append-only state
            # changed; substantive weight changes require a fresh reconciliation.
            if not session.pending or session.pending["predictor"] != event["predictor"]:
                raise ValueError("Resumption predictor changed")
        seed = 45100 + int(question["id"][:8], 16)
        omitted = int(question["id"][8:12], 16) % 3 == 0
        source = Simulation(event, seed, omitted=omitted)
        policies = ("random",) if phase == "train" else POLICIES
        selected = "random" if phase == "train" else "entropy" if phase == "validation" else session.selected_policy
        runs, metrics = {}, []
        for policy in policies:
            checkpoint = folder / (policy + ".json")
            saved = read(checkpoint) if checkpoint.exists() else None
            def boundary(value):
                save_json(checkpoint, value)
            result = investigate(event, frontier["matrices"], source, policy, weights, seed, boundary, saved)
            runs[policy] = result
            if phase != "train":
                evaluated = evaluate_investigation(result, source)
                metrics.append(evaluated)
                write(folder / (policy + "-assessment.json"), evaluated)
        chosen = runs[selected]
        proof = [proof_event(e, directory / "proofs") for e in chosen["events"]]
        replay_path = folder / "replay.json"
        replay = read(replay_path) if replay_path.exists() else replay_investigation(chosen, source, frontier["matrices"])
        write(replay_path, replay)
        if question["id"] not in session.records:
            session.pending = {"goal": question["id"], "predictor": event["predictor"],
                               "checkpoint": (folder / (selected + ".json")).relative_to(ROOT).as_posix()}
            if phase == "train":
                session.learn_procedure(chosen, replay)
            session.retain(chosen, proof, replay, frontier["matrices"],
                           {"path": (folder / (selected + ".json")).relative_to(ROOT).as_posix(), "sha256": sha(folder / (selected + ".json"))})
            previous = persist(session, previous)
        item = {"domain": question["domain"], "axis": question["axis"], "target": question["target"], "question": question["id"],
                "generated": event["generated"], "candidates": len(event["candidates"]), "survivors": len(chosen["survivors"]),
                "followups": len(chosen["followups"]), "rules_added": len(session.records[question["id"]]["rules_added"]),
                "points": session.records[question["id"]]["points"], "omitted_mechanism": omitted,
                "metrics": metrics, "seconds": time.perf_counter() - start}
        write(directory / "done" / (question["id"] + ".json"), item)
        done[question["id"]] = item
        print(json.dumps({k: v for k, v in item.items() if k != "metrics"}), flush=True)
    rows = [r for q in done.values() for r in q["metrics"]]
    summary = {"phase": phase, "questions": len(done), "owner": session.identity(), "rules": len(session.rules),
               "points": sum(r["points"] for r in session.records.values()), "comparison": summarize(rows),
               "policy_weights": session.policy_weights(), "source": sha(OUT / "freeze.json")}
    if phase == "validation":
        comparison = summary["comparison"]
        eligible = [p for p in POLICIES if comparison[p]["wrong_consensus"] == 0]
        if not eligible:
            raise ValueError("No procedure passed validation; preserve all results and diagnose")
        best = max(eligible, key=lambda p: (comparison[p]["correct_consensus"], comparison[p]["covered"],
                                         -comparison[p]["unresolved"], -comparison[p]["observations"], -POLICIES.index(p)))
        session.selected_policy = best
        previous = persist(session, previous)
        write(OUT / "selection.json", {"policy": best, "comparison": comparison, "owner": session.identity(),
                                      "head_weights": session.policy_weights(), "criterion": "Zero wrong consensus, most correct consensus, coverage, completion, then cost"})
    write(OUT / (phase + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


def prepare():
    if (RUN / "frontier.json").exists():
        raise FileExistsError("Preserve the current question frontier")
    session, _ = restore()
    inventory = read(OUT / "inventory.json")
    if session.identity() != inventory["owner"]:
        raise ValueError("Newer owner needs reconciliation")
    questions, matrices = generate(session, inventory)
    write(RUN / "frontier.json", {"owner": session.identity(), "questions": questions, "matrices": matrices,
                                  "inventory": sha(OUT / "inventory.json")})
    print(json.dumps({"questions": len(questions), "domains": sorted({q["domain"] for q in questions}),
                      "unique_structural_models": len({m["id"] for q in questions for m in q["models"]}),
                      "word_linked_questions": sum(any(q["word_bindings"].values()) for q in questions)}, indent=2))


def development():
    from experiments.counterfactual_check import validate_event

    frontier = read(RUN / "frontier.json")
    directory = RUN / "development-v1"
    if directory.exists():
        raise FileExistsError("Preserve completed development evidence")
    directory.mkdir()
    chosen = []
    for domain in sorted({q["domain"] for q in frontier["questions"]}):
        eligible = [q for q in frontier["questions"] if q["domain"] == domain and int(q["id"][:8], 16) % 4 == 0]
        chosen.extend(sorted(eligible, key=lambda q: q["id"])[:2])
    write(directory / "sources.json", {p.name: sha(p) for p in (ROOT / "experiments").glob("counterfactual_*.py")})
    results = []
    for question in chosen:
        start = time.perf_counter()
        event = propose(question, frontier["matrices"])
        event["predictor"] = frontier["owner"]
        event["commitment"] = digest({k: v for k, v in event.items() if k != "commitment"})
        write(directory / (question["id"] + "-proposals.json"), event)
        checked = validate_event(event, frontier["owner"])
        write(directory / (question["id"] + "-check.json"), checked)
        result = {"question": question["id"], "domain": question["domain"], "axis": question["axis"], "target": question["target"],
                  "generated": event["generated"], "unique": len(event["candidates"]), "rejected_executions": len(event["failures"]),
                  "proved": sum(p["accepted"] for p in checked["proofs"]), "seconds": time.perf_counter() - start}
        results.append(result)
        write(directory / "summary.json", results)
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--development", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--phase", choices=("train", "validation", "final"))
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        prepare()
    elif args.development:
        development()
    elif args.freeze:
        freeze()
    elif args.phase:
        run_phase(args.phase)
    else:
        raise ValueError("Choose a declared study phase")


if __name__ == "__main__":
    main()
