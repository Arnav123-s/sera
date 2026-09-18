"""Credit reconciliation and exact practical lifecycle checks, preserving PS-001."""

import argparse
import copy
import gc
import json

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.solution_credit import ReconciledSession, correction_digest, known_execution
from scripts.solution_use import perform, restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/44_solution_portfolios"
RUN = ROOT / "runs/PS-study-001"


def reconcile():
    if (OUT / "credit-audit.json").exists():
        raise FileExistsError("Preserve completed credit reconciliation")
    original, growth = restore("runs/sera-solutions-live")
    original_identity = original.identity()
    hashes = {k: digest(v.detach().tolist()) for k, v in original.owner.state_dict().items()}
    session = ReconciledSession(original)
    changed = [k for k, v in session.owner.state_dict().items() if hashes[k] != digest(v.detach().tolist())]
    if any(not k.startswith("solution_policy.") for k in changed):
        raise ValueError("Credit correction altered learned answer weights")
    if growth.base.base.base.base.base.base.grounded.gate["autonomous"]:
        raise ValueError("Source applicability gate changed")
    snapshot = session.state()
    expected = session.identity()
    points = sum(v["points"] for v in session.credits.values())
    from experiments.self_chosen import equations as eq
    newly_executable = []
    for value in session.questions.values():
        q = value["question"]
        available = sorted(set(eq.layout(q["domain"])) - {q["target"], *(q["missing"] or [])})
        if value["solutions"] and not known_execution({"domain": q["domain"], "target": q["target"], "requires": available},
                                                      session.base.base.base.records):
            newly_executable.append(q["id"])
    audit = {"passed": True, "owner": expected, "parent_owner": original_identity,
             "preserved_model_tensors": len(hashes)-len(changed), "changed_tensors": changed,
             "corrected_records": session.corrections, "correction_hash": correction_digest(session.corrections),
             "original_points": read(OUT / "selection.json")["points"], "corrected_points": points,
             "novel_input_routes": sum(r["points"] > 0 for r in session.records.values()),
             "newly_executable_questions": newly_executable,
             "retained_solutions": len(session.records), "source_gate": False,
             "policy": "Original predictor archived; neutral exhaustive procedure scheduling retained. No holdout retuning.",
             "scientific_proposals_and_final_evaluation_changed": False}
    del original, session, growth
    gc.collect()
    original, _ = restore("runs/sera-solutions-live")
    replay = ReconciledSession(original, snapshot)
    if replay.identity() != expected or replay.state() != snapshot:
        raise ValueError("Reconciled owner did not restore exactly")
    audit["exact_restore"] = True
    write(OUT / "credit-audit.json", audit)
    store = Store(ROOT / "runs/sera-solution-progress-live")
    if store.read() is not None:
        raise FileExistsError("Preserve the current progressing learner")
    store.commit({"schema": "sera.solution-progress.store.1", "parent_store": "runs/sera-solutions-live",
                  "parent_owner": original_identity, "parent_current": sha(ROOT / "runs/sera-solutions-live/current.json"),
                  "state": snapshot, "owner": expected, "audit": sha(OUT / "audit.json"),
                  "final": sha(OUT / "final.json"), "credit_audit": sha(OUT / "credit-audit.json")}, None)
    print(json.dumps(audit, indent=2))


def runtime():
    from experiments.self_chosen import equations as eq
    if (OUT / "runtime-audit.json").exists():
        raise FileExistsError("Preserve completed runtime verification")
    session, growth = restore()
    choices = [v["question"] for v in session.questions.values() if v.get("solutions")]
    frontier = session.base.base.base
    with torch.no_grad():
        scores = frontier.owner.self_question_policy(frontier.features(choices)).squeeze(-1).tolist()
    q = max(zip(scores, choices, strict=True), key=lambda p: (p[0], p[1]["id"]))[1]
    row = eq.imagine(session.owner, q["domain"], 449831, count=1)[0]
    observed = {n: str(v) for n, v in row.items() if n not in {q["target"], *(q["missing"] or [])}}
    checkpoint = RUN / "runtime-pending.json"
    calls = []
    def interrupt(state):
        write(checkpoint, state)
        calls.append(state["pending"] is not None)
        raise InterruptedError("Deliberate interruption after durable proposal and before grading")
    try:
        session.autonomous_round(q, observed, interrupt)
    except InterruptedError:
        pass
    if calls != [True] or session.pending is None:
        raise ValueError("No exact pending-proposal checkpoint was saved")
    pending = read(checkpoint)
    del session, growth
    gc.collect()
    parent, growth = restore("runs/sera-solutions-live")
    session = ReconciledSession(parent, pending)
    snapshots = []
    answer = session.autonomous_round(q, observed, lambda state: snapshots.append(copy.deepcopy(state)))
    if answer["answers"] != [str(row[q["target"]])]:
        raise ValueError("Runtime did not solve its original imagined question")
    identity, points = session.identity(), sum(c["points"] for c in session.credits.values())
    session.autonomous_round(q, observed, lambda state: snapshots.append(copy.deepcopy(state)))
    if session.identity() != identity or sum(c["points"] for c in session.credits.values()) != points:
        raise ValueError("A repeated completed question changed weights or credit")
    write(RUN / "runtime-successor.json", session.state())
    result = {"passed": True, "selected_question": q, "selection": "retained question-policy score over owned answerable questions",
              "input_origin": "actual learned forward imagination; target withheld from the proposal input",
              "expected": str(row[q["target"]]), "answer": answer, "interrupted_before_feedback": True,
              "exact_pending_restore": True, "repeated_question_no_new_credit_or_weight_change": True,
              "branch_owner": identity, "live_owner_preserved": True}
    write(OUT / "runtime-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "answer"}, indent=2))


def examples():
    from experiments.self_chosen import equations as eq
    if (OUT / "example-results.json").exists():
        raise FileExistsError("Preserve completed practical examples")
    session, growth = restore()
    tasks = []
    for domain in ("motion_0", "motion_1", "motion_2", "polynomials"):
        records = [r for r in session.records.values() if r["candidate"]["domain"] == domain and r["points"] > 0]
        chosen = max(records, key=lambda r: (len(r["questions"]), -r["candidate"]["cost"], r["candidate"]["id"]))
        c = chosen["candidate"]
        row = eq.reference_rows(domain, 449947, count=1)[0]
        observations = {n: str(v) for n, v in row.items() if n != c["target"]}
        tasks.append({"id": domain, "kind": "solve_solution", "domain": domain, "target": c["target"], "observations": observations})
    earlier = read(ROOT / "research-continuation/43_knowledge_gaps/example-tasks.json")
    tasks += earlier
    results = [perform(session, growth, task) for task in tasks]
    write(OUT / "example-tasks.json", tasks)
    write(OUT / "example-results.json", results)
    write(RUN / "example-results.json", results)
    print(json.dumps({"tasks": len(tasks), "owner": session.identity(),
                      "new_answers": [r["result"].get("answers") for r in results[:4]],
                      "routes": [len(r["result"]["routes"]) for r in results[:4]]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reconcile", "runtime", "examples", "all"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "all":
        reconcile()
        runtime()
        examples()
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
