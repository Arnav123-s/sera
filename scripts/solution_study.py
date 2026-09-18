"""Complete an owned question frontier before revealing independent feedback."""

import argparse
import json
from collections import Counter

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.self_chosen import equations as eq
from scripts.portable_gap_use import restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/44_solution_portfolios"
RUN = ROOT / "runs/PS-study-001"


def source_contracts():
    paths = [ROOT / "experiments/solution_search.py", ROOT / "experiments/solution_check.py",
             ROOT / "experiments/solution_owner.py", ROOT / "scripts/solution_study.py", OUT / "PROTOCOL.md"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verified_freeze():
    frozen = read(RUN / "freeze.json")
    if frozen["sources"] != source_contracts() or frozen["inventory"] != sha(RUN / "inventory.json"):
        raise ValueError("Frozen solution research changed; preserve and explicitly migrate")
    return frozen


def prepare():
    if (RUN / "freeze.json").exists():
        raise FileExistsError("Preserve the frozen study")
    session, _ = restore()
    state = read(RUN / "inventory.json")
    if state["owner"] != session.identity():
        raise ValueError("Newer owner requires reconciliation")
    questions = [v["question"] for v in state["questions"]]
    frontier = session.base.base
    with torch.no_grad():
        scores = frontier.owner.self_question_policy(frontier.features(questions)).squeeze(-1).tolist()
    queue = [q for _, q in sorted(zip(scores, questions, strict=True), key=lambda p: (-p[0], p[1]["id"]))]
    frozen = {"sources": source_contracts(), "inventory": sha(RUN / "inventory.json"), "owner": session.identity(),
              "queue": queue, "scores": dict(zip([q["id"] for q in questions], scores, strict=True)),
              "final_seed": 445001, "external_answers": 0, "ordering": "retained question-policy scores, stable ID ties"}
    write(RUN / "freeze.json", frozen)
    write(OUT / "freeze.json", frozen)
    print(f"Frozen {len(queue)} owned questions, 14 declared search procedures per question.")


def search():
    from experiments.solution_search import investigate
    frozen = verified_freeze()
    session, _ = restore()
    if session.identity() != frozen["owner"]:
        raise ValueError("The committed predictor changed")
    completed = []
    for q in frozen["queue"]:
        path = RUN / "proposals" / (q["id"] + ".json")
        if path.exists():
            event = read(path)
            body = {k: v for k, v in event.items() if k != "commitment"}
            if digest(body) != event["commitment"] or event["predictor"] != frozen["owner"]:
                raise ValueError("Changed preserved proposal")
        else:
            event = investigate(session.owner, q)
            event["predictor"] = session.identity()
            event["commitment"] = digest(event)
            write(path, event)
        completed.append({"question": q["id"], "sha256": sha(path), "proposals": len(event["proposals"])})
        write(RUN / "search-state.json", {"completed": completed, "next_index": len(completed),
                                          "queue": digest(frozen["queue"]), "feedback_revealed": False})
        print(f"Question {len(completed)}/{len(frozen['queue'])}: {len(event['proposals'])} distinct answers proposed", flush=True)
    write(RUN / "search-commitment.json", {"questions": completed, "predictor": session.identity(),
                                           "all_finite_searches_exhausted": True, "feedback_revealed": False})
    print("Entire finite frontier committed before independent feedback.", flush=True)


def committed_events():
    frozen = verified_freeze()
    committed = read(RUN / "search-commitment.json")
    if not committed["all_finite_searches_exhausted"] or len(committed["questions"]) != len(frozen["queue"]):
        raise ValueError("Independent feedback waits for complete finite exploration")
    for entry in committed["questions"]:
        path = RUN / "proposals" / (entry["question"] + ".json")
        if sha(path) != entry["sha256"]:
            raise ValueError("Proposal changed after commitment")
        yield read(path)


def grade():
    from experiments.solution_check import certify
    from experiments.solution_owner import SolutionSession, train_policy
    from experiments.verified_completion.credit import encode
    if (OUT / "selection.json").exists():
        raise FileExistsError("Preserve graded selection; never repeat feedback")
    session, _ = restore()
    old = session.base.base.records
    aggregates, practice, counts = {}, [], Counter()
    for event in committed_events():
        q = event["question"]
        checked = []
        for proposal in event["proposals"]:
            c = proposal["candidate"]
            receipt = certify(q, c, event["commitment"], event["predictor"])
            counts["accepted" if receipt["accepted"] else "rejected"] += 1
            checked.append({"candidate": c["id"], "receipt": receipt})
            if receipt["accepted"]:
                record = aggregates.setdefault(c["id"], {"candidate": c, "question": q, "receipt": receipt,
                                      "commitment": event["commitment"], "questions": [], "derivations": []})
                record["questions"].append(q["id"])
                record["derivations"].extend(proposal["derivations"])
        write(RUN / "grades" / (q["id"] + ".json"), {"question": q, "commitment": event["commitment"], "checked": checked})
        practice.append(event)
    # All answers remain in the audit. The live portfolio keeps the least costly
    # expression for each distinct input/denominator guard, without a size cap.
    grouped = {}
    for record in sorted(aggregates.values(), key=lambda r: (r["candidate"]["cost"], r["candidate"]["id"])):
        c = record["candidate"]
        key = digest([c["domain"], c["target"], c["requires"], c["denominator"]])
        grouped.setdefault(key, record)
    novel_inputs, per_question, records = set(), Counter(), []
    for record in grouped.values():
        c = record["candidate"]
        familiar = any(r["question"]["domain"] == c["domain"] and r["proposal"]["target"] == c["target"]
                       and "coefficients" in r["proposal"] and set(r["proposal"]["requires"]) <= set(c["requires"])
                       and set(r["proposal"].get("nonzero", [])) <= {"t", "m"} for r in old)
        key = digest([c["domain"], c["target"], c["requires"]])
        points = 0.
        if not familiar and key not in novel_inputs:
            points = 1. + .25 * bool(per_question[record["question"]["id"]])
            novel_inputs.add(key)
            per_question[record["question"]["id"]] += 1
        record["points"] = points
        record["novel_input_route"] = points > 0
        records.append(record)
    reward = {r["candidate"]["id"]: r["points"] for r in records}
    learning_rows = []
    for event in practice:
        for attempt in event["attempts"]:
            ids = {c["id"] for c in attempt["proposals"]}
            learning_rows.append({"question": event["question"], "method": attempt["method"], "degree": attempt["degree"],
                                  "reward": sum(reward.get(key, 0.) for key in ids)})
    owner = SolutionSession(session)
    policy_result = train_policy(owner, learning_rows)
    policy = {"policy": encode(owner.owner.solution_policy.state_dict()), "optimizer": encode(owner.optimizer.state_dict()),
              "result": policy_result, "rows": learning_rows}
    write(RUN / "policy.json", policy)
    selection = {"records": records, "counts": dict(counts), "unique_proved": len(aggregates),
                 "retained": len(records), "points": sum(r["points"] for r in records), "novel_input_routes": len(novel_inputs),
                 "policy": policy_result, "commitment": sha(RUN / "search-commitment.json"),
                 "extra_information_after_search": "Independent proof feedback now available; no new source supplied before exhaustion"}
    write(OUT / "selection.json", selection)
    print(json.dumps({k: v for k, v in selection.items() if k != "records"}, indent=2))


def final():
    from experiments.solution_check import final_check
    frozen = verified_freeze()
    if (OUT / "final.json").exists():
        raise FileExistsError("Preserve the once-only sealed evaluation")
    selection = read(OUT / "selection.json")
    results = []
    for record in selection["records"]:
        candidate = record["candidate"]
        path = RUN / "final" / (candidate["id"] + ".json")
        if path.exists():
            result = read(path)
        else:
            result = final_check(candidate, frozen["final_seed"] + int(candidate["id"][:8], 16))
            write(path, result)
        results.append(result)
    by_domain = {}
    for domain in ("motion_0", "motion_1", "motion_2", "polynomials"):
        ids = {r["candidate"]["id"] for r in selection["records"] if r["candidate"]["domain"] == domain}
        group = [r for r in results if r["candidate"] in ids]
        by_domain[domain] = {"routes": len(group), "correct": sum(r["correct"] for r in group),
                             "guarded": sum(r["guarded"] for r in group), "wrong": sum(len(r["wrong"]) for r in group)}
    result = {"selection": sha(OUT / "selection.json"), "results": results, "by_domain": by_domain,
              "accepted": bool(results) and all(r["accepted"] for r in results),
              "correct": sum(r["correct"] for r in results), "cases": sum(r["n"] for r in results),
              "guarded": sum(r["guarded"] for r in results), "wrong": sum(len(r["wrong"]) for r in results)}
    write(OUT / "final.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


def audit():
    import gc

    from experiments.solution_owner import SolutionSession
    from experiments.verified_completion.credit import decode
    verified_freeze()
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve completed audit")
    selected, evaluated = read(OUT / "selection.json"), read(OUT / "final.json")
    if not evaluated["accepted"] or evaluated["selection"] != sha(OUT / "selection.json"):
        raise ValueError("Final portfolio requires diagnosis before integration")
    base, growth = restore()
    session = SolutionSession(base)
    finals = {r["candidate"]: r for r in evaluated["results"]}
    for record in selected["records"]:
        session.retain(record, finals[record["candidate"]["id"]])
    policy = read(RUN / "policy.json")
    if policy["result"]["selected"]:
        session.owner.solution_policy.load_state_dict(decode(policy["policy"]))
        session.optimizer.load_state_dict(decode(policy["optimizer"]))
    before = read(RUN / "inventory.json")
    for key, expected in before["inherited_tensors"].items():
        if digest(session.owner.state_dict()[key].detach().tolist()) != expected:
            raise ValueError("Inherited learned weights changed")
    for entry in before["questions"]:
        q = entry["question"]
        routes = [r["candidate"]["id"] for r in session.records.values() if q["id"] in r["questions"]]
        session.questions[q["id"]] = {"question": q, "solutions": routes,
                   "status": "VERIFIED_SOLUTIONS_RETURNED" if routes else "FINITE_SEARCH_EXHAUSTED",
                   "prior_routes": entry["retained_routes"], "next_action": "Use the checked routes with their explicit guards" if routes
                   else "Preserve the full search; investigate a richer representation or obtain independent missing evidence"}
    repeat_rejected = False
    if selected["records"]:
        record = selected["records"][0]
        try:
            session.retain(record, finals[record["candidate"]["id"]])
        except ValueError:
            repeat_rejected = True
    gate = growth.base.base.base.base.base.base.grounded.gate["autonomous"]
    if gate or not repeat_rejected or session.owner is not base.owner:
        raise ValueError("Owner identity, source gate or credit integrity failed")
    snapshot = session.state()
    write(RUN / "successor.json", snapshot)
    expected_owner = session.identity()
    del session, base, growth
    gc.collect()
    base, _ = restore()
    replay = SolutionSession(base, snapshot)
    if replay.identity() != expected_owner or replay.state() != snapshot:
        raise ValueError("Exact restoration failed")
    check_count = 0
    from experiments.self_chosen.equations import reference_rows
    for record in replay.records.values():
        c = record["candidate"]
        for row in reference_rows(c["domain"], 448871, 3):
            observations = {k: str(row[k]) for k in c["requires"]}
            answer = replay.solve(c["domain"], c["target"], observations)
            if answer["answers"] and answer["answers"] != [str(row[c["target"]])]:
                raise ValueError("Original-question return failed independent audit")
            check_count += 1
    result = {"passed": True, "owner": expected_owner, "parent_owner": before["owner"],
              "unchanged_inherited_tensors": len(before["inherited_tensors"]), "source_gate": gate,
              "exact_restore": True, "repeated_credit_rejected": repeat_rejected, "same_actual_owner": True,
              "return_to_question_checks": check_count, "records": len(replay.records),
              "questions_with_solutions": sum(bool(q["solutions"]) for q in replay.questions.values()),
              "newly_answered_questions": sum(bool(q["solutions"]) and not q["prior_routes"] for q in replay.questions.values())}
    write(OUT / "audit.json", result)
    print(json.dumps(result, indent=2))


def integrate():
    verified_freeze()
    audit_result = read(OUT / "audit.json")
    if not audit_result["passed"]:
        raise ValueError("Independent audit must pass")
    live = Store(ROOT / "runs/sera-solutions-live")
    if live.read() is not None:
        raise FileExistsError("Preserve the newer live owner")
    value = {"schema": "sera.solution-portfolios.store.1", "parent_store": "runs/sera-gap-inquiry-live",
             "parent_owner": audit_result["parent_owner"], "state": read(RUN / "successor.json"),
             "owner": audit_result["owner"], "audit": sha(OUT / "audit.json"), "final": sha(OUT / "final.json")}
    live.commit(value, None)
    write(OUT / "integration.json", {"admitted": True, "store": "runs/sera-solutions-live", "owner": value["owner"],
              "parent_owner": value["parent_owner"], "records": audit_result["records"],
              "newly_answered_questions": audit_result["newly_answered_questions"], "same_actual_owner": True})
    print(json.dumps(read(OUT / "integration.json"), indent=2))


def run():
    stages = ((RUN / "freeze.json", prepare), (RUN / "search-commitment.json", search),
              (OUT / "selection.json", grade), (OUT / "final.json", final),
              (OUT / "audit.json", audit), (OUT / "integration.json", integrate))
    for artifact, action in stages:
        if artifact.exists():
            print(f"Preserving completed {action.__name__}: {artifact.name}", flush=True)
        else:
            print(f"Starting {action.__name__}", flush=True)
            action()


def inventory():
    if (RUN / "inventory.json").exists():
        raise FileExistsError("Preserve the recorded owner inventory")
    session, growth = restore()
    frontier = session.base.base
    questions = {}
    for event in frontier.events:
        q = event["question"]
        if q["domain"] not in {"motion_0", "motion_1", "motion_2", "polynomials"}:
            continue
        available = set(eq.layout(q["domain"])) - {q["target"], *(q["missing"] or [])}
        applicable = [r["id"] for r in frontier.records if r["question"]["domain"] == q["domain"]
                      and r["proposal"]["target"] == q["target"]
                      and set(r["proposal"]["requires"]) <= available]
        entry = questions.setdefault(q["id"], {"question": q, "events": [], "retained_routes": applicable})
        entry["events"].append(event["id"])
    result = {"owner": session.identity(), "parent_current_sha256": sha(ROOT / "runs/sera-gap-inquiry-live/current.json"),
              "source": "Actual retained FrontierSession.events and owned equation registry",
              "questions": list(questions.values()), "retained_records": len(frontier.records),
              "open_empirical_goals": session.open_goals,
              "inherited_tensors": {k: digest(v.detach().tolist()) for k, v in session.owner.state_dict().items()},
              "domains": dict(Counter(v["question"]["domain"] for v in questions.values())),
              "unanswered_domains": dict(Counter(v["question"]["domain"] for v in questions.values() if not v["retained_routes"]))}
    write(RUN / "inventory.json", result)
    write(OUT / "reconciliation.json", {k: v for k, v in result.items() if k not in {"questions", "inherited_tensors"}})
    print(json.dumps({k: v for k, v in result.items() if k not in {"questions", "inherited_tensors", "open_empirical_goals"}}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inventory", "prepare", "search", "grade", "final", "audit", "integrate", "run"))
    args = p.parse_args()
    torch.set_num_threads(1)
    globals()[args.action]()


if __name__ == "__main__":
    main()
