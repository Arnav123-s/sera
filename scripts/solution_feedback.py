"""Apply corrected proof rewards to the actual owner's procedure-value weights.

Use only the original practice partition and unchanged training rule. The opened
holdout and final cases are not consulted or rescored. Exhaustive exploration
remains the default; trained values are retained for future prospective study.
"""

import gc
import json

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.solution_owner import features
from experiments.task_transfer.runtime import lock
from scripts.solution_use import perform, restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/44_solution_portfolios"
RUN = ROOT / "runs/PS-study-001"


def main():
    torch.set_num_threads(1)
    if (OUT / "feedback-update.json").exists():
        raise FileExistsError("Preserve the completed corrected-feedback update")
    session, growth = restore()
    store = Store(ROOT / "runs/sera-solution-progress-live")
    previous = store.read()
    before = session.identity()
    protected = {k: digest(v.detach().tolist()) for k, v in session.owner.state_dict().items() if not k.startswith("solution_policy.")}
    rewards = {key: record["points"] for key, record in session.records.items()}
    examples = []
    excluded = []
    for q in read(RUN / "freeze.json")["queue"]:
        if not int(q["id"][:8], 16) % 5:
            excluded.append(q["id"])
            continue
        event = read(RUN / "proposals" / (q["id"] + ".json"))
        for attempt in event["attempts"]:
            ids = {c["id"] for c in attempt["proposals"]}
            examples.append({"question": q, "method": attempt["method"], "degree": attempt["degree"],
                             "reward": sum(rewards.get(key, 0.) for key in ids)})
    x = torch.tensor([features(r["question"], r["method"], r["degree"]) for r in examples], dtype=torch.float64)
    y = torch.tensor([r["reward"] for r in examples], dtype=torch.float64)
    with torch.no_grad():
        initial = float((session.owner.solution_policy(x).squeeze(-1) - y).square().mean())
    for _ in range(80):
        session.optimizer.zero_grad(set_to_none=True)
        loss = (session.owner.solution_policy(x).squeeze(-1) - y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(session.owner.solution_policy.parameters(), 2.)
        session.optimizer.step()
    with torch.no_grad():
        fitted = float((session.owner.solution_policy(x).squeeze(-1) - y).square().mean())
    if not torch.isfinite(session.owner.solution_policy.weight).all() or fitted >= initial:
        raise ValueError("Corrected feedback failed its practice-fit check; preserve the existing live owner")
    if any(digest(session.owner.state_dict()[k].detach().tolist()) != h for k, h in protected.items()):
        raise ValueError("Reward update changed retained answer weights")
    tasks = read(OUT / "example-tasks.json")
    results = [perform(session, growth, task) for task in tasks]
    prior_results = read(OUT / "example-results.json")
    if [r["result"] for r in results] != [r["result"] for r in prior_results]:
        raise ValueError("Feedback altered practical answers")
    snapshot, identity = session.state(), session.identity()
    write(RUN / "corrected-feedback-rows.json", examples)
    write(RUN / "corrected-feedback-successor.json", snapshot)
    write(OUT / "feedback-example-results.json", results)
    evidence = {"owner_before": before, "owner": identity, "rows": len(examples), "excluded_questions": len(excluded),
                "excluded_question_ids_sha256": digest(excluded), "epochs": 80, "learning_rate": .02,
                "practice_mse_before": initial, "practice_mse_after": fitted, "changed_weights": "solution_policy only",
                "protected_tensors": len(protected), "unchanged_practical_answers": len(results),
                "opened_holdout_or_final_used": False, "new_points_for_rehearsal": 0,
                "scheduling": "Complete all declared approaches; no policy-superiority admission inferred from practice fit",
                "corrected_credit": sha(OUT / "credit-audit.json"), "training_rows": sha(RUN / "corrected-feedback-rows.json")}
    with lock(store.directory):
        if store.read() != previous:
            raise ValueError("Another writer advanced the owner")
        store.commit({**previous, "state": snapshot, "owner": identity, "feedback_update": evidence}, previous)
    del session, growth
    gc.collect()
    restored, _ = restore()
    if restored.identity() != identity or restored.state() != snapshot:
        raise ValueError("Corrected feedback weights did not restore exactly")
    evidence["exact_restore"] = True
    write(OUT / "feedback-update.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
