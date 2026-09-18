"""Finalize guard-aware credit without repeating a proposal or final case."""

import gc
import json

import torch

from experiments.gap_inquiry import ROOT, digest, read, write
from experiments.solution_credit import ReconciledSession
from experiments.solution_owner import features
from experiments.task_transfer.runtime import lock
from scripts.solution_use import perform, restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/44_solution_portfolios"
RUN = ROOT / "runs/PS-study-001"


def main():
    torch.set_num_threads(1)
    if (OUT / "coverage-audit.json").exists():
        raise FileExistsError("Preserve completed coverage reconciliation")
    store = Store(ROOT / "runs/sera-solution-progress-live")
    previous = store.read()
    if previous["owner"] != read(OUT / "feedback-update.json")["owner"] or previous["state"]["pending"]:
        raise ValueError("Newer valid work needs reconciliation before this source migration")
    original, growth = restore("runs/sera-solutions-live")
    if (set(previous["state"]["records"]) != set(original.records)
            or previous["state"]["questions"] != original.questions
            or any(previous["state"]["records"][k]["candidate"] != r["candidate"] for k, r in original.records.items())):
        raise ValueError("Preserve newer learned answers or questions")
    protected = {k: digest(v.detach().tolist()) for k, v in original.owner.state_dict().items() if not k.startswith("solution_policy.")}
    session = ReconciledSession(original)
    rewards = {key: r["points"] for key, r in session.records.items()}
    rows = []
    for q in read(RUN / "freeze.json")["queue"]:
        if not int(q["id"][:8], 16) % 5:
            continue
        event = read(RUN / "proposals" / (q["id"] + ".json"))
        for attempt in event["attempts"]:
            ids = {c["id"] for c in attempt["proposals"]}
            rows.append({"question": q, "method": attempt["method"], "degree": attempt["degree"],
                         "reward": sum(rewards.get(key, 0.) for key in ids)})
    x = torch.tensor([features(r["question"], r["method"], r["degree"]) for r in rows], dtype=torch.float64)
    y = torch.tensor([r["reward"] for r in rows], dtype=torch.float64)
    initial = float(y.square().mean())
    for _ in range(80):
        session.optimizer.zero_grad(set_to_none=True)
        loss = (session.owner.solution_policy(x).squeeze(-1)-y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(session.owner.solution_policy.parameters(), 2.)
        session.optimizer.step()
    with torch.no_grad():
        fitted = float((session.owner.solution_policy(x).squeeze(-1)-y).square().mean())
    if not fitted < initial or any(digest(session.owner.state_dict()[k].detach().tolist()) != v for k, v in protected.items()):
        raise ValueError("Practice update or tensor preservation failed")
    results = [perform(session, growth, task) for task in read(OUT / "example-tasks.json")]
    if [r["result"] for r in results] != [r["result"] for r in read(OUT / "example-results.json")]:
        raise ValueError("Coverage credit changed an answer")
    evidence = {"passed": True, "source_migration": "repairs/source-preservation.json", "previous_owner": previous["owner"],
                "owner": session.identity(), "original_points": 85., "native_reconciled_points": 72.75,
                "points": sum(v["points"] for v in session.credits.values()),
                "credited_routes": sum(r["points"] > 0 for r in session.records.values()),
                "corrections": session.corrections, "retained_solutions": len(session.records),
                "protected_tensors": len(protected), "practice_rows": len(rows), "epochs": 80,
                "practice_mse_before": initial, "practice_mse_after": fitted,
                "holdout_or_final_reopened": False, "unchanged_practical_answers": len(results),
                "policy_admission": "Corrected practice value weights; exhaustive scheduling preserved"}
    snapshot = session.state()
    write(RUN / "coverage-feedback-rows.json", rows)
    write(RUN / "coverage-successor.json", snapshot)
    write(OUT / "coverage-example-results.json", results)
    with lock(store.directory):
        if store.read() != previous:
            raise ValueError("Another writer advanced the owner")
        store.commit({**previous, "state": snapshot, "owner": session.identity(), "coverage_update": evidence}, previous)
    del session, original, growth
    gc.collect()
    session, _ = restore()
    if session.state() != snapshot:
        raise ValueError("Current owner failed exact restored equality")
    evidence["exact_restore"] = True
    evidence["append_only_revisions"] = store.verify_history()
    write(OUT / "coverage-audit.json", evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "corrections"}, indent=2))


if __name__ == "__main__":
    main()
