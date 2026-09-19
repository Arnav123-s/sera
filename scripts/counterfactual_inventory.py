"""Reconcile the actual owner and its retained vocabulary before investigation."""
import json
import subprocess

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from scripts.solution_use import restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/45_counterfactual_inquiry"


def main():
    torch.set_num_threads(1)
    if (OUT / "inventory.json").exists():
        raise FileExistsError("Preserve the reconciled inventory")
    parent = "runs/sera-solution-progress-live"
    stored = Store(ROOT / parent).read()
    session, growth = restore(parent)
    if session.identity() != read(ROOT / "research-continuation/44_solution_portfolios/coverage-audit.json")["owner"]:
        raise ValueError("Reconcile newer valid owner before continuing")
    grounded = growth.base.base.base.base.base.base.grounded
    if grounded.owner is not session.owner:
        raise ValueError("Use the actual shared language/physics owner")
    bindings, excluded = [], []
    current_owner = session.identity()
    for row in grounded.bank:
        try:
            binding = grounded.bind(row["term"], row["text"], row["source"])
        except ValueError as error:
            excluded.append({"entry": row["id"], "term": row["term"], "reason": str(error), "definition_length": len(row["text"])})
            continue
        if binding["admitted"]:
            bindings.append({"entry": row, "binding": binding, "actual_owner": current_owner})
    values = {"head": subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "parent_store": parent, "parent_current": sha(ROOT / parent / "current.json"), "owner": session.identity(),
              "pending_question": session.pending, "store_revision": stored["revision"],
              "solutions": [r["candidate"] for r in session.records.values()],
              "old_questions": [{"id": k, "status": v.get("status"), "question": v.get("question")} for k, v in session.questions.items()],
              "grounded_bindings": bindings, "unbound_entries": excluded, "novel_definition_gate": grounded.gate,
              "parameters": {k: {"shape": list(v.shape), "sha256": digest(v.detach().tolist())} for k, v in session.owner.state_dict().items()},
              "empirical_models": len(session.base.models), "source_gate_unchanged": True,
              "resource_policy": "research-continuation/RESOURCE_POLICY.md", "time_limit_seconds": None,
              "evidence_policy": "Retained executions generate questions and predictions; separate reference evidence follows committed proposals"}
    write(OUT / "inventory.json", values)
    print(json.dumps({"owner": values["owner"], "solution_routes": len(values["solutions"]), "grounded_bindings": len(bindings),
                      "tensor_records": len(values["parameters"]), "source_gate": grounded.gate}, indent=2))


if __name__ == "__main__":
    main()
