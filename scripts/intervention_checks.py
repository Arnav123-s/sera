"""Reconcile and independently audit the intervention successor."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from scripts.structural_study import restore as restore_parent
from scripts.structural_use import perform as prior_perform

OUT = ROOT / "research-continuation/47_intervention_understanding"
RUN = ROOT / "runs/IU-study-001"
PARENT = ROOT / "runs/sera-structural-live/current.json"


def reconcile():
    target = OUT / "LOCAL_RECONCILIATION.json"
    if target.exists():
        raise FileExistsError("Preserve the original reconciliation")
    repositories = {}
    for name, directory in (("sera", ROOT), ("kavi", ROOT / "research-continuation/00_sources/kavi-pinned")):
        def git(*args):
            return subprocess.run(["git", "-c", "safe.directory=" + directory.as_posix(), *args],
                                  cwd=directory, check=True, text=True, capture_output=True).stdout.strip()
        repositories[name] = {"head": git("rev-parse", "HEAD"),
                              "branch": git("branch", "--show-current"),
                              "dirty_files": git("status", "--short").splitlines()}
    session, growth = restore_parent()
    parent_state = session.state()
    tasks = read(ROOT / "research-continuation/46_structural_refinement/example-tasks.json")
    outputs = [prior_perform(session, growth, task) for task in tasks]
    write(OUT / "retention-tasks.json", tasks)
    write(RUN / "parent-results.json", outputs)
    tensors = {k: {"sha256": digest(v.detach().tolist()), "shape": list(v.shape),
                   "bytes": v.numel() * v.element_size()} for k, v in session.owner.state_dict().items()}
    grounded = growth.base.base.base.base.base.base.grounded
    taught = read(ROOT / "research-continuation/45_counterfactual_inquiry/inventory.json")["grounded_bindings"]
    bindings = {r["entry"]["id"]: grounded.bind(r["entry"]["term"], r["entry"]["text"], r["entry"]["source"]) for r in taught}
    if len(bindings) != 26 or not all(v["admitted"] for v in bindings.values()):
        raise ValueError("Inherited meanings require reconciliation")
    write(RUN / "parent-bindings.json", bindings)
    if session.state() != parent_state:
        raise ValueError("Read-only reconciliation changed the parent")
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    record = {"owner": session.identity(), "parent_current": sha(PARENT), "tensors": tensors,
              "repositories": repositories, "resource_authorization": ledger["unlimited_local_time"],
              "resource_ledger_sha256": sha(ROOT / "runs/v3-batch-001/budget.json"),
              "owned_job": read(ROOT / "runs/v3-batch-001/active.lock"),
              "parent_state": digest(parent_state), "source_gate": grounded.gate,
              "shared_language_owner": grounded.owner is session.owner,
              "route_ids": sorted(session.base.base.records), "practical_tasks": len(tasks),
              "practical_results": sha(RUN / "parent-results.json"), "state_bytes": session.owner.core_state_bytes(),
              "taught_meanings": len(bindings), "binding_results": sha(RUN / "parent-bindings.json"),
              "parent_audit": sha(ROOT / "research-continuation/46_structural_refinement/audit.json"),
              "packet_collector": sha(ROOT / "runs/IU-intake/collector.json")}
    write(target, record)
    print({k: v for k, v in record.items() if k not in ("tensors", "route_ids")}, flush=True)


def release_checks(output):
    if output is None or output.exists() or not output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned verification directory")
    output.mkdir(parents=True)
    checks = [("lint", ["-m", "ruff", "check", "src", "tests", "scripts", "experiments"]),
              ("regression", ["-m", "pytest", "--junitxml", str(output / "junit.xml")]),
              *[(name, [f"scripts/{name}.py"]) for name in (
                  "verify_release", "verify_continuation", "verify_applicability", "verify_v3_release",
                  "verify_transfer_release", "verify_continuing_release", "verify_concept_release")],
              ("cli", ["-m", "sera", "--help"]),
              ("current_examples", ["scripts/sera_current.py", "--input", str(OUT / "example-tasks.json"),
                                    "--output", str(output / "example-results.json")])]
    results = []
    for name, command in checks:
        started = time.perf_counter()
        with (output / (name+".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results.append({"name": name, "command": command, "returncode": result.returncode,
                        "seconds": time.perf_counter()-started})
        write(output / "checks.json", results)
        print(results[-1], flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["reconcile", "release"])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.action == "reconcile":
        reconcile()
    else:
        release_checks(args.output)


if __name__ == "__main__":
    main()
