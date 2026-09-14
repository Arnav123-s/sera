"""Assemble a usable shared solver, train its bounded R2/controller, then teach binding."""

import argparse
import json
from pathlib import Path

import torch

from sera.accounting import Costs
from sera.connected import control_table, intervene
from sera.curriculum import ImprovementPolicy, fit_policy
from sera.environments import WorldSpec
from sera.experience import EvidenceReplay
from sera.shared import make_shared_solver
from sera.shared_archive import load_shared_checkpoint
from sera.shared_continual import initialize_shared_store, learn_binding
from sera.shared_learning import SharedEvidence
from sera.solver import SolverStore, Work, tensor_digest
from sera.storage import write_json
from sera.study import policy_episode, policy_summary
from sera.training import environment, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("runs/shared-learner-repaired/0/delta"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.review.exists():
        raise FileExistsError("Keep previous solvers and reviews; choose fresh paths")
    args.review.mkdir(parents=True)
    torch.set_num_threads(1)
    costs, work = Costs(), Work()
    report = {"environment": environment(), "source_trial": str(args.study),
              "selection": "Seed 0 delta preselected before canonical query results; no best-seed selection"}
    try:
        assemble(args, report, costs, work)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(costs=costs.record(), work=work.record())
        write_json(args.review / "assembly.json", report)
    print("Shared solver:", args.output, "binding update:", report["learning"]["status"], flush=True)


def assemble(args, report, costs, work):
    trial = json.loads((args.study / "trial.json").read_text(encoding="utf-8"))
    if trial["status"] != "completed" or trial["environment"]["source_sha256"] != source_hash():
        raise ValueError("Assembly requires a completed trial from this exact source")
    raw = trial["world"]
    spec = WorldSpec(raw["identifier"], tuple(tuple(r) for r in raw["table"]), tuple(raw["colors"]), raw["resettable"])
    evidence = SharedEvidence.load(args.study / "evidence")
    core = load_shared_checkpoint(args.study / "base.pt")
    base = make_shared_solver(core)
    report["base_tensor_sha256"] = tensor_digest(core)
    with costs.phase("r2-training-and-verified-library", work):
        report["before_control"] = control_table(base, spec, work=work)
        acquired = EvidenceReplay(evidence.world.records)
        solver, report["r2_construction"] = intervene(base, spec, acquired, evidence.world, method="program",
                                                      seed=1_100_000, steps=64, work=work)
        report["after_control"] = control_table(solver, spec, work=work)
        evidence.world = acquired
        if tensor_digest(solver.components["r1"]) != report["base_tensor_sha256"]:
            raise ValueError("Program construction unexpectedly changed the pretrained shared owner")
        report["r2_snapshot"] = SolverStore(args.review / "r2-snapshot").initialize(solver)
    methods = ("none", "update", "replay", "planning", "program")
    groups = {}
    for split, count in (("meta-train", 8), ("meta-validation", 3)):
        with costs.phase(f"controller-evidence/{split}"):
            groups[split] = [policy_episode(solver, spec, evidence.world, seed=1100, index=index, split=split,
                output=args.review / "policy-episodes", steps=12, methods=methods, feature_count=16, samples=32)
                for index in range(count)]
    with costs.phase("controller-parameter-learning", work):
        torch.manual_seed(1100)
        policy = ImprovementPolicy(features=16, methods=methods)
        report["controller_training"] = fit_policy(policy, groups["meta-train"], groups["meta-validation"], steps=400, seed=1100)
        work.add("controller_optimizer_steps", 400)
        work.add("controller_training_episode_draws", 400*len(groups["meta-train"]))
        solver.components["controller"] = policy
        torch.save({"config": policy.export_config(), "state": policy.state_dict()}, args.review / "controller.pt")
    with costs.phase("controller-sealed-evaluation"):
        groups["meta-test"] = [policy_episode(solver, spec, evidence.world, seed=1100, index=index, split="meta-test",
            output=args.review / "policy-episodes", steps=12, methods=methods, feature_count=16, samples=32) for index in range(4)]
        report["controller_test"] = policy_summary(policy, groups["meta-test"])
    report["controller_scope"] = (
        "Fresh learned finite intervention selector for this shared core, trained against a fixed starting solver. "
        "It enables ordinary world continuation. This bootstrap does not establish improved sequential meta-learning; "
        "diagnostic rules and method grammar remain supplied. Meta targets use world scores; production gates also check typed/sequence retention.")
    with costs.phase("persist-connected-shared-solver", work):
        report["initial"] = initialize_shared_store(args.output, solver, evidence, [spec])
    with costs.phase("ordinary-corrective-binding"):
        report["learning"] = learn_binding(args.output, seed=1_200_000, support_count=128, steps=1024,
                                            method="scoped", samples=1024)
    report["policy_episodes"] = {name: [row["episode_id"] for row in rows] for name, rows in groups.items()}
    report["cost_scope"] = "Whole invocation CPU/RSS/time include all phases. Policy counterfactual operation counts are recorded in their episode files; the top-level Work object is not their sum. Nested learning costs must not be added to the enclosing invocation twice."


if __name__ == "__main__":
    main()
