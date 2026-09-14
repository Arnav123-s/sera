"""SERA research command line. Every costly action has an explicit finite budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sera.engine import improve, rollback, run_world
from sera.evaluation import evaluate
from sera.experiments import benchmark, instrument_experiment, integrated_run
from sera.models import KINDS, ModelConfig
from sera.solver import load_solver
from sera.storage import Journal, write_json
from sera.training import TrainConfig, adaptation_experiment, train


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def parser():
    p = argparse.ArgumentParser(description="SERA — State-Space Engine for Reasoning and Adaptation")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "benchmark", "run", "world", "instrument"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument(
            "--steps", type=positive, default=500 if name in {"train", "benchmark", "run"} else 300
        )
        child.add_argument("--seed", type=int, default=0)
        if name in {"train", "benchmark"}:
            child.add_argument("--resume", action="store_true")
        if name == "train":
            child.add_argument("--kind", choices=KINDS, default="delta")
        if name == "benchmark":
            child.add_argument(
                "--kinds",
                nargs="+",
                choices=KINDS,
                default=["delta", "gru", "rotor", "real", "hybrid"],
            )
            child.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        if name in {"benchmark", "run"}:
            child.add_argument(
                "--samples", type=positive, default=512 if name == "benchmark" else 1024
            )
        if name == "instrument":
            child.add_argument("--real", action="store_true")
    for name in ("evaluate", "adapt", "improve"):
        child = sub.add_parser(name)
        child.add_argument("checkpoint", type=Path)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--samples", type=positive, default=1024)
        if name == "improve":
            child.add_argument("--max-queries", type=positive, default=100)
            child.add_argument("--task", type=int, choices=range(5), default=2)
    child = sub.add_parser("status")
    child.add_argument("directory", type=Path)
    child = sub.add_parser("rollback")
    child.add_argument("directory", type=Path)
    child = sub.add_parser("study")
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--steps", type=positive, default=900)
    child.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    child.add_argument("--meta-train", type=positive, default=8)
    child.add_argument("--meta-validation", type=positive, default=4)
    child.add_argument("--meta-test", type=positive, default=4)
    child.add_argument("--inner-steps", type=positive, default=16)
    child.add_argument("--instrument-steps", type=positive, default=200)
    for name in ("solve", "learn"):
        child = sub.add_parser(name)
        child.add_argument("directory", type=Path)
        child.add_argument("--world-seed", type=int, required=True)
        child.add_argument("--family", choices=("rotation", "permutation", "reset"), default="rotation")
        if name == "solve":
            child.add_argument("--start", type=int, choices=range(4), required=True)
            child.add_argument("--goal", type=int, choices=range(4), required=True)
        else:
            child.add_argument("--seed", type=int, default=0)
            child.add_argument("--samples", type=positive, default=1024)
            child.add_argument("--steps", type=positive, default=32)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.command == "train":
            model = train(
                ModelConfig(kind=args.kind),
                TrainConfig(steps=args.steps, seed=args.seed),
                args.output,
                resume=args.resume,
            )
            result = {
                "checkpoint": str(args.output / "checkpoint.pt"),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        elif args.command == "benchmark":
            result = benchmark(
                args.output,
                kinds=args.kinds,
                seeds=args.seeds,
                steps=args.steps,
                samples=args.samples,
                resume=args.resume,
            )["aggregate"]
        elif args.command == "run":
            report = integrated_run(
                args.output, steps=args.steps, seed=args.seed, samples=args.samples
            )
            result = {
                "report": str(args.output / "run.json"),
                "improvement": report["improvement"]["status"],
            }
        elif args.command == "world":
            result = run_world(args.output, seed=args.seed, steps=args.steps)
        elif args.command == "instrument":
            model, result = instrument_experiment(
                seed=args.seed, steps=args.steps, complex_valued=not args.real
            )
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "instrument.json", result)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dimension": 4,
                    "vocabulary": 4,
                    "complex": not args.real,
                },
                args.output / "instrument.pt",
            )
        elif args.command in {"evaluate", "adapt", "improve"}:
            model = load_solver(args.checkpoint)
            if args.command == "evaluate":
                result, _ = evaluate(
                    model, seed=args.seed, split="user-evaluation", samples=args.samples
                )
                write_json(args.output / "evaluation.json", result)
            elif args.command == "adapt":
                result = adaptation_experiment(model, seed=args.seed, samples=args.samples)
                write_json(args.output / "adaptation.json", result)
            else:
                result = improve(
                    model,
                    args.output,
                    seed=args.seed,
                    samples=args.samples,
                    max_queries=args.max_queries,
                    task=args.task,
                )
        elif args.command == "study":
            from sera.study import connected_study
            reports = connected_study(args.output, seeds=args.seeds, steps=args.steps,
                                       meta_train=args.meta_train, meta_validation=args.meta_validation,
                                       meta_test=args.meta_test, inner_steps=args.inner_steps,
                                       instrument_steps=args.instrument_steps)
            result = {"output": str(args.output), "runs": len(reports),
                      "versions": [report["current_version"] for report in reports]}
        elif args.command in {"solve", "learn"}:
            from dataclasses import asdict

            from sera.connected import autonomous_round, execute_goal
            from sera.environments import WorldSpec, make_world
            from sera.experience import EvidenceReplay
            from sera.solver import SolverStore
            store = SolverStore(args.directory)
            world = make_world(args.world_seed, family=args.family)
            if args.command == "solve":
                solver = store.load()
                result = {"version": solver.version,
                          **execute_goal(solver, world, args.start, args.goal)}
            else:
                replay = EvidenceReplay.load(args.directory / "experience.json")
                known = [WorldSpec(row["identifier"], tuple(tuple(r) for r in row["table"]),
                                   tuple(row["colors"]), row["resettable"])
                         for row in json.loads((args.directory / "worlds.json").read_text(encoding="utf-8"))]
                result, construction, evidence = autonomous_round(store, world, known, replay,
                                                                    seed=args.seed, samples=args.samples,
                                                                    steps=args.steps)
                for row in evidence.records:
                    replay.admit(row)
                replay.save(args.directory / "experience.json")
                known = list({spec.identifier: spec for spec in [*known, world]}.values())
                write_json(args.directory / "worlds.json", [asdict(spec) for spec in known])
                result["construction"] = construction
        elif args.command == "rollback":
            result = rollback(args.directory)
        else:
            journal = Journal(args.directory / "journal.sqlite")
            journal.verify()
            current_path = args.directory / "current.json"
            result = {
                "events": len(journal.events()),
                "journal_integrity": "verified",
                "current": json.loads(current_path.read_text(encoding="utf-8"))
                if current_path.exists()
                else None,
            }
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"SERA: {error}", file=sys.stderr)
        return 1
    return 0
