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
from sera.storage import Journal, write_json
from sera.training import TrainConfig, adaptation_experiment, load_model, train


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
    child = sub.add_parser("status")
    child.add_argument("directory", type=Path)
    child = sub.add_parser("rollback")
    child.add_argument("directory", type=Path)
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
            model, _ = load_model(args.checkpoint)
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
                )
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
