"""Small budgeted research runs with raw artifacts and explicit negative results."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from sera.data import seed_for
from sera.engine import improve, run_world
from sera.evaluation import evaluate
from sera.models import ModelConfig
from sera.quantum import EventInstrument, density_residuals
from sera.storage import write_json
from sera.training import TrainConfig, adaptation_experiment, environment, train


def instrument_experiment(*, seed=0, steps=300, complex_valued=True):
    if steps < 1:
        raise ValueError("Training steps must be positive")
    torch.manual_seed(seed_for("instrument-init", seed))
    model = EventInstrument(complex_valued=complex_valued)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    generator = torch.Generator().manual_seed(seed_for("instrument-train", seed))
    evaluation = (torch.arange(4)[:, None] + torch.arange(24)[None]) % 4

    def nll(length):
        # The first symbol has unpredictable uniformly random phase.
        return -model(evaluation[:, :length])[:, 1:].log().mean()

    with torch.no_grad():
        initial = float(nll(8))
    for _ in range(steps):
        starts = torch.randint(4, (32, 1), generator=generator)
        events = (starts + torch.arange(8)[None]) % 4
        loss = -model(events)[:, 1:].log().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Invalid event likelihood")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        k = model.kraus()
        residual = float(((k.mH @ k).sum(0) - torch.eye(model.dimension)).abs().max())
        state, _ = model.observe(model.initial_state(4), torch.arange(4))
        generated = model.generate(64, seed=seed_for("instrument-generation", seed))
        result = {
            "seed": seed,
            "steps": steps,
            "complex": complex_valued,
            "real_parameter_count": sum(
                p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
            ),
            "initial_nll": initial,
            "id_nll": float(nll(8)),
            "length24_nll": float(nll(24)),
            "completeness_residual": residual,
            "posterior": density_residuals(state),
            "generated": generated,
            "cycle_consistency": sum(b == (a + 1) % 4 for a, b in zip(generated, generated[1:]))
            / 63,
            "exact_bigram_control_nll": 0.0,
            "scope": "Only four cyclic phases; real/complex raw dimensions differ; a valid learned instrument need not beat a bigram rule",
        }
    return model, result


def benchmark(output: Path, *, kinds, seeds, steps=500, samples=512, resume=False):
    if not kinds or not seeds or len(set(kinds)) != len(kinds) or len(set(seeds)) != len(seeds):
        raise ValueError("Specify unique nonempty model kinds and seeds")
    output.mkdir(parents=True, exist_ok=True)
    runs, failures = [], []
    started = time.perf_counter()
    manifest = {
        "kinds": list(kinds),
        "seeds": list(seeds),
        "steps": steps,
        "samples_per_task": samples,
        "environment": environment(),
        "budget": "same optimizer steps and batch size; parameter and state sizes differ",
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or existing != manifest:
            raise ValueError("Use a fresh benchmark directory or resume an identical experiment")
    write_json(manifest_path, manifest)
    for kind in kinds:
        for seed in seeds:
            run_dir = output / f"{kind}-{seed}"
            result_path = run_dir / "result.json"
            try:
                if resume and result_path.exists():
                    runs.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue
                model = train(
                    ModelConfig(kind=kind),
                    TrainConfig(steps=steps, seed=seed),
                    run_dir,
                    resume=resume and (run_dir / "checkpoint.pt").exists(),
                )
                id_result, _ = evaluate(model, seed=seed, split="test-id", samples=samples)
                ood_result, _ = evaluate(
                    model, seed=seed, split="test-ood", length=24, samples=samples
                )
                result = {
                    "kind": kind,
                    "seed": seed,
                    "id": id_result,
                    "ood": ood_result,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "core_state_bytes": model.state_bytes(),
                    "training": json.loads((run_dir / "training.json").read_text(encoding="utf-8")),
                }
                write_json(result_path, result)
                runs.append(result)
                print(
                    f"RESULT {kind}/{seed}: ID {id_result['macro_accuracy']:.3f}, OOD {ood_result['macro_accuracy']:.3f}",
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {"kind": kind, "seed": seed, "type": type(error).__name__, "error": str(error)}
                )
                write_json(output / "failures.json", failures)
                raise
    aggregate = []
    for kind in kinds:
        group = [r for r in runs if r["kind"] == kind]
        item = {
            "kind": kind,
            "seeds": len(group),
            "parameters": group[0]["parameters"],
            "core_state_bytes": group[0]["core_state_bytes"],
        }
        for split in ("id", "ood"):
            values = [r[split]["macro_accuracy"] for r in group]
            item[f"{split}_mean"] = float(np.mean(values))
            item[f"{split}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate.append(item)
    summary = {
        "manifest": manifest,
        "aggregate": aggregate,
        "runs": runs,
        "failures": failures,
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def integrated_run(output: Path, *, steps=500, seed=0, samples=1024):
    if (output / "run.json").exists():
        raise FileExistsError("Integrated run already exists")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = train(ModelConfig(), TrainConfig(steps=steps, seed=seed), output / "neural")
    main, _ = evaluate(model, seed=seed, split="integrated-test", samples=samples)
    write_json(output / "neural_evaluation.json", main)
    world = run_world(output / "world", seed=seed)
    print(f"World transition accuracy: {world['transition_accuracy']:.3f}", flush=True)
    adaptation = adaptation_experiment(model, seed=seed)
    write_json(output / "adaptation.json", adaptation)
    improvement = improve(model, output / "improvement", seed=seed, samples=samples)
    report = {
        "environment": environment(),
        "seed": seed,
        "steps": steps,
        "neural": main,
        "world": world,
        "adaptation": adaptation,
        "improvement": improvement,
    }
    write_json(output / "run.json", report)
    return report
