"""Reproducible from-scratch training, resume, and controlled adaptation."""

from __future__ import annotations

import copy
import hashlib
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sera.data import make_batch, seed_for
from sera.evaluation import evaluate
from sera.models import ModelConfig, StatefulModel
from sera.storage import write_json


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 500
    batch_size: int = 64
    length: int = 12
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    validate_every: int = 100
    validation_samples: int = 128
    seed: int = 0
    threads: int = 1

    def __post_init__(self):
        if (
            min(
                self.steps,
                self.batch_size,
                self.validate_every,
                self.validation_samples,
                self.threads,
            )
            < 1
        ):
            raise ValueError("Training dimensions must be positive")
        if self.length < 3 or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid sequence length or learning rate")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def source_hash():
    root = Path(__file__).parent
    hasher = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return hasher.hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "source_sha256": source_hash(),
    }


def atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_model(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model = StatefulModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["best_state"])
    model.eval()
    return model, payload


def train(model_config: ModelConfig, config: TrainConfig, output: Path, *, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Run exists: {output}. Use --resume or a fresh output directory.")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed_for("model-init", config.seed))
    model = StatefulModel(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, start_step, best, best_state = [], 0, -1.0, None
    elapsed_before = 0.0
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        previous = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if previous.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        old_config = dict(previous["train_config"])
        old_config["steps"] = config.steps
        if old_config != asdict(config) or previous["model_config"] != asdict(model_config):
            raise ValueError(
                "Resume may extend steps but must preserve the experiment configuration"
            )
        if previous["environment"]["source_sha256"] != source_hash():
            raise ValueError("Source changed since checkpoint; begin a new experiment")
        model.load_state_dict(previous["current_state"])
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["rng_state"])
        history, start_step = previous["history"], previous["step"]
        best, best_state = previous["best_score"], previous["best_state"]
        elapsed_before = previous["train_seconds"]
        if config.steps < start_step:
            raise ValueError("Resume cannot reduce the number of training steps")
    started = time.perf_counter()
    for step in range(start_step + 1, config.steps + 1):
        batch = make_batch(config.batch_size, config.length, config.seed, split="train", index=step)
        model.train()
        loss = F.cross_entropy(model(batch.inputs), batch.targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % config.validate_every == 0 or step == config.steps:
            validation, _ = evaluate(
                model,
                seed=config.seed,
                split="validation",
                length=config.length,
                samples=config.validation_samples,
            )
            score = validation["macro_accuracy"]
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                    "validation_macro": score,
                }
            )
            if score > best:
                best, best_state = score, copy.deepcopy(model.state_dict())
            payload = {
                "schema_version": 1,
                "model_config": asdict(model_config),
                "train_config": asdict(config),
                "step": step,
                "current_state": model.state_dict(),
                "best_state": best_state,
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "history": history,
                "best_score": best,
                "train_seconds": elapsed_before + time.perf_counter() - started,
                "environment": environment(),
            }
            atomic_checkpoint(checkpoint, payload)
            write_json(
                output / "training.json",
                {
                    k: v
                    for k, v in payload.items()
                    if k not in {"current_state", "best_state", "optimizer", "rng_state"}
                },
            )
            print(
                f"{model_config.kind} seed={config.seed} step={step}/{config.steps} validation={score:.3f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model


def adaptation_experiment(model, *, seed, steps=64, support_examples=128, samples=256):
    if min(steps, support_examples, samples) < 1:
        raise ValueError("Adaptation budgets must be positive")
    support = make_batch(support_examples, 12, seed, split="adapt-support", task=4)
    conditions = {
        "no_update": copy.deepcopy(model),
        "full_update": copy.deepcopy(model),
        "replay": copy.deepcopy(model),
    }
    torch.manual_seed(seed_for("model-init", seed))
    conditions["scratch"] = StatefulModel(model.config)
    reports = {}
    for name, candidate in conditions.items():
        if name != "no_update":
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.002, weight_decay=0.0001)
            generator = torch.Generator().manual_seed(seed_for("adapt-order", seed))
            candidate.train()
            for step in range(steps):
                novel_n = 32 if name == "replay" else 64
                ix = torch.randint(support_examples, (novel_n,), generator=generator)
                inputs, targets = support.inputs[ix], support.targets[ix]
                if name == "replay":
                    replay = make_batch(32, 12, seed, split="adapt-replay", index=step)
                    inputs = torch.cat((inputs, replay.inputs))
                    targets = torch.cat((targets, replay.targets))
                loss = F.cross_entropy(candidate(inputs), targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        novel, _ = evaluate(candidate, seed=seed, split="adapt-query", samples=samples, tasks=[4])
        old, _ = evaluate(candidate, seed=seed, split="adapt-retention", samples=samples)
        reports[name] = {
            "novel": novel,
            "retention": old,
            "optimizer_steps": 0 if name == "no_update" else steps,
            "novel_draws": 0 if name == "no_update" else steps * (32 if name == "replay" else 64),
        }
    return {
        "support_dataset_id": support.dataset_id,
        "support_examples": support_examples,
        "conditions": reports,
        "scope": "Related earliest-binding task; equal update batches; replay uses half as many novel draws",
    }
