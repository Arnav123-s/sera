"""Frozen shared-R1 pretraining, support curves and paired adaptation controls."""

from __future__ import annotations

import copy
import gzip
from dataclasses import asdict
from pathlib import Path

import torch

from sera.accounting import Costs
from sera.binding import binding_cases
from sera.evaluation import AdmissionPolicy, assess
from sera.shared import IndependentTypedControl, SharedR1, make_shared_solver
from sera.shared_archive import load_shared_checkpoint, save_shared_checkpoint
from sera.shared_evaluation import evaluate_shared, score_record
from sera.shared_learning import adapt_shared, development, pretrain_shared
from sera.solver import Work, tensor_digest
from sera.storage import canonical, write_json
from sera.training import environment
from sera.typed_programs import induce


def shared_trial(root, *, seed, kind, pretrain_steps=1600, adapt_steps=192,
                 support_sizes=(32, 128, 512), samples=256, retained_samples=256, typed_samples=128):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    costs, work = Costs(), Work()
    record = {"schema_version": 1, "seed": seed, "kind": kind, "environment": environment(),
              "role": "confirmatory bounded shared-learner study after validation-only development",
              "budget": {"pretrain_steps": pretrain_steps, "adapt_steps": adapt_steps,
                         "batch_size": 32, "support_sizes": list(support_sizes), "samples": samples,
                         "retained_samples": retained_samples, "typed_samples": typed_samples},
              "runs": [], "status": "running"}
    try:
        with costs.phase("development-evidence", work):
            evidence, validation, spec, manifest = development(seed)
            evidence.save(root / "evidence")
            record["world"] = asdict(spec)
            record["typed_protocol"] = manifest
            (root / "validation.json.gz").write_bytes(gzip.compress(canonical({
                "typed": [asdict(r) for r in validation["typed"]],
                "latest": [asdict(r) for r in validation["latest"]],
                "world": [asdict(r) for r in validation["world"]], "truth": validation["truth"].tolist(),
                "sequence_seed": validation["sequence_seed"]}).encode(), mtime=0))
        with costs.phase("joint-pretraining", work):
            torch.manual_seed(seed)
            base = SharedR1(kind=kind)
            record["pretraining"] = pretrain_shared(base, evidence, validation,
                                                    seed=seed, steps=pretrain_steps, work=work)
        with costs.phase("bounded-typed-procedure-induction", work):
            record["procedures"] = {}
            for task in ("modular_sum", "byte_sum"):
                search = induce(evidence.typed.records, validation["typed"], task=task, work=work)
                record["procedures"][task] = search
                if search["accepted"]:
                    base.programs[task] = search["record"]
        record["base_checkpoint"] = save_shared_checkpoint(root / "base.pt", base)
        record["parameters"] = sum(p.numel() for p in base.parameters())
        record["core_state_bytes"] = base.core_state_bytes()
        record["base_tensor_sha256"] = tensor_digest(base)
        support = binding_cases(seed=980000+seed, count=max(support_sizes))
        checks = binding_cases(seed=990000+seed, count=128, split="validation")
        (root / "adaptation-evidence.json.gz").write_bytes(gzip.compress(canonical({
            "support": [asdict(r) for r in support], "validation": [asdict(r) for r in checks]}).encode(), mtime=0))
        candidates = []
        # All weights and validation selections are frozen before any query is scored.
        for count in support_sizes:
            for method in ("full", "replay", "adapter", "scratch"):
                label = f"{method}-{count}"
                with costs.phase(f"adaptation/{label}", work):
                    core, training = adapt_shared(base, evidence, support[:count], checks, method=method,
                                                  seed=seed, steps=adapt_steps, work=work)
                    checkpoint = save_shared_checkpoint(root / f"{label}.pt", core, parent=root / "base.pt")
                    candidates.append({"label": label, "method": method, "support_count": count,
                        "checkpoint": checkpoint, "training": training,
                        "parameters": sum(p.numel() for p in core.parameters()),
                        "tensor_sha256": tensor_digest(core)})
                write_json(root / "progress.json", {"frozen_candidates": len(candidates), "last": label})
                print(f"{kind} seed {seed}: froze {label}", flush=True)
        with costs.phase("paired-sealed-evaluation", work):
            args = dict(seed=1_000_000+seed, samples=samples, retained_samples=retained_samples,
                        typed_samples=typed_samples, work=work)
            baseline, before = evaluate_shared(make_shared_solver(base), [spec], **args)
            record["baseline"] = baseline
            raw = {"none": score_record(before)}
            for item in candidates:
                core = load_shared_checkpoint(root / f"{item['label']}.pt")
                if tensor_digest(core) != item["tensor_sha256"]:
                    raise ValueError("Frozen candidate reconstruction changed weights")
                solver = make_shared_solver(core)
                report, after = evaluate_shared(solver, [spec], **args)
                # This fixed-suite assessment is descriptive, never a production promotion.
                decision = assess(after, before, round_index=0, invariants_ok=solver.validate(), candidate_cost=0,
                                  policy=AdmissionPolicy(max_candidate_cost=10_000_000))
                raw[item["label"]] = score_record(after)
                record["runs"].append({**item, "evaluation": report, "retention_audit": decision})
                if item["method"] == "full":
                    separate = make_shared_solver(copy.deepcopy(base))
                    separate.components["typed"] = IndependentTypedControl(core)
                    result, values = evaluate_shared(separate, [spec], **args)
                    label = f"separate-{item['support_count']}"
                    raw[label] = score_record(values)
                    record["runs"].append({"label": label, "method": "separate", "support_count": item["support_count"],
                        "checkpoint_source": item["label"], "training": "Same adapted typed weights as full; frozen independent owner for world/sequence",
                        "additional_training_updates": 0, "parameters": sum(p.numel() for p in separate.parameters()),
                        "evaluation": result, "retention_audit": assess(values, before, round_index=0,
                            invariants_ok=separate.validate(), candidate_cost=0)})
                print(f"{kind} seed {seed}: scored {item['label']}", flush=True)
            (root / "scores.json.gz").write_bytes(gzip.compress(canonical(raw).encode(), mtime=0))
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(costs=costs.record(), work=work.record())
        write_json(root / "trial.json", record)
    return record
