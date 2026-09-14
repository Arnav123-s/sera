"""Sealed binding objectives and individually measured retained capabilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from sera.binding import binding_cases
from sera.capability_evaluation import evaluate_capabilities, required_capabilities
from sera.evaluation import CapabilityScores
from sera.storage import digest
from sera.typed_learning import score_typed
from sera.typed_protocol import semantic_id, typed_suite

FAMILIES = ("ordinary", "long", "composition")


@torch.no_grad()
def binding_scores(model, rows, work=None):
    model.eval()
    correct, losses, briers = [], [], []
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        probabilities = model([r.observations for r in batch], [r.task for r in batch])["categorical"].softmax(-1)
        targets = torch.tensor([r.target for r in batch])
        correct.extend((probabilities.argmax(-1) == targets).float().tolist())
        losses.extend((-probabilities[torch.arange(len(batch)), targets].clamp_min(1e-12).log()).tolist())
        briers.extend((probabilities - F.one_hot(targets, 4)).square().sum(-1).tolist())
        if work is not None:
            work.add("binding_evaluation_events", sum(len(r.observations) for r in batch))
    ids = [semantic_id(r) for r in rows]
    return {"accuracy": float(np.mean(correct)), "nll": float(np.mean(losses)), "brier": float(np.mean(briers)),
            "examples": len(rows), "unique_cases": len(set(ids)), "dataset_id": digest(ids)}, {
                "accuracy": np.asarray(correct), "calibration": 1 - np.asarray(briers)/2}


def shared_capabilities(solver, specs, *, world_learning=False):
    return required_capabilities(solver, specs) | {
        f"instructed-{rule}/{family}/{metric}" for rule in (("earliest", "latest") if world_learning else ("latest",))
        for family in FAMILIES for metric in ("accuracy", "calibration")}


def evaluate_shared(solver, specs, *, seed, samples=512, retained_samples=256, typed_samples=128, work=None,
                    world_learning=False):
    report, old = evaluate_capabilities(solver, specs, seed=seed, samples=retained_samples,
                                        typed_samples=typed_samples, work=work)
    suite, _ = typed_suite(seed=seed, support_count=0, validation_count=0, test_count=typed_samples)
    report["typed_neural_only"] = {partition: score_typed(solver.components["typed"], rows, work=work)
                                     for partition, rows in suite.items() if rows}
    objective, ids, binding = {}, {}, {}
    for rule in ("earliest", "latest"):
        binding[rule] = {}
        for family in FAMILIES:
            # Independent draws with replacement support the objective's gain bound.
            rows = binding_cases(seed=seed, count=samples, rule=rule, family=family,
                                  split="promotion-shared", unique=False)
            result, values = binding_scores(solver.components["typed"], rows, work)
            binding[rule][family] = result
            if rule == "earliest":
                objective[family] = values["accuracy"]
                ids[family] = result["dataset_id"]
            if rule == "latest" or world_learning:
                for metric, vector in values.items():
                    name = f"instructed-{rule}/{family}/{metric}"
                    old.capabilities[name] = vector
                    old.dataset_ids[name] = result["dataset_id"]
    scores = CapabilityScores(dict(old) if world_learning else objective, old.capabilities, old.dataset_ids)
    if set(scores.capabilities) != shared_capabilities(solver, specs, world_learning=world_learning):
        raise ValueError("Shared retention coverage is incomplete")
    report.update(binding=binding, dataset_id=digest([report["dataset_id"], ids, scores.dataset_ids]),
                  objective=("World prediction/control mean" if world_learning else
                             "Mean earliest-binding accuracy across ordinary, long and extra-overwrite families"),
                  objective_dataset_ids=ids, admission_contract="shared-binding-v1",
                  binding_sampling="IID draws with replacement, restricted to the sealed semantic bucket",
                  added_retention=f"{'Both rules' if world_learning else 'Latest binding'} accuracy and bounded Brier quality (1 - Brier/2), separately per family")
    return report, scores


def score_record(scores):
    return {"objectives": {k: v.tolist() for k, v in scores.items()},
            "capabilities": {k: v.tolist() for k, v in scores.capabilities.items()},
            "dataset_ids": scores.dataset_ids}
