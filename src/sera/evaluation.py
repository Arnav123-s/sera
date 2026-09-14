"""Paired scoring and conservative admission of a candidate on fresh data."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from sera.data import TASKS, make_batch
from sera.storage import digest


@torch.no_grad()
def evaluate(model, *, seed, split, length=12, samples=512, tasks=range(4), program=None):
    model.eval()
    device = str(next(model.parameters()).device)
    results, all_scores, ids = {}, {}, []
    for task in tasks:
        batch = make_batch(samples, length, seed, split=split, index=task, task=task)
        if hasattr(model, "predict_probabilities"):
            probabilities = torch.cat([model.predict_probabilities(chunk.to(device)).cpu()
                                       for chunk in batch.inputs.split(128)])
        else:
            logits = torch.cat([model(chunk.to(device)).cpu() for chunk in batch.inputs.split(128)])
            probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("Invalid predictive probabilities")
        if program is not None and task == 2:
            predicted = torch.tensor(
                [
                    program.execute(
                        actions, initial_state=int(initial), environment_id="ordered-control-v1"
                    )
                    for actions, initial in zip(batch.actions, batch.initial_states)
                ]
            )
            # A symbolic procedure returns an exact categorical answer in its domain.
            probabilities = F.one_hot(predicted, 4).float()
        else:
            predicted = probabilities.argmax(-1)
        scores = (predicted == batch.targets).float().numpy()
        truth = F.one_hot(batch.targets, 4)
        target_probs = probabilities[torch.arange(samples), batch.targets]
        # Finite floor is reported explicitly; accuracy is the promotion score.
        nll = float(-target_probs.clamp_min(1e-12).log().mean())
        results[TASKS[task]] = {
            "accuracy": float(scores.mean()),
            "nll_floor_1e_12": nll,
            "brier": float((probabilities - truth).square().sum(-1).mean()),
            "majority_baseline": float(torch.bincount(batch.targets, minlength=4).max() / samples),
            "samples": samples,
        }
        all_scores[TASKS[task]] = scores
        ids.append(batch.dataset_id)
    if not results:
        raise ValueError("Evaluation requires at least one task")
    return {
        "tasks": results,
        "macro_accuracy": float(np.mean([v["accuracy"] for v in results.values()])),
        "dataset_id": digest(ids),
        "length": length,
    }, all_scores


@dataclass(frozen=True)
class AdmissionPolicy:
    global_alpha: float = 0.05
    minimum_gain: float = 0.01
    max_retention_loss: float = 0.02
    max_candidate_cost: float = 1_000_000

    def __post_init__(self):
        if not 0 < self.global_alpha < 1:
            raise ValueError("Invalid error budget")
        if not 0 <= self.minimum_gain <= 1 or not 0 <= self.max_retention_loss <= 1:
            raise ValueError("Invalid gain or retention tolerance")
        if not math.isfinite(self.max_candidate_cost) or self.max_candidate_cost <= 0:
            raise ValueError("Invalid candidate cost cap")


class CapabilityScores(dict):
    """Objective samples plus independent empirical retention checks.

    Only objective samples enter the gain bound. Correlated capability checks do
    not inflate its sample count. Dataset identities cover ordered paired cases.
    """

    def __init__(self, objective, capabilities, dataset_ids):
        super().__init__(objective)
        self.capabilities = capabilities
        self.dataset_ids = dataset_ids
        if not capabilities or set(capabilities) != set(dataset_ids):
            raise ValueError("Every required capability needs scores and a dataset identity")


def assess(
    candidate,
    incumbent,
    *,
    round_index,
    invariants_ok,
    candidate_cost,
    policy: AdmissionPolicy = AdmissionPolicy(),
):
    if type(round_index) is not int or round_index < 0:
        raise ValueError("Round index must be a nonnegative integer")
    if (
        not candidate
        or set(candidate) != set(incumbent)
        or not math.isfinite(candidate_cost)
        or candidate_cost < 0
    ):
        raise ValueError("Invalid paired task scores or cost")
    differences, retention = [], {}
    for task in sorted(candidate):
        a, b = np.asarray(candidate[task], float), np.asarray(incumbent[task], float)
        if a.ndim != 1 or len(a) == 0 or a.shape != b.shape:
            raise ValueError("Paired task samples must be nonempty and have identical shapes")
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.any(
            (a < 0) | (a > 1) | (b < 0) | (b > 1)
        ):
            raise ValueError("Scores must be finite in [0,1]")
        differences.append(a - b)
        retention[task] = float((b - a).mean())
    # Equal per-task counts keep this pooled score equal to the macro score.
    if len({len(d) for d in differences}) != 1:
        raise ValueError("Admission requires equal sample counts per task")
    differences = np.concatenate(differences)
    log_alpha = math.log(policy.global_alpha) - (round_index + 1) * math.log(2)
    mean_gain = float(differences.mean())
    lower_bound = mean_gain - math.sqrt(-2 * log_alpha / len(differences))
    reasons = []
    if lower_bound <= policy.minimum_gain:
        reasons.append("insufficient_independent_gain")
    if any(loss > policy.max_retention_loss for loss in retention.values()):
        reasons.append("retention_regression")
    if not invariants_ok:
        reasons.append("invalid_candidate")
    if candidate_cost > policy.max_candidate_cost:
        reasons.append("cost_budget_exceeded")
    result = {
        "admitted": not reasons,
        "reasons": reasons,
        "mean_gain": mean_gain,
        "gain_lower_bound": lower_bound,
        "log_alpha": log_alpha,
        "round_index": round_index,
        "samples": len(differences),
        "retention_loss_by_task": retention,
        "candidate_cost": candidate_cost,
        "policy": asdict(policy),
        "assumptions": "Fixed bounded scores; independent fresh examples conditional on prior search. Retention gates are empirical, not confidence bounds.",
    }
    if isinstance(candidate, CapabilityScores) or isinstance(incumbent, CapabilityScores):
        if not isinstance(candidate, CapabilityScores) or not isinstance(incumbent, CapabilityScores):
            raise ValueError("Both paired evaluations must use the same capability contract")
        if (not incumbent.capabilities or set(candidate.capabilities) != set(incumbent.capabilities)
                or set(candidate.capabilities) != set(candidate.dataset_ids)
                or set(incumbent.capabilities) != set(incumbent.dataset_ids)
                or candidate.dataset_ids != incumbent.dataset_ids):
            raise ValueError("Required capabilities or paired capability datasets differ")
        losses = {}
        for name in sorted(incumbent.capabilities):
            a = np.asarray(candidate.capabilities[name], float)
            b = np.asarray(incumbent.capabilities[name], float)
            if a.ndim != 1 or not len(a) or a.shape != b.shape:
                raise ValueError("Capability samples must be nonempty paired vectors")
            if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
                raise ValueError("Capability scores must be finite in [0,1]")
            losses[name] = float((b - a).mean())
        failed = [name for name, loss in losses.items() if loss > policy.max_retention_loss]
        if failed:
            reasons.append("critical_capability_regression")
        result.update(admitted=not reasons, capability_contract="separate-retention-v2",
                      retention_loss_by_capability=losses, failed_capabilities=failed,
                      capability_dataset_ids=incumbent.dataset_ids,
                      capability_samples={name: len(v) for name, v in incumbent.capabilities.items()})
    return result
