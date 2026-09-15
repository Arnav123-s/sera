"""Read-only causal preflight with the preserved before/after correction owners."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.cross_route_transfer.data import circle_cases, state_digest, teacher_identity
from experiments.cross_route_transfer.study import parent
from sera.connected import restore_component

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT/"runs/SHARED-GG-001/circle/pre-correction.pt"
AFTER = ROOT/"runs/SHARED-GG-001/circle/corrected/versions/v0.pt"


class AcquisitionNotIdentified(ValueError):
    pass


def assert_acquired_effect(before, after, *, tolerance=1e-10):
    before, after = np.asarray(before), np.asarray(after)
    if before.size == 0 or before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Matched finite intervention outcomes required")
    effect = float(np.max(np.abs(before-after)))
    if effect <= tolerance:
        raise AcquisitionNotIdentified("The teaching representation erases the acquired-state intervention")
    return effect


def load_before():
    payload = torch.load(BEFORE, weights_only=True, map_location="cpu")
    item = payload["components"]["r1"]
    model = restore_component(item["config"])
    model.load_state_dict(item["state"], strict=True)
    model.validity()
    return model.eval(), hashlib.sha256(BEFORE.read_bytes()).hexdigest()


def truth(phases):
    # Original integration fixture's published generating parameters, assessor only.
    angle = np.pi*np.asarray(phases)
    return np.stack((.25+.8*np.cos(angle)-.3*np.sin(angle),
                     -.4+.8*np.sin(angle)+.3*np.cos(angle)), -1)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before, before_file_sha = load_before()
    after = parent().components["r1"]
    before_state, after_state = state_digest(before), state_digest(after)
    if teacher_identity(before) == teacher_identity(after):
        raise ValueError("No actual knowledge intervention")
    if any(not torch.equal(p, dict(after.named_parameters())[n]) for n, p in before.named_parameters()):
        raise ValueError("Numerical learner changed along with K; intervention is confounded")
    normalized_maximum, comparisons = 0.0, []
    for seed in (17001, 17009, 17021, 17027, 17033, 17041, 17047, 17053):
        left = circle_cases(before, seed=seed, count=128, split="support-intervention", conditional=True)
        right = circle_cases(after, seed=seed, count=128, split="support-intervention", conditional=True)
        x = np.asarray([[*sum((list(o.values) for o in r.observations), []), *r.target] for r in left])
        y = np.asarray([[*sum((list(o.values) for o in r.observations), []), *r.target] for r in right])
        difference = float(np.max(np.abs(x-y)))
        normalized_maximum = max(normalized_maximum, difference)
        rejected = False
        try:
            assert_acquired_effect(x, y)
        except AcquisitionNotIdentified:
            rejected = True
        if not rejected:
            raise ValueError("Algebraic normalization deduction was not reproduced")
        comparisons.append({"seed": seed, "worlds": len(left), "maximum_coordinate_difference": difference,
                            "causal_acquisition_claim_rejected": rejected})
    phases = np.random.default_rng(17060916).uniform(-1.5, 1.5, 257)
    with torch.no_grad():
        raw_before = before.forward_generator(phases.tolist())["class_means"][0].numpy()
        raw_after = after.forward_generator(phases.tolist())["class_means"][0].numpy()
    observed_truth = truth(phases)
    effect = assert_acquired_effect(raw_before, raw_after)
    if state_digest(before) != before_state or state_digest(after) != after_state:
        raise ValueError("Read-only intervention audit mutated a checkpoint")
    result = {"status": "PASS", "before_teacher": teacher_identity(before), "after_teacher": teacher_identity(after),
              "before_checkpoint_sha256": before_file_sha,
              "after_checkpoint_sha256": hashlib.sha256(AFTER.read_bytes()).hexdigest(),
              "neural_parameters_bitwise_unchanged": True,
              "source_knowledge_intervention": "Preserved original noisy circle before versus after the independent sensor correction",
              "normalized_worlds": 1024, "normalized_maximum_difference": normalized_maximum,
              "normalized_teaching_identifies_acquired_coefficients": False,
              "raw_coordinate_queries": len(phases), "raw_maximum_intervention_effect": effect,
              "before_raw_mse": float(np.square(raw_before-observed_truth).mean()),
              "after_raw_mse": float(np.square(raw_after-observed_truth).mean()),
              "cohort": comparisons, "seconds": time.perf_counter()-started,
              "deduction": "Normalization cancels all fitted coefficients; raw-coordinate prediction preserves their effect.",
              "next_contract": "Before training, require an acquired/unacquired or corrected/uncorrected K intervention to affect an otherwise matched teaching/task channel. Nonzero effect is necessary, not sufficient, for useful causal transfer.",
              "scope": "This audit preserves useful original knowledge and rejects only the stronger attribution of the normalization-based teaching experiment."}
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    evidence = {"phases": phases.tolist(), "before": raw_before.tolist(), "after": raw_after.tolist(), "truth": observed_truth.tolist()}
    (output/"raw-coordinate-evidence.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
