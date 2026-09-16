"""Test sparse correction on a copy of the actual saved SERA language owner.

This is readout fault repair against retained pre-fault responses, not new
semantic learning. Hypothetical model output remains labeled as model output.
"""

import argparse
import copy
import gzip
import json
import time
import zipfile

import numpy as np
import torch

from experiments.grounded_language.data import examples
from experiments.grounded_language.model import tensors
from sera.session_state import model_identity
from workbench.model import Learner

from .arrays import ArrayWriter
from .core import l1, omp, relative
from .study import RELEASE, ROOT, sha, source_files, write


def capture(owner, rows):
    captured, outputs = [], []
    def remember(module, args):
        captured.append(args[0].detach().cpu().numpy().copy())
    hook = owner.language_readout[1].register_forward_pre_hook(remember)
    try:
        with torch.no_grad():
            for offset in range(0, len(rows), 32):
                ids, _ = tensors(rows[offset:offset+32])
                outputs.append([value.cpu().numpy() for value in owner.language(ids)])
    finally:
        hook.remove()
    return np.concatenate(captured), [np.concatenate([row[slot] for row in outputs]) for slot in range(4)]


def residual_weight_error(updated, pristine, fault):
    return float(np.linalg.norm(np.asarray(updated)-pristine)/max(np.linalg.norm(fault), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = RELEASE/args.name
    folder.mkdir(exist_ok=False)
    pointer_path = ROOT/"runs/sera-workbench/current.json"
    pointer_before = pointer_path.read_bytes()
    pointer = json.loads(pointer_before)
    revision_path = ROOT/"runs/sera-workbench/revisions"/pointer["revision"]
    assert sha(revision_path) == pointer["sha256"]
    record = json.loads(revision_path.read_text())
    sources = source_files()
    for directory in ("workbench", "experiments", "src/sera"):
        sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/directory).rglob("*.py")})
    protocol = {"id": args.name, "config": vars(args), "sources": sources,
                "parent_owner": record["owner_sha256"], "parent_revision_sha256": pointer["sha256"],
                "parent_checkpoint": record["language"]["checkpoint"], "counts": [32, 64, 96],
                "families": ["sparse4", "dense"], "methods": ["sparse_delta", "omp_delta", "ridge_delta", "damaged", "minimum_norm_relearn"],
                "scope": "One actual trained 256-wide language R1, fixed pooled recurrent features; repair one existing output neuron's 256 weights after simulated unknown corruption. 12 fault realizations do not constitute 12 independent trained owners.",
                "evidence": "96 pre-fault teaching anchors, 32 development anchors, 256 withheld prompts. Trusted pre-fault scalar responses are numerical checksums of the original readout, not independent factual observations. The target scalar is evaluated in float64 using cached float32 recurrent features/weights; actual float32 model execution is checked separately.",
                "faults": "Four unknown weight changes of magnitude .5..1, or a dense delta scaled to comparable total norm. True support and pristine weights are evaluator-only; learners receive damaged weights and retained anchor responses.",
                "controls": "Same anchors for sparse delta, OMP selected 1..8, ridge selected from 0/1e-4/.01/.1/1, unchanged damaged readout, and min-norm fitting from scratch. All 32 development anchors charged to all arms.",
                "primary": "Future correction NRMSE <=.05, coefficient error, exact semantic-slot accuracy and agreement with pristine model. Actual layer parameters are installed on an isolated owner copy; all other tensors and the live revision must remain unchanged.",
                "boundary": "Maintenance of a known finite language route. No new knowledge, discovered sparsity, shared-core learning, retained-ability improvement or eta claim. Even a success cannot approve a new live supported-answer policy."}
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT/name, name)
    started = time.perf_counter()
    learner = Learner(copy.deepcopy(record))
    owner = learner.session.owner
    owner.eval()
    assert model_identity(owner) == record["owner_sha256"]
    assert learner.solver.neural.owner is learner.solver.components["typed"].owner is owner
    initial_state = {name: value.detach().clone() for name, value in owner.state_dict().items()}
    # Same prompt partitions for all fault realizations within a cohort.
    bank = {"support": examples(args.start_seed+1, 96, "train"),
            "selection": examples(args.start_seed+2, 32, "development"),
            "evaluation": examples(args.start_seed+3, 256, "final")}
    features, logits = {}, {}
    for split, rows in bank.items():
        features[split], logits[split] = capture(owner, rows)
    arrays = ArrayWriter(folder/"arrays.zip")
    references = {split: arrays.put(value) for split, value in features.items()}
    logit_refs = {split: [arrays.put(value) for value in values] for split, values in logits.items()}
    write(folder/"evidence.json", {"prompts": bank, "features": references, "logits": logit_refs,
                                  "owner": record["owner_sha256"], "feature_provenance": "Actual recurrent language forward path, no feature fitting in this study"})
    raw, maximum_execution_discrepancy = [], 0.
    with gzip.open(folder/"records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            seed = args.start_seed+101*index
            rng = np.random.default_rng(seed)
            neuron = int(rng.integers(11))
            pristine = initial_state["language_readout.1.weight"][neuron].numpy().astype(float)
            bias = float(owner.language_readout[1].bias[neuron])
            for family in protocol["families"]:
                fault = np.zeros(256)
                if family == "sparse4":
                    fault[rng.choice(256, 4, replace=False)] = rng.choice([-1., 1.], 4)*rng.uniform(.5, 1., 4)
                else:
                    fault = rng.normal(scale=1.5/16, size=256)
                damaged = pristine+fault
                for count in protocol["counts"]:
                    a = features["support"][:count].astype(float)
                    retained = a@pristine+bias
                    target = retained-(a@damaged+bias)
                    val = features["selection"].astype(float)
                    validation_target = val@pristine-val@damaged
                    for method in protocol["methods"]:
                        begin = time.perf_counter()
                        certificates, candidates = [], []
                        if method == "sparse_delta":
                            fit, certificates = l1(a, target[:, None], 1e-7)
                            fitted = fit[:, 0]
                        elif method == "omp_delta":
                            for k in range(1, 9):
                                fitted = omp(a, target[:, None], k)[:, 0]
                                candidates.append((float(np.mean((val@fitted-validation_target)**2)), k, fitted))
                            fitted = min(candidates, key=lambda row: row[0])[2]
                        elif method == "ridge_delta":
                            for alpha in (0., .0001, .01, .1, 1.):
                                fitted = np.linalg.lstsq(a, target, rcond=None)[0] if alpha == 0 else a.T@np.linalg.solve(a@a.T+alpha*np.eye(count), target)
                                candidates.append((float(np.mean((val@fitted-validation_target)**2)), alpha, fitted))
                            fitted = min(candidates, key=lambda row: row[0])[2]
                        elif method == "minimum_norm_relearn":
                            fitted = np.linalg.lstsq(a, retained-bias, rcond=None)[0]-damaged
                        else:
                            fitted = np.zeros(256)
                        fit_seconds = time.perf_counter()-begin
                        updated = (damaged+fitted).astype(np.float32)
                        with torch.no_grad():
                            owner.language_readout[1].weight[neuron].copy_(torch.from_numpy(updated))
                            actual_logits = owner.language_readout[1](torch.from_numpy(features["evaluation"])).numpy()
                        # One full recurrent execution per fault family validates cached-head execution.
                        if count == 32 and method == "sparse_delta":
                            _, executed = capture(owner, bank["evaluation"][:32])
                            discrepancy = float(np.max(abs(executed[1]-actual_logits[:32])))
                            maximum_execution_discrepancy = max(maximum_execution_discrepancy, discrepancy)
                            assert discrepancy <= 1e-5
                        q = features["evaluation"].astype(float)
                        predicted_correction = q@(updated.astype(float)-damaged)
                        actual_correction = -q@fault
                        base_labels = np.column_stack([value.argmax(axis=1) for value in logits["evaluation"]])
                        labels = base_labels.copy()
                        labels[:, 1] = actual_logits.argmax(axis=1)
                        expected_labels = np.array([r["labels"] for r in bank["evaluation"]])
                        row = {"seed": seed, "family": family, "count": count, "method": method, "neuron": neuron,
                               "pristine": pristine.tolist(), "fault": fault.tolist(), "damaged": damaged.tolist(),
                               "fitted_delta": fitted.tolist(), "updated": updated.tolist(), "bias": bias,
                               "retained_responses": retained.tolist(), "target": target.tolist(),
                               "certificates": certificates, "candidates": [[v, p] for v, p, _ in candidates],
                               "prediction_error": relative(predicted_correction, actual_correction),
                               "coefficient_error": residual_weight_error(updated.astype(float), pristine, fault),
                               "parent_exact_slot_accuracy": float(np.mean((base_labels == expected_labels).all(axis=1))),
                               "exact_slot_accuracy": float(np.mean((labels == expected_labels).all(axis=1))),
                               "parent_agreement": float(np.mean((labels == base_labels).all(axis=1))),
                               "evaluation_logits": arrays.put(actual_logits), "paid_anchor_scalars": count+32,
                               "decoder_seconds": fit_seconds}
                        stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                        raw.append({k: row[k] for k in ("seed", "family", "count", "method", "prediction_error", "coefficient_error", "parent_exact_slot_accuracy", "exact_slot_accuracy", "parent_agreement", "decoder_seconds")})
                        with torch.no_grad():
                            owner.language_readout[1].weight[neuron].copy_(torch.from_numpy(pristine.astype(np.float32)))
                stream.flush()
            write(folder/"progress.json", {"records": len(raw), "last_fault_seed": seed})
    arrays.close()
    assert all(torch.equal(value, initial_state[name]) for name, value in owner.state_dict().items())
    assert model_identity(owner) == record["owner_sha256"] and pointer_path.read_bytes() == pointer_before
    result = {"status": "COMPLETE", "records": len(raw), "rows": raw, "parent_owner": record["owner_sha256"],
              "live_unchanged": True, "actual_shared_owner_aliases": True,
              "maximum_full_execution_discrepancy": maximum_execution_discrepancy,
              "worker_seconds": time.perf_counter()-started}
    write(folder/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
