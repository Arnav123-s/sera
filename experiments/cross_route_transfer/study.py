"""Prospectively frozen, resumable generator-to-motion experiments."""

import argparse
import copy
import gzip
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from experiments.guarded_consolidation.study import PARENT, ReadOnlyStore
from sera.environments import WorldSpec
from sera.shared import IndependentTypedControl, replace_shared_owner
from sera.shared_archive import load_shared_checkpoint, save_shared_checkpoint
from sera.shared_evaluation import evaluate_shared, score_record
from sera.shared_learning import SharedEvidence
from sera.storage import canonical, digest
from sera.world_graph import SharedOwnerRef

from .data import circle_cases, partition_audit, state_digest, teacher_identity
from .learning import ARMS, fit, motion_score

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/17_transfer"
CHECKPOINT = PARENT/"versions/v0.pt"
PARENT_SHA = "eaf0b8fc315c943f4408f921f16ecfe8582bf2f675977c166f53e8242cbc4f37"
EVIDENCE = ROOT/"runs/sera-0.5-current/evidence/e8ac1aebb3fc907ee"
SHARED_BASE = ROOT/"runs/A06-transfer/parent.pt"
DEFAULT = {"steps": 192, "batch_size": 32, "replay_batch": 32, "learning_rate": .0005,
           "replay_weight": 1.0, "scope": "shared", "select_every": 48,
           "observed": 32, "dreams": 512, "development": 128,
           "final": 256, "extent": 128, "retention_samples": 64, "typed_samples": 64}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    paths = list((ROOT/"src/sera").glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    paths += [ROOT/"experiments/guarded_consolidation"/name for name in
              ("__init__.py", "core.py", "indexed_proof.py", "migration.py", "study.py")]
    paths += [ROOT/n for n in ("scripts/run_transfer_bounded.py", "scripts/run_v3_bounded.py",
                              "scripts/windows_job_v3.py", "scripts/run_transfer_batch.ps1",
                              "tests/test_cross_route_transfer.py")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def parent():
    if sha(CHECKPOINT) != PARENT_SHA:
        raise ValueError("Trained predecessor checkpoint changed")
    return ReadOnlyStore(PARENT).load(version="v0")


def world_specs():
    return [WorldSpec(r["identifier"], tuple(map(tuple, r["table"])), tuple(r["colors"]), r["resettable"])
            for r in read(ROOT/"runs/sera-0.5-current/worlds.json")]


def freeze(name, cfg, seeds):
    output = RELEASE/name
    output.mkdir(parents=True, exist_ok=False)
    payload = {"experiment": name, "config": cfg, "seeds": seeds, "arms": list(ARMS),
               "parent_sha256": PARENT_SHA, "sources": source_files(),
               "evidence": {p.relative_to(ROOT).as_posix(): sha(p) for p in EVIDENCE.iterdir()},
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
               "primary": "Neural next-position MSE on new circle worlds, one query/world; target inference has no generator access.",
               "gate": {"all_streams_improve_over_observed": True, "relative_mse_gain_at_least": .10,
                        "paired_stream_bootstrap_lower_gain_above": 0,
                        "maximum_old_capability_mean_drop": .02,
                        "detached_target_maximum_absolute_mse_difference": 1e-7},
               "controls": {"observed_only": "Same numerical owner/steps; resample 32 observed worlds.",
                            "detached": "Same conditional labels/updates on an independent owner; original owner frozen.",
                            "integrated": "Conditional labels update the actual descendant shared owner; no extra parameters.",
                            "extra_capacity": "Independent extra owner trained only on observed worlds; parent preserved.",
                            "oracle_exposure": "Same new worlds, independent simulator outcomes, extra 512 observed labels charged."},
               "scope": "Supplied circle family/normalization, conditional synthetic distillation, one pretrained parent. No eta learning or broad applicability certificate.",
               "retention": "Fresh existing 40-group bank plus 21 neural-only typed groups; no new final data used for checkpoint selection.",
               "counts": "Paired optimizer starts/updates, target/replay/anchor draws and shapes. Parent anchor outputs are conditional targets, never ground truth. Wall times/bytes separate; no fictitious matched FLOP claim.",
               "uncertainty": "20,000 paired learning-stream bootstrap replicates, seed 17060915, 1.25/98.75 percentiles. Descriptive interval with two candidate scopes frozen together; not a distribution-free guarantee.",
               "rejected_work": "All pilots, failed jobs and full supervision/checkpoint/evaluation costs retained.",
               "promotion": "Only a narrow conditional component can qualify; no operational replacement in this study."}
    write(output/"protocol.json", {"payload": payload, "sha256": digest(payload)})
    with zipfile.ZipFile(output/"frozen-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in payload["sources"]:
            archive.write(ROOT/name, name)
    print(json.dumps({"frozen": str(output), "sha256": digest(payload)}))


def check_contract(directory):
    record = read(directory/"protocol.json")
    cfg = record["payload"]
    if digest(cfg) != record["sha256"] or cfg["sources"] != source_files():
        raise ValueError("Frozen source/protocol identity mismatch")
    for path, identity in cfg["evidence"].items():
        if sha(ROOT/path) != identity:
            raise ValueError("Replay evidence changed")
    actual = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    if actual != cfg["runtime"]:
        raise ValueError("Numerical runtime changed")
    return cfg


def data_bank(owner, cfg, seed, phase):
    # Optimizer stream, teaching, selection and assessment have distinct seed domains.
    base = seed*1000
    observed = circle_cases(owner, seed=base+11, count=cfg["observed"], split="support-observed")
    dreams = circle_cases(owner, seed=base+23, count=cfg["dreams"], split="support-conditional", conditional=True)
    oracle = circle_cases(owner, seed=base+23, count=cfg["dreams"], split="support-oracle")
    dev = circle_cases(owner, seed=base+37, count=cfg["development"], split="development-motion")
    partition = {"observed": observed, "conditional": dreams, "development": dev}
    if phase == "final":
        partition["final"] = circle_cases(owner, seed=base+101, count=cfg["final"], split="evaluation-motion")
        partition["extent"] = circle_cases(owner, seed=base+113, count=cfg["extent"], split="evaluation-extent", extent=True)
    partition_audit(partition)
    # Oracle twins are intentionally paired control data, not an independent split.
    maximum = max(float(np.max(np.abs(np.asarray(d.target)-o.target))) for d, o in zip(dreams, oracle))
    if maximum > 1e-10:
        raise ValueError("Conditional geometry does not match the declared family")
    return observed, dreams, oracle, dev, partition, maximum


def evaluate_retention(solver, seed, cfg):
    report, scores = evaluate_shared(solver, world_specs(), seed=seed,
                                     samples=cfg["retention_samples"], retained_samples=cfg["retention_samples"],
                                     typed_samples=cfg["typed_samples"], world_learning=True)
    groups = {name: float(np.mean(values)) for name, values in scores.capabilities.items()}
    for partition, details in report["typed_neural_only"].items():
        groups.update({f"neural/{partition}/{task}": value["score"] for task, value in details["tasks"].items()})
    return {"groups": groups, "report": report, "scores": score_record(scores)}


def run(name, seed, arm, *, pilot=False, overrides=None):
    directory = RELEASE/name
    if pilot:
        cfg = {**DEFAULT, **(overrides or {})}
        directory.mkdir(parents=True, exist_ok=True)
        protocol_sha = None
    else:
        contract = check_contract(directory)
        if seed not in contract["seeds"] or arm not in contract["arms"]:
            raise ValueError("Undeclared final arm/seed")
        cfg, protocol_sha = contract["config"], read(directory/"protocol.json")["sha256"]
    destination = directory/f"seed-{seed}"/arm
    destination.mkdir(parents=True, exist_ok=False)
    checkpoints = ROOT/"runs"/"A06-transfer"/name/f"seed-{seed}"/arm
    checkpoints.mkdir(parents=True, exist_ok=False)
    write(destination/"status.json", {"status": "RUNNING", "pilot": pilot, "config": cfg, "protocol": protocol_sha})
    started = time.perf_counter()
    solver = parent()
    parent_solver_identity = solver.identity()
    owner = solver.components["r1"]
    owner_id = SharedOwnerRef.from_solver(solver).identity
    observed, dreams, oracle, dev, partitions, teacher_difference = data_bank(owner, cfg, seed, "pilot" if pilot else "final")
    write(destination/"partition-audit.json", partition_audit(partitions))
    rows_record = {name: [r.record() for r in rows] for name, rows in partitions.items()}
    # Identical seed-bank bytes in every arm are audited; retain one common copy.
    bank_path = directory/f"seed-{seed}"/"bank.json.gz"
    bank_bytes = gzip.compress(canonical(rows_record).encode(), mtime=0)
    if bank_path.exists() and bank_path.read_bytes() != bank_bytes:
        raise ValueError("Paired arm inputs changed")
    if not bank_path.exists():
        bank_path.write_bytes(bank_bytes)
    evidence = SharedEvidence.load(EVIDENCE)
    evidence.validate()
    resumed = []

    def checkpoint(step, model, optimizer, best_state, history, draw_state, replay_state, anchor_state, work):
        path = checkpoints/f"step-{step}.pt"
        if path.exists():
            raise FileExistsError("Checkpoint is immutable")
        original = owner.state_dict()
        def delta(state):
            return {n: v.detach().cpu().clone() for n, v in state.items() if not torch.equal(v, original[n])}
        torch.save({"schema": "conditional-transfer-resume-v1", "model_delta": delta(model.state_dict()),
                    "config": model.export_config(), "optimizer": optimizer.state_dict(), "step": step,
                    "best_delta": delta(best_state), "work": work,
                    "history": history, "draw_rng": draw_state, "replay_rng": replay_state, "anchor_rng": anchor_state,
                    "torch_rng": torch.get_rng_state(), "protocol_sha256": protocol_sha,
                    "arm": arm, "seed": seed, "training_config": cfg,
                    "parent_checkpoint_sha256": PARENT_SHA}, path)
        resumed.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
        write(destination/"status.json", {"status": "TRAINING", "step": step, "checkpoint": resumed[-1],
                                         "pilot": pilot, "protocol": protocol_sha})

    learned, training = fit(owner, evidence, observed, dreams, oracle, dev, cfg, arm, seed, checkpoint)
    selected = checkpoints/"selected.pt"
    if not SHARED_BASE.exists():
        save_shared_checkpoint(SHARED_BASE, owner)
    if state_digest(load_shared_checkpoint(SHARED_BASE)) != state_digest(owner):
        raise ValueError("Immutable shared checkpoint base changed")
    save_shared_checkpoint(selected, learned, parent=SHARED_BASE)
    restored = load_shared_checkpoint(selected)
    if state_digest(restored) != state_digest(learned):
        raise ValueError("Selected neural checkpoint failed exact restoration")
    evaluated = copy.deepcopy(solver)
    if arm in {"detached", "extra_capacity"}:
        evaluated.components["typed"] = IndependentTypedControl(learned)
    else:
        replace_shared_owner(evaluated, learned)
        if evaluated.neural.owner is not learned or evaluated.components["typed"].owner is not learned:
            raise ValueError("Shared route aliasing was lost")
    retained = evaluate_retention(evaluated, seed+5000000, cfg)
    reference_path = directory/f"seed-{seed}"/"parent-retention.json.gz"
    if not reference_path.exists():
        reference_path.write_bytes(gzip.compress(canonical(evaluate_retention(solver, seed+5000000, cfg)).encode(), mtime=0))
    baseline = json.loads(gzip.decompress(reference_path.read_bytes()))
    drops = {name: baseline["groups"][name]-value for name, value in retained["groups"].items()}
    metrics, raw = {}, {}
    score_rows = {"development": dev} if pilot else {k: partitions[k] for k in ("final", "extent")}
    for kind, rows in score_rows.items():
        metrics[kind], raw[kind] = motion_score(learned, rows)
        metrics[kind+"_parent"], raw[kind+"_parent"] = motion_score(owner, rows)
    if solver.identity() != parent_solver_identity or sha(CHECKPOINT) != PARENT_SHA:
        raise ValueError("Original parent changed")
    (destination/"raw.json.gz").write_bytes(gzip.compress(canonical({"motion": raw, "retention": retained}).encode(), mtime=0))
    result = {"status": "COMPLETE", "pilot": pilot, "arm": arm, "seed": seed, "config": cfg,
              "protocol_sha256": protocol_sha, "training": training, "metrics": metrics,
              "retention_drops": drops, "maximum_retention_drop": max(drops.values()),
              "retention_groups": len(drops), "teacher_sha256": teacher_identity(owner),
              "conditional_vs_simulator_max_target_difference": teacher_difference,
              "original_solver_unchanged": True, "original_owner_sha256": owner_id,
              "evaluated_owner_sha256": SharedOwnerRef.from_solver(evaluated).identity,
              "shared_owner_was_updated": arm not in {"detached", "extra_capacity"},
              "selected_checkpoint": {"path": selected.relative_to(ROOT).as_posix(), "sha256": sha(selected), "bytes": selected.stat().st_size},
              "resumable_checkpoints": resumed, "total_worker_seconds": time.perf_counter()-started,
              "checkpoint_bytes": sum(r["bytes"] for r in resumed)+selected.stat().st_size,
              "raw_sha256": sha(destination/"raw.json.gz"), "bank_sha256": sha(bank_path),
              "source_sha256": digest(source_files()),
              "supported_answer_certificate": None,
              "stale_dependencies": "Original live situations/libraries stay bound to the immutable parent; none rebound to updated weights."}
    write(destination/"result.json", result)
    write(destination/"status.json", {"status": "COMPLETE", "result_sha256": sha(destination/"result.json")})
    print(json.dumps({"arm": arm, "seed": seed, "metrics": metrics, "maximum_retention_drop": max(drops.values()),
                      "seconds": result["total_worker_seconds"], "trainable_parameters": training["trainable_parameters"]}))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "pilot", "freeze", "run"))
    parser.add_argument("--name", default="A06-PILOT-001")
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--arm", choices=ARMS, default="integrated")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[181, 191, 211, 223, 227])
    args = parser.parse_args()
    cfg = read(args.config) if args.config else DEFAULT
    if args.action == "inspect":
        solver = parent()
        owner = solver.components["r1"]
        print(json.dumps({"config": owner.export_config(), "parameters": sum(p.numel() for p in owner.parameters()),
                          "teacher": teacher_identity(owner), "coefficients": owner.generator_mean[0].tolist(),
                          "probabilities": owner.generator_class_probabilities.tolist(),
                          "observations": int(owner.generator_observations), "parent_sha256": sha(CHECKPOINT)}, indent=2))
    elif args.action == "freeze":
        freeze(args.name, cfg, args.seeds)
    else:
        run(args.name, args.seed, args.arm, pilot=args.action == "pilot", overrides=cfg)


if __name__ == "__main__":
    main()
