"""Preserve the source instance while intervening on its actual acquired K."""

import argparse
import copy
import gzip
import json
import platform
import time
import zipfile

import numpy as np
import torch

from experiments.acquisition_dependence.study import assert_acquired_effect, load_before, truth
from experiments.cross_route_transfer.data import (
    CONDITIONS,
    MotionCase,
    partition_audit,
    state_digest,
    teacher_identity,
)
from experiments.cross_route_transfer.learning import fit, motion_score
from experiments.cross_route_transfer.study import (
    EVIDENCE,
    ROOT,
    SHARED_BASE,
    evaluate_retention,
    parent,
    read,
    sha,
    source_files,
    write,
)
from sera.contracts import EvidenceKind, Observation, Provenance
from sera.shared import replace_shared_owner
from sera.shared_archive import load_shared_checkpoint, save_shared_checkpoint
from sera.shared_learning import SharedEvidence
from sera.storage import canonical, digest

RELEASE = ROOT/"research-continuation/18_instance_transfer"
ARMS = ("observed_only", "corrected_teacher", "uncorrected_teacher", "oracle_exposure")
SEEDS = (307, 311, 313, 317, 331)
CONFIG = {"steps": 256, "batch_size": 32, "replay_batch": 32, "learning_rate": .0003,
          "replay_weight": 1.0, "anchor_weight": 4.0, "anchor_batch": 32, "scope": "readout",
          "select_every": 64, "observed": 32, "dreams": 512, "development": 128,
          "final": 256, "extent": 128, "retention_samples": 128, "typed_samples": 128}


def sources():
    result = source_files()
    for folder in ("experiments/acquisition_dependence", "experiments/instance_transfer"):
        result.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/folder).glob("*.py")})
    for name in ("scripts/run_instance_bounded.py", "scripts/run_instance_batch.ps1", "tests/test_instance_transfer.py"):
        result[name] = sha(ROOT/name)
    return result


def cases(owner, *, seed, count, split, conditional, extent=False):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-1, 1, count)
    delta = rng.uniform(.20 if extent else .04, .30 if extent else .17, count)*rng.choice([-1, 1], count)
    phases = phase[:, None] + np.arange(4)*delta[:, None]
    if conditional:
        with torch.no_grad():
            points = owner.forward_generator(phases.reshape(-1).tolist())["class_means"][0].numpy().reshape(count, 4, 2)
        source, kind = "raw-acquired-circle", EvidenceKind.PREDICTION
    else:
        points = truth(phases)
        source, kind = "original-instance-simulator", EvidenceKind.OBSERVATION
    result = []
    for index in range(count):
        provenance = Provenance(source, f"{split}/{seed}/{index}", kind)
        observations = tuple(Observation("numeric", tuple(map(float, point)), position, provenance, units="m")
                             for position, point in enumerate(points[index, :3]))
        result.append(MotionCase(observations, tuple(map(float, points[index, 3])),
                                 "conditional_model_target" if conditional else "simulator_ground_truth",
                                 split, teacher_identity(owner) if conditional else None, CONDITIONS))
    return result


def bank(after, before, seed, cfg=CONFIG, *, final=True):
    offset = seed*1000
    result = {
        "observed": cases(after, seed=offset+11, count=cfg["observed"], split="support-instance", conditional=False),
        "corrected": cases(after, seed=offset+23, count=cfg["dreams"], split="support-corrected", conditional=True),
        "uncorrected": cases(before, seed=offset+23, count=cfg["dreams"], split="support-uncorrected", conditional=True),
        "oracle": cases(after, seed=offset+23, count=cfg["dreams"], split="support-oracle", conditional=False),
        "development": cases(after, seed=offset+37, count=cfg["development"], split="development-instance", conditional=False),
    }
    if final:
        result.update({"final": cases(after, seed=offset+101, count=cfg["final"], split="evaluation-instance", conditional=False),
                       "extent": cases(after, seed=offset+113, count=cfg["extent"], split="evaluation-instance-extent", conditional=False, extent=True)})
    partition_audit(result)
    # The same phase/action schedule is held fixed. Raw coordinates retain K.
    effect = assert_acquired_effect([r.target for r in result["corrected"]], [r.target for r in result["uncorrected"]])
    return result, effect


def freeze():
    RELEASE.mkdir(parents=True, exist_ok=True)
    output = RELEASE/"protocol.json"
    if output.exists():
        raise FileExistsError("Protocol already frozen")
    before, before_sha = load_before()
    solver = parent()
    after = solver.components["r1"]
    if any(not torch.equal(value, dict(after.named_parameters())[name]) for name, value in before.named_parameters()):
        raise ValueError("K intervention also changed theta")
    preflight, effect = bank(after, before, 293, final=False)
    payload = {"experiment": "A06-INSTANCE-001", "config": CONFIG, "seeds": SEEDS, "arms": ARMS,
               "source_sha256": sources(), "before_checkpoint_sha256": before_sha,
               "after_solver_sha256": solver.identity(), "preflight": {"seed": 293, "maximum_target_effect": effect},
               "source_knowledge": "Real preserved before/after independent sensor correction, neural initial parameters identical",
               "historical_acquisition": {"observations_each": int(after.generator_observations),
                                          "corrections_before": int(before.generator_labels),
                                          "corrections_after": int(after.generator_labels),
                                          "cost_status": "Previously executed acquisition retained in SHARED-GG-001; the additional independent correction is not free information."},
               "question": "Does keeping acquired source coordinates let its correction improve a different neural prediction route?",
               "change_from_normalized_study": "No re-centering, radius normalization, phase compensation or similarity augmentation. The identity transform is used. Original instance coordinates are retained.",
               "training": "Same 514-parameter readout, optimizer, replay, anchor loss, counts and selection schedule fixed from development before this experiment; no tuning from normalized final outcomes.",
               "arms_detail": {"observed_only": "32 new observed trajectories; repeated exposure",
                               "corrected_teacher": "Same 32 plus 512 conditional raw-coordinate trajectories from corrected K",
                               "uncorrected_teacher": "Same 32 plus matched 512 phase schedules from preserved incorrect K",
                               "oracle_exposure": "Same 32 plus independent simulator labels/input trajectories for the matched 512 schedules"},
               "gate": "Corrected teacher reduces mean MSE by at least 10% versus both observed-only and uncorrected teacher; both paired 97.5% bootstrap intervals exclude zero; improves both comparisons in all five streams; worst 61-group old-score loss <=0.02.",
               "uncertainty": "20,000 paired bootstrap draws over five learning-stream means, seed 18060915; 1.25/98.75 percentiles for two planned contrasts. Descriptive, not a finite-sample guarantee.",
               "controls_reused": "Exact detached/extra-capacity identity and full-source ownership controls from the frozen 17_transfer study use the same fit implementation. This cycle tests K attribution, not a new ownership topology.",
               "scope": "One original physical instance, supplied circle family, noiseless future simulator trajectories, conditional teacher labels. Not a new law, independent physical environments, improved shared representation or learned eta.",
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write(output, {"payload": payload, "sha256": digest(payload)})
    write(RELEASE/"preflight.json", {"maximum_target_effect": effect,
                                    "partitions": partition_audit(preflight), "no_final_data_generated": True})
    with zipfile.ZipFile(RELEASE/"frozen-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in payload["source_sha256"]:
            archive.write(ROOT/name, name)
    print(json.dumps({"status": "FROZEN", "sha256": digest(payload), "intervention_effect": effect}))


def contract():
    item = read(RELEASE/"protocol.json")
    if item["sha256"] != digest(item["payload"]) or item["payload"]["source_sha256"] != sources():
        raise ValueError("Instance-transfer source or protocol changed")
    return item


def run(seed, arm):
    spec = contract()
    cfg = spec["payload"]["config"]
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Undeclared learning stream")
    path = RELEASE/f"seed-{seed}"/arm
    path.mkdir(parents=True, exist_ok=False)
    checkpoints = ROOT/"runs/A06-instance-transfer"/f"seed-{seed}"/arm
    checkpoints.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original = parent()
    after = original.components["r1"]
    before, before_sha = load_before()
    if before_sha != spec["payload"]["before_checkpoint_sha256"] or original.identity() != spec["payload"]["after_solver_sha256"]:
        raise ValueError("An acquired-state predecessor changed")
    owner = before if arm == "uncorrected_teacher" else after
    data, effect = bank(after, before, seed, cfg)
    recorded = {name: [r.record() for r in rows] for name, rows in data.items()}
    raw_bank = gzip.compress(canonical(recorded).encode(), mtime=0)
    bank_path = RELEASE/f"seed-{seed}"/"bank.json.gz"
    if bank_path.exists() and bank_path.read_bytes() != raw_bank:
        raise ValueError("Paired source data changed")
    if not bank_path.exists():
        bank_path.write_bytes(raw_bank)
    write(path/"status.json", {"status": "RUNNING", "protocol_sha256": spec["sha256"]})
    evidence = SharedEvidence.load(EVIDENCE)
    mapped_arm = arm if arm in {"observed_only", "oracle_exposure"} else "integrated"
    dreams = data["uncorrected"] if arm == "uncorrected_teacher" else data["corrected"]
    saved = []

    def checkpoint(step, model, optimizer, best_state, history, draw_rng, replay_rng, anchor_rng, work):
        reference = owner.state_dict()
        def delta(state):
            return {n: v.detach().clone() for n, v in state.items() if not torch.equal(v, reference[n])}
        target = checkpoints/f"step-{step}.pt"
        torch.save({"model_delta": delta(model.state_dict()), "best_delta": delta(best_state),
                    "optimizer": optimizer.state_dict(), "step": step, "history": history,
                    "draw_rng": draw_rng, "replay_rng": replay_rng, "anchor_rng": anchor_rng,
                    "torch_rng": torch.get_rng_state(), "work": work, "arm": mapped_arm, "seed": seed,
                    "research_arm": arm, "training_config": cfg, "protocol_sha256": spec["sha256"],
                    "teacher_identity": teacher_identity(owner)}, target)
        saved.append({"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target), "bytes": target.stat().st_size})
        write(path/"status.json", {"status": "TRAINING", "checkpoint": saved[-1]})

    learned, training = fit(owner, evidence, data["observed"], dreams, data["oracle"], data["development"],
                            cfg, mapped_arm, seed, checkpoint)
    selected = checkpoints/"selected.pt"
    save_shared_checkpoint(selected, learned, parent=SHARED_BASE)
    restored = load_shared_checkpoint(selected)
    if state_digest(restored) != state_digest(learned):
        raise ValueError("Raw-instance descendant does not restore exactly")
    evaluated = copy.deepcopy(original)
    replace_shared_owner(evaluated, learned)
    retention = evaluate_retention(evaluated, seed+5000000, cfg)
    baseline_path = RELEASE/f"seed-{seed}"/"parent-retention.json.gz"
    if not baseline_path.exists():
        baseline_path.write_bytes(gzip.compress(canonical(evaluate_retention(original, seed+5000000, cfg)).encode(), mtime=0))
    baseline = json.loads(gzip.decompress(baseline_path.read_bytes()))
    drops = {name: baseline["groups"][name]-value for name, value in retention["groups"].items()}
    metrics, raw = {}, {}
    for kind in ("final", "extent"):
        metrics[kind], raw[kind] = motion_score(learned, data[kind])
        metrics[kind+"_parent"], raw[kind+"_parent"] = motion_score(after, data[kind])
    (path/"raw.json.gz").write_bytes(gzip.compress(canonical({"motion": raw, "retention": retention}).encode(), mtime=0))
    result = {"status": "COMPLETE", "arm": arm, "seed": seed, "training": training, "metrics": metrics,
              "retention_drops": drops, "maximum_retention_drop": max(drops.values()),
              "source_intervention_effect": effect, "protocol_sha256": spec["sha256"],
              "selected_checkpoint": {"path": selected.relative_to(ROOT).as_posix(), "sha256": sha(selected), "bytes": selected.stat().st_size},
              "checkpoints": saved, "bank_sha256": sha(bank_path), "raw_sha256": sha(path/"raw.json.gz"),
              "worker_seconds": time.perf_counter()-started, "parent_unchanged": original.identity() == spec["payload"]["after_solver_sha256"],
              "numerical_initialization_matched": True, "gate": "Assessed only after every declared arm/stream completes"}
    if not result["parent_unchanged"]:
        raise ValueError("Original owner changed")
    write(path/"result.json", result)
    write(path/"status.json", {"status": "COMPLETE", "result_sha256": sha(path/"result.json")})
    print(json.dumps({"arm": arm, "seed": seed, "metrics": metrics, "retention_drop": max(drops.values())}))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--arm", choices=ARMS, default="corrected_teacher")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run(args.seed, args.arm)
