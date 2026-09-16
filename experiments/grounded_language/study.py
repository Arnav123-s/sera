"""Prospective grounded-language experiments with resumable, owned CPU training."""

import argparse
import gzip
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sera.session_state import model_identity
from workbench.model import Learner

from .data import TOKENS, examples, execute, independent_solutions
from .model import LanguageR1, fingerprint, tensors, trainable

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/21_grounded_language"
BASE = ROOT/"research-continuation/20_live_workbench/application-snapshot.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed(path, data):
    Path(path).write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def freeze(directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    paths = sorted([*Path(__file__).parent.glob("*.py"), ROOT/"scripts/run_language_bounded.py",
                    ROOT/"scripts/run_v3_bounded.py", ROOT/"scripts/windows_job_v3.py",
                    ROOT/"tests/test_grounded_language.py"])
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    protocol = {"schema": "sera.grounded-language.protocol.1", "config": config,
                "base_snapshot": BASE.relative_to(ROOT).as_posix(), "base_sha256": sha(BASE),
                "predecessor_release_sha256": sha(ROOT/"research-continuation/20_live_workbench/release-manifest.json"),
                "language_source": fingerprint(), "sources": sources,
                "boundary": "Supplied finite sentence grammar and arithmetic. Formal checking certifies the decoded statement, not its translation. No raw-document or general English learning claim.",
                "split": "Semantic triples hash to residues 2,3,4 for training; 1 for development/calibration; 0 for final. No final data are read in a pilot.",
                "unit": "Fixed generated sentence bank; repeated semantic triples are not independent experiments.",
                "selection": "Select lowest development joint cross-entropy at the declared checkpoint intervals, never by final evaluation.",
                "promotion": {"known_exact_translation": .95, "accepted_translation_accuracy": .99,
                              "accepted_coverage": .5, "maximum_qualified_retention_drop": .02},
                "retention_policy": "User permits forgetting for research. Automatic replacement of the working release still requires its measured tasks to remain usable; failed replacements remain separately callable research checkpoints."}
    write(directory/"protocol.json", protocol)
    with zipfile.ZipFile(directory/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.writestr(name, (ROOT/name).read_bytes())
    return protocol


def check(directory):
    protocol = read(directory/"protocol.json")
    if sha(ROOT/protocol["base_snapshot"]) != protocol["base_sha256"]:
        raise ValueError("Frozen predecessor changed")
    for name, expected in protocol["sources"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Frozen experiment source changed: {name}")
    return protocol


def construct(seed, *, scratch=False):
    learner = Learner(read(BASE))
    owner = LanguageR1.attach(learner.session.owner, seed)
    if scratch:
        # Same owned architecture and language initialization, but reset shared weights.
        # This is a control, not an alternate claimed persistent successor.
        torch.manual_seed(seed+900000)
        for container in (owner.memory, owner.fusion):
            for module in container.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
    if learner.solver.components["r1"] is not owner or learner.solver.components["typed"].owner is not owner or learner.solver.neural.owner is not owner:
        raise ValueError("Actual shared ownership lost")
    return learner


@torch.no_grad()
def evaluate(owner, rows, *, threshold=0., reset_memory=False):
    owner.eval()
    raw = []
    total_loss = 0.
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        ids, targets = tensors(batch)
        logits = owner.language(ids, reset_memory=reset_memory)
        total_loss += sum(float(F.cross_entropy(value, targets[:, i], reduction="sum")) for i, value in enumerate(logits))
        probabilities = [value.softmax(-1) for value in logits]
        labels = torch.stack([value.argmax(-1) for value in probabilities], 1).tolist()
        confidence = torch.stack([value.max(-1).values for value in probabilities], 1).min(1).values.tolist()
        for example, decoded, score in zip(batch, labels, confidence):
            from .data import tokens
            missing = sorted(set(tokens(example["text"])[1])-set(owner.language_words))
            result = execute(decoded)
            raw.append({"text": example["text"], "target": example["labels"], "decoded": decoded,
                        "confidence": score, "translation_correct": decoded == example["labels"],
                        "answer_correct": result["solutions"] == independent_solutions(example["labels"]),
                        "solutions": result["solutions"], "formal_check": result["formal_verification"],
                        "missing_words": missing, "accepted": not missing and score >= threshold})
    accepted = [r for r in raw if r["accepted"]]
    summary = {"examples": len(raw), "joint_cross_entropy": total_loss/len(raw),
               "exact_translation": float(np.mean([r["translation_correct"] for r in raw])),
               "answer_accuracy": float(np.mean([r["answer_correct"] for r in raw])),
               "slot_accuracy": [float(np.mean([r["decoded"][i] == r["target"][i] for r in raw])) for i in range(4)],
               "formal_checks": sum(r["formal_check"] for r in raw),
               "accepted": len(accepted), "coverage": len(accepted)/len(raw),
               "accepted_translation_accuracy": float(np.mean([r["translation_correct"] for r in accepted])) if accepted else None,
               "accepted_answer_accuracy": float(np.mean([r["answer_correct"] for r in accepted])) if accepted else None,
               "threshold": threshold, "unique_texts": len({r["text"] for r in raw}),
               "unique_semantic_triples": len({tuple(r["target"][1:]) for r in raw})}
    return summary, raw


def calibrate(raw):
    # Predeclared finite grid; this is an empirical filter, not a probability guarantee.
    for threshold in (0., .5, .7, .8, .9, .95, .975, .99, .995, 1.000001):
        accepted = [r for r in raw if not r["missing_words"] and r["confidence"] >= threshold]
        if len(accepted) >= 100 and np.mean([r["translation_correct"] for r in accepted]) >= .995:
            return {"threshold": threshold, "accepted": len(accepted), "calibration_examples": len(raw),
                    "empirical_accuracy": float(np.mean([r["translation_correct"] for r in accepted]))}
    return {"threshold": 1.000001, "accepted": 0, "calibration_examples": len(raw), "empirical_accuracy": None}


def save_checkpoint(path, owner, optimizer, *, step, best, best_loss, history, rng, config, protocol_sha, base_state):
    if path.exists():
        raise FileExistsError("Completed checkpoints are immutable")
    delta = {name: value.detach().clone() for name, value in owner.state_dict().items()
             if name not in base_state or not torch.equal(value, base_state[name])}
    state = {"schema": "sera.grounded-language.checkpoint.1", "delta": delta, "config": config,
             "owner_config": owner.export_config(), "owner_sha256": model_identity(owner),
             "optimizer": optimizer.state_dict(), "step": step, "best_state": best,
             "best_loss": best_loss, "history": history, "numpy_rng": rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "protocol_sha256": protocol_sha,
             "base_snapshot_sha256": sha(BASE)}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size, "step": step}


def run(args):
    torch.set_num_threads(1)
    directory = RELEASE/args.name
    checkpoints = ROOT/"runs/grounded-language"/args.name
    config = {"seed": args.seed, "scope": args.scope, "scratch": args.scratch, "steps": args.steps,
              "batch": args.batch, "train": args.train, "development": args.development,
              "learning_rate": args.learning_rate, "checkpoint_every": args.checkpoint_every, "pilot": not args.final}
    if args.resume:
        protocol = check(directory)
        if protocol["config"] != config or (directory/"result.json").exists():
            raise ValueError("Resume requires the exact unfinished experiment")
    else:
        protocol = freeze(directory, config)
        checkpoints.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha(directory/"protocol.json")
    started = time.perf_counter()
    learner = construct(args.seed, scratch=args.scratch)
    owner = learner.session.owner
    base_state = {name: value.clone() for name, value in Learner(read(BASE)).session.owner.state_dict().items()}
    original = model_identity(owner)
    rows = examples(args.seed+10, args.train, "train")
    dev = examples(args.seed+20, args.development, "development")
    if not args.resume:
        packed(directory/"teaching.json.gz", {"train": rows, "development": dev})
    ids, labels = tensors(rows)
    trainable(owner, args.scope)
    parameters = [p for p in owner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.0001)
    rng = np.random.default_rng(args.seed+30)
    history, best, best_loss, step, receipts = [], None, float("inf"), 0, []
    if args.resume:
        receipt = read(directory/"status.json")
        path = ROOT/receipt["checkpoint"]["path"]
        if sha(path) != receipt["checkpoint"]["sha256"]:
            raise ValueError("Resume checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["protocol_sha256"] != protocol_sha or saved["base_snapshot_sha256"] != sha(BASE):
            raise ValueError("Resume predecessor or protocol changed")
        state = owner.state_dict()
        state.update(saved["delta"])
        owner.load_state_dict(state, strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        step, best, best_loss, history = saved["step"], saved["best_state"], saved["best_loss"], saved["history"]
        receipts = receipt["checkpoints"]
    else:
        initial, _ = evaluate(owner, dev)
        write(directory/"initial.json", initial)
    for index in range(step, args.steps):
        owner.train()
        selection = rng.integers(len(rows), size=args.batch)
        logits = owner.language(ids[selection])
        loss = sum(F.cross_entropy(value, labels[selection, i]) for i, value in enumerate(logits))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite language objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": index+1, "loss": float(loss.detach())})
        if (index+1) % args.checkpoint_every == 0 or index+1 == args.steps:
            scored, _ = evaluate(owner, dev)
            history[-1]["development"] = scored
            if scored["joint_cross_entropy"] < best_loss:
                best_loss = scored["joint_cross_entropy"]
                best = {name: value.detach().clone() for name, value in owner.state_dict().items()
                        if name not in base_state or not torch.equal(value, base_state[name])}
            receipt = save_checkpoint(checkpoints/f"step-{index+1:05d}.pt", owner, optimizer,
                                      step=index+1, best=best, best_loss=best_loss, history=history,
                                      rng=rng, config=config, protocol_sha=protocol_sha, base_state=base_state)
            receipts.append(receipt)
            write(directory/"status.json", {"status": "TRAINING", "step": index+1, "checkpoint": receipt, "checkpoints": receipts})
            print(json.dumps({"step": index+1, "loss": history[-1]["loss"], "development": scored,
                              "worker_seconds": time.perf_counter()-started}), flush=True)
    selected = construct(args.seed, scratch=args.scratch)
    owner = selected.session.owner
    selected_state = owner.state_dict()
    selected_state.update(best)
    owner.load_state_dict(selected_state, strict=True)
    checkpoint = {"schema": "sera.grounded-language.selected.1", "config": config,
                  "delta": best, "owner_config": owner.export_config(), "owner_sha256": model_identity(owner),
                  "base_snapshot_sha256": sha(BASE), "protocol_sha256": protocol_sha}
    if (checkpoints/"selected.pt").exists():
        previous = torch.load(checkpoints/"selected.pt", map_location="cpu", weights_only=True)
        if previous["owner_sha256"] != checkpoint["owner_sha256"] or previous["protocol_sha256"] != protocol_sha:
            raise ValueError("An existing selected checkpoint differs; preserve and diagnose")
    else:
        torch.save(checkpoint, checkpoints/"selected.pt")
    selected_receipt = {"path": (checkpoints/"selected.pt").relative_to(ROOT).as_posix(), "sha256": sha(checkpoints/"selected.pt")}
    calibration, cal_raw = evaluate(owner, examples(args.seed+40, max(512, args.development), "calibration"))
    cutoff = calibrate(cal_raw)
    write(directory/"calibration.json", {"summary": calibration, "filter": cutoff})
    packed(directory/"calibration-raw.json.gz", cal_raw)
    split = "final" if args.final else "development"
    metrics, raw = {}, {}
    for name in ("known", "novel"):
        metric, detail = evaluate(owner, examples(args.seed+50, 512, split, wording=name), threshold=cutoff["threshold"])
        metrics[name], raw[name] = metric, detail
    metrics["memory_reset"], raw["memory_reset"] = evaluate(owner, examples(args.seed+50, 512, split),
                                                          threshold=cutoff["threshold"], reset_memory=True)
    packed(directory/"evaluation.json.gz", raw)
    result = {"status": "COMPLETE", "config": config, "protocol_sha256": protocol_sha,
              "initial_owner": original, "selected_owner": model_identity(owner),
              "shared_owner_aliases": selected.solver.components["r1"] is owner and selected.solver.neural.owner is owner and selected.solver.components["typed"].owner is owner,
              "owner_settings": owner.settings, "metrics": metrics, "calibration": cutoff,
              "trainable_parameters": sum(p.numel() for p in parameters), "training_draws": args.steps*args.batch,
              "checkpoints": receipts, "selected_checkpoint": selected_receipt, "worker_seconds_this_invocation": time.perf_counter()-started,
              "language_source": fingerprint(), "vocabulary_size": len(TOKENS),
              "qualified": None, "qualification_boundary": "Retention and independent audit are still required; final metrics are not a deployment decision."}
    write(directory/"result.json", result)
    write(directory/"status.json", {"status": "COMPLETE", "result_sha256": sha(directory/"result.json")})
    print(json.dumps(result, indent=2), flush=True)


def load_selected(directory):
    directory = Path(directory)
    result = read(directory/"result.json")
    receipt = result["selected_checkpoint"]
    if sha(ROOT/receipt["path"]) != receipt["sha256"]:
        raise ValueError("Selected checkpoint changed")
    saved = torch.load(ROOT/receipt["path"], map_location="cpu", weights_only=True)
    if saved["base_snapshot_sha256"] != sha(BASE) or saved["owner_config"]["language_source"] != fingerprint():
        raise ValueError("Language or predecessor source changed")
    config = saved["config"]
    learner = construct(config["seed"], scratch=config["scratch"])
    owner = learner.session.owner
    state = owner.state_dict()
    state.update(saved["delta"])
    owner.load_state_dict(state, strict=True)
    owner.language_words = saved["owner_config"]["language_words_admitted"]
    if model_identity(owner) != saved["owner_sha256"]:
        raise ValueError("Restored language owner changed")
    return learner, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=150001)
    p.add_argument("--scope", choices=("interface", "shared"), default="shared")
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train", type=int, default=4096)
    p.add_argument("--development", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=.003)
    p.add_argument("--final", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 1 <= args.steps <= 3000 or not 1 <= args.batch <= 64 or not 128 <= args.train <= 16384 or not 32 <= args.development <= 1024 or not 1 <= args.checkpoint_every <= 100:
        raise ValueError("Bounded declared training configuration required")
    run(args)


if __name__ == "__main__":
    main()
