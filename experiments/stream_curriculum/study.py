"""Finite streamed optimization, actual-owner checks and preserved development results."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.constraint_inquiry.data import corpus as old_corpus
from experiments.constraint_inquiry.data import encode as old_encode
from experiments.constraint_inquiry.runtime import ConstraintRuntime

from .data import PREPARED, ROOT, RowStream, first_rows, read, sha, write
from .model import StreamR1, apply, batch, delta, source

OUT = ROOT / "research-continuation/26_stream_curriculum"


def load_parent():
    record = read(OUT / "parent.json")
    return ConstraintRuntime(record["checkpoint"], record)


def attach(base, config):
    learner = base.base.session.learner
    facts, library = learner.legacy.snapshot(), learner.library.record()
    StreamR1.attach(base.owner, config["vocabulary"], config["seed"], config["kind"])
    learner.rebind(facts, library)
    if not (base.owner is learner.session.owner is learner.solver.neural.owner
            is learner.solver.components["typed"].owner is base.base.session.owner):
        raise AssertionError("Lost actual shared ownership")
    for name, parameter in base.owner.named_parameters():
        parameter.requires_grad_(name.startswith("stream_"))
    base.owner.eval()
    return base.owner


def spans(tags):
    result, active = [], None
    for i, tag in enumerate([*tags, "O"]):
        continuing = active is not None and tag == "I-" + active[0]
        if active is not None and not continuing:
            result.append((active[0], active[1], i))
            active = None
        if tag != "O" and not continuing:
            active = (tag[2:], i)
    return set(result)


@torch.no_grad()
def measure(owner, rows, vocabulary):
    records, correct, exact, tp, fp, fn, token_correct, tokens = [], 0, 0, 0, 0, 0, 0, 0
    for offset in range(0, len(rows), 32):
        selected = rows[offset:offset + 32]
        x, _, _ = batch(selected, vocabulary)
        intents, slots = owner.request_logits(x)
        confidence = intents.softmax(-1).max(-1).values.tolist()
        for i, (row, intent, tags) in enumerate(zip(selected, intents.argmax(-1).tolist(), slots.argmax(-1).tolist())):
            pred_intent = vocabulary["intents"][intent]
            pred_tags = [vocabulary["tags"][k] for k in tags[:len(row["tokens"])]]
            predicted, expected = spans(pred_tags), spans(row["tags"])
            matched = pred_intent == row["intent"]
            correct += matched
            exact += matched and predicted == expected
            tp += len(predicted & expected)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            tokens += len(pred_tags)
            token_correct += sum(a == b for a, b in zip(pred_tags, row["tags"]))
            records.append({"id": row["id"], "partition": row["partition"], "text": row["text"],
                            "scenario": row["scenario"], "target_intent": row["intent"],
                            "predicted_intent": pred_intent, "target_tags": row["tags"],
                            "predicted_tags": pred_tags, "softmax_score": confidence[i]})
    return {"examples": len(rows), "intent_correct": correct, "intent_accuracy": correct / len(rows),
            "exact_frames": exact, "frame_accuracy": exact / len(rows),
            "slot_true_positive": tp, "slot_false_positive": fp, "slot_false_negative": fn,
            "slot_span_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "token_accuracy": token_correct / tokens, "records": records}


def update(owner, optimizer, rows, vocabulary):
    x, intents, slots = batch(rows, vocabulary)
    optimizer.zero_grad(set_to_none=True)
    intent_logits, slot_logits = owner.request_logits(x)
    iloss = F.cross_entropy(intent_logits, intents)
    sloss = F.cross_entropy(slot_logits.flatten(0, 1), slots.flatten())
    loss = iloss + sloss
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in owner.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "intent_loss": float(iloss.detach()),
            "slot_loss": float(sloss.detach()), "gradient_norm_before_clip": float(norm)}


def identities():
    return {"interface_source": source(), "study_source": sha(Path(__file__)),
            "parent": sha(OUT / "parent.json"), "data": sha(PREPARED / "manifest.json"),
            "protocol": sha(OUT / "protocol.md")}


def checkpoint(path, owner, optimizer, stream, config, step, history, seen):
    value = {"schema": "sera.stream-training.1", "identities": identities(), "config": config,
             "step": step, "delta": delta(owner), "optimizer": optimizer.state_dict(),
             "torch_rng": torch.get_rng_state(), "stream": stream.snapshot(),
             "history": history, "seen_ids": sorted(seen)}
    if path.exists():
        raise ValueError("Never overwrite a completed training checkpoint")
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = None if args.resume is None else torch.load(args.resume, weights_only=True, map_location="cpu")
    if saved is not None and saved["identities"] != identities():
        raise ValueError("Changed source, predecessor or data: an explicit new cohort is required")
    config = ({"vocabulary": read(PREPARED / "vocabulary.json"), "seed": args.seed,
               "kind": args.kind, "limit": args.limit, "buffer_size": 128,
               "batch_size": 32, "learning_rate": .003} if saved is None else saved["config"])
    write(args.output / "freeze.json", {"identities": identities(), "config": config,
                                        "resume": None if args.resume is None else {"path": str(args.resume), "sha256": sha(args.resume)},
                                        "target_step": args.steps, "soft_work_seconds": args.work_seconds})
    base = load_parent()
    old = {n: v.detach().clone() for n, v in base.owner.state_dict().items()}
    old_x = torch.tensor(old_encode([r["text"] for part in ("train", "pairs") for r in old_corpus(part)]))
    with torch.no_grad():
        old_logits = [x.clone() for x in base.owner.binding(old_x)]
    owner = attach(base, config)
    optimizer = torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=config["learning_rate"])
    stream = RowStream(PREPARED / "train.jsonl", config["seed"], limit=config["limit"],
                       buffer_size=config["buffer_size"], state=None if saved is None else saved["stream"])
    step, history, seen = 0, [], set()
    if saved is not None:
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        step, history, seen = saved["step"], saved["history"], set(saved["seen_ids"])
    if args.steps <= step:
        raise ValueError("Requested updates are already completed")
    vocabulary = config["vocabulary"]
    teaching = first_rows("train", min(config["limit"] or 256, 256))
    development = first_rows("dev", 128)
    before = {"teaching": measure(owner, teaching, vocabulary), "development": measure(owner, development, vocabulary)}
    write(args.output / "before.json", before)
    probe = next(owner.fusion.parameters())
    probe.requires_grad_(True)
    x, y, _ = batch(teaching[:8], vocabulary)
    probe_loss = F.cross_entropy(owner.request_logits(x)[0], y)
    derivative, = torch.autograd.grad(probe_loss, probe, allow_unused=True)
    derivative_norm = 0. if derivative is None else float(derivative.norm())
    probe.requires_grad_(False)
    if config["kind"] == "shared" and derivative_norm == 0.:
        raise AssertionError("No derivative through the shared fusion")
    initial_step = step
    last_checkpoint = None
    while step < args.steps and time.perf_counter() - started < args.work_seconds:
        rows = stream.take(config["batch_size"])
        history.append(update(owner, optimizer, rows, vocabulary))
        seen.update(str(r["id"]) for r in rows)
        step += 1
        if step % 20 == 0:
            last_checkpoint = args.output / f"step-{step:06d}.pt"
            checkpoint(last_checkpoint, owner, optimizer, stream, config, step, history, seen)
            print(json.dumps({"step": step, "loss": history[-1]["loss"], "seconds": time.perf_counter() - started}), flush=True)
    if last_checkpoint is None or step % 20:
        last_checkpoint = args.output / f"step-{step:06d}.pt"
        checkpoint(last_checkpoint, owner, optimizer, stream, config, step, history, seen)
    after = {"teaching": measure(owner, teaching, vocabulary), "development": measure(owner, development, vocabulary)}
    write(args.output / "after.json", after)
    if not all(torch.equal(v, owner.state_dict()[n]) for n, v in old.items()):
        raise AssertionError("Predecessor tensor changed")
    with torch.no_grad():
        if not all(torch.equal(a, b) for a, b in zip(old_logits, owner.binding(old_x))):
            raise AssertionError("Earlier finite language behavior changed")
    result = {"schema": "sera.stream-development.1", "phase": "development_only",
              "steps": step, "new_steps": step - initial_step, "presentations": step * config["batch_size"],
              "unique_training_examples": len(seen), "old_tensors_unchanged": len(old),
              "retained_language_requests": len(old_x), "shared_fusion_gradient_norm": derivative_norm,
              "new_parameters": sum(p.numel() for n, p in owner.named_parameters() if n.startswith("stream_")),
              "checkpoint": last_checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(last_checkpoint),
              "numerical_worker_seconds": time.perf_counter() - started,
              "before": {k: {n: v for n, v in row.items() if n != "records"} for k, row in before.items()},
              "after": {k: {n: v for n, v in row.items() if n != "records"} for k, row in after.items()},
              "final_test_opened": False, "status": "planned_updates_complete" if step == args.steps else "resumable_resource_checkpoint"}
    write(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2631)
    parser.add_argument("--kind", choices=("shared", "pooled"), default="shared")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--work-seconds", type=float, default=25)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "prepare":
        from .prepare import main as prepare_main
        prepare_main()
    else:
        if args.output is None or not args.output.resolve().is_relative_to(ROOT / "runs"):
            raise ValueError("A fresh output inside runs is required")
        args.output = args.output.resolve()
        train(args)


if __name__ == "__main__":
    main()
