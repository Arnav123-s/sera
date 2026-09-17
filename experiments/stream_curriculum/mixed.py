"""Compare global source mixing at the same supervised optimizer budget."""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from experiments.constraint_inquiry.data import corpus as old_corpus
from experiments.constraint_inquiry.data import encode as old_encode

from .data import PREPARED, ROOT, RowStream, first_rows, read, sha, write
from .model import apply
from .study import attach, checkpoint, identities, load_parent, measure, update

OUT = ROOT / "research-continuation/26_stream_curriculum"


def prepare(seed):
    folder = ROOT / "runs" / f"SC-mixed-view-{seed}"
    source = PREPARED / "train.jsonl"
    if folder.exists():
        saved = read(folder / "manifest.json")
        if saved["source"] != sha(source) or saved["view_sha256"] != sha(folder / "train.jsonl"):
            raise ValueError("Changed source or prepared permutation")
        return saved
    folder.mkdir(parents=True, exist_ok=False)
    offsets = []
    with source.open("rb") as file:
        while True:
            offset = file.tell()
            if not file.readline():
                break
            offsets.append(offset)
    random.Random(seed).shuffle(offsets)
    with source.open("rb") as file, (folder / "train.jsonl").open("xb") as target:
        for offset in offsets:
            file.seek(offset)
            target.write(file.readline())
    write(folder / "offsets.json", offsets)
    saved = {"source": sha(source), "seed": seed, "rows": len(offsets),
             "view": (folder / "train.jsonl").relative_to(ROOT).as_posix(),
             "view_sha256": sha(folder / "train.jsonl"), "offsets_sha256": sha(folder / "offsets.json"),
             "offset_count": len(offsets), "algorithm": "Python Random(seed).shuffle of raw row offsets; same order each epoch"}
    write(folder / "manifest.json", saved)
    return saved


def train(args):
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = None if args.resume is None else torch.load(args.resume, map_location="cpu", weights_only=True)
    if saved is None:
        schedule = prepare(args.seed)
        config = {"vocabulary": read(PREPARED / "vocabulary.json"), "seed": args.seed,
                  "kind": args.kind, "limit": 0, "buffer_size": 128, "batch_size": 32,
                  "learning_rate": .003, "schedule": schedule, "trainer_source": sha(Path(__file__)),
                  "mixed_protocol": sha(OUT / "mixed-protocol.md")}
    else:
        config = saved["config"]
        if (saved["identities"] != identities() or config["trainer_source"] != sha(Path(__file__))
                or config["mixed_protocol"] != sha(OUT / "mixed-protocol.md")):
            raise ValueError("Changed mixed experiment identity")
        schedule = config["schedule"]
    stream_path = ROOT / schedule["view"]
    if sha(stream_path) != schedule["view_sha256"]:
        raise ValueError("Changed permutation")
    write(output / "freeze.json", {"identities": identities(), "config": config, "target_step": args.steps,
                                    "resume": None if args.resume is None else {"path": str(args.resume), "sha256": sha(args.resume)}})
    base = load_parent()
    old = {n: v.detach().clone() for n, v in base.owner.state_dict().items()}
    old_x = torch.tensor(old_encode([r["text"] for part in ("train", "pairs") for r in old_corpus(part)]))
    with torch.no_grad():
        old_logits = [x.clone() for x in base.owner.binding(old_x)]
    owner = attach(base, config)
    optimizer = torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=.003)
    stream = RowStream(stream_path, config["seed"], buffer_size=128, state=None if saved is None else saved["stream"])
    step, history, seen = 0, [], set()
    if saved is not None:
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        step, history, seen = saved["step"], saved["history"], set(saved["seen_ids"])
    if args.steps <= step:
        raise ValueError("Mixed updates already completed")
    start_step = step
    last_checkpoint = None
    while step < args.steps and time.perf_counter() - started < args.work_seconds:
        rows = stream.take(32)
        history.append(update(owner, optimizer, rows, config["vocabulary"]))
        seen.update(str(r["id"]) for r in rows)
        step += 1
        if step % 200 == 0:
            last_checkpoint = output / f"step-{step:06d}.pt"
            checkpoint(last_checkpoint, owner, optimizer, stream, config, step, history, seen)
            print(json.dumps({"step": step, "loss": history[-1]["loss"], "seconds": time.perf_counter() - started}), flush=True)
    if last_checkpoint is None or step % 200:
        last_checkpoint = output / f"step-{step:06d}.pt"
        checkpoint(last_checkpoint, owner, optimizer, stream, config, step, history, seen)
    after = {"teaching": measure(owner, first_rows("train", 256), config["vocabulary"]),
             "development": measure(owner, first_rows("dev", 128), config["vocabulary"])}
    write(output / "after.json", after)
    assert all(torch.equal(v, owner.state_dict()[n]) for n, v in old.items())
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(old_logits, owner.binding(old_x)))
    result = {"schema": "sera.stream-development.1", "phase": "development_only",
              "steps": step, "new_steps": step - start_step, "presentations": step * 32,
              "unique_training_examples": len(seen), "old_tensors_unchanged": len(old),
              "retained_language_requests": len(old_x), "schedule": schedule,
              "checkpoint": last_checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(last_checkpoint),
              "after": {k: {n: v for n, v in row.items() if n != "records"} for k, row in after.items()},
              "numerical_worker_seconds": time.perf_counter() - started,
              "final_test_opened": False, "status": "planned_updates_complete" if step == args.steps else "resumable_resource_checkpoint"}
    write(output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "after"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2641)
    parser.add_argument("--kind", choices=("shared", "pooled"), default="shared")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--work-seconds", type=float, default=210)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("A fresh local run path is required")
    torch.set_num_threads(1)
    train(args)


if __name__ == "__main__":
    main()
