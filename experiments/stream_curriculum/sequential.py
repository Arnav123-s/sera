"""Matched current-example/total-work continual teaching with bounded replay."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from experiments.constraint_inquiry.data import corpus as old_corpus
from experiments.constraint_inquiry.data import encode as old_encode

from .data import PREPARED, ROOT, RowStream, first_rows, read, sha, write
from .model import apply, delta
from .study import attach, identities, load_parent, measure, update

OUT = ROOT / "research-continuation/26_stream_curriculum"


class Reservoir:
    def __init__(self, seed, capacity=256, saved=None):
        self.capacity, self.rows, self.seen = capacity, [], set()
        self.insert_rng, self.sample_rng = random.Random(seed + 731), random.Random(seed + 991)
        if saved is not None:
            if saved["capacity"] != capacity:
                raise ValueError("Changed replay capacity")
            self.rows, self.seen = copy.deepcopy(saved["rows"]), set(saved["seen"])
            self.insert_rng.setstate(saved["insert_rng"])
            self.sample_rng.setstate(saved["sample_rng"])

    def sample(self, current):
        return self.sample_rng.choices(self.rows, k=len(current)) if self.rows else copy.deepcopy(current)

    def admit(self, rows):
        if any(row["partition"] != "train" for row in rows):
            raise ValueError("Only source training examples may enter replay")
        for row in rows:
            identifier = str(row["id"])
            if identifier in self.seen:
                continue
            self.seen.add(identifier)
            if len(self.rows) < self.capacity:
                self.rows.append(copy.deepcopy(row))
            else:
                index = self.insert_rng.randrange(len(self.seen))
                if index < self.capacity:
                    self.rows[index] = copy.deepcopy(row)

    def snapshot(self):
        return {"capacity": self.capacity, "rows": copy.deepcopy(self.rows), "seen": sorted(self.seen),
                "insert_rng": self.insert_rng.getstate(), "sample_rng": self.sample_rng.getstate()}


def prepare(seed):
    folder = ROOT / "runs" / f"SC-sequential-view-{seed}"
    source = PREPARED / "train.jsonl"
    if folder.exists():
        manifest = read(folder / "manifest.json")
        if manifest["source"] != sha(source) or any(sha(ROOT / r["path"]) != r["sha256"] for r in manifest["blocks"]):
            raise ValueError("Changed sequential source")
        return manifest
    folder.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    domains = sorted({r["scenario"] for r in rows})
    if len(domains) != 18:
        raise ValueError("Declared curriculum needs exactly 18 source scenarios")
    blocks = []
    for i in range(6):
        names = domains[3 * i:3 * i + 3]
        selected = [r for r in rows if r["scenario"] in names]
        random.Random(seed + i).shuffle(selected)
        path = folder / f"block-{i}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as file:
            for row in selected:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        blocks.append({"domains": names, "rows": len(selected), "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
    manifest = {"seed": seed, "source": sha(source), "blocks": blocks}
    write(folder / "manifest.json", manifest)
    return manifest


def save(path, owner, optimizer, stream, reservoir, config, step, history, assessments):
    if path.exists():
        raise ValueError("Never overwrite a sequential checkpoint")
    value = {"schema": "sera.stream-training.1", "identities": identities(), "config": config,
             "step": step, "delta": delta(owner), "optimizer": optimizer.state_dict(),
             "torch_rng": torch.get_rng_state(), "stream": stream.snapshot(),
             "reservoir": reservoir.snapshot(), "history": history, "assessments": assessments,
             "seen_ids": sorted(reservoir.seen)}
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = None if args.resume is None else torch.load(args.resume, weights_only=True, map_location="cpu")
    if saved is None:
        config = {"vocabulary": read(PREPARED / "vocabulary.json"), "seed": args.seed,
                  "kind": "shared", "limit": 0, "buffer_size": 128, "batch_size": 32,
                  "learning_rate": .003, "replay": args.replay, "curriculum": prepare(args.seed),
                  "trainer_source": sha(Path(__file__)), "sequential_protocol": sha(OUT / "sequential-protocol.md")}
    else:
        config = saved["config"]
        if (saved["identities"] != identities() or config["trainer_source"] != sha(Path(__file__))
                or config["sequential_protocol"] != sha(OUT / "sequential-protocol.md")):
            raise ValueError("Changed sequential training identity")
    if any(sha(ROOT / block["path"]) != block["sha256"] for block in config["curriculum"]["blocks"]):
        raise ValueError("Changed current or future curriculum block")
    write(output / "freeze.json", {"identities": identities(), "config": config,
                                    "resume": None if args.resume is None else {"path": str(args.resume), "sha256": sha(args.resume)}})
    base = load_parent()
    old = {n: v.detach().clone() for n, v in base.owner.state_dict().items()}
    old_x = torch.tensor(old_encode([r["text"] for part in ("train", "pairs") for r in old_corpus(part)]))
    with torch.no_grad():
        old_logits = [x.clone() for x in base.owner.binding(old_x)]
    owner = attach(base, config)
    optimizer = torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=.003)
    step, history, assessments = 0, [], []
    reservoir = Reservoir(config["seed"], saved=None if saved is None else saved["reservoir"])
    if saved is not None:
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        step, history, assessments = saved["step"], saved["history"], saved["assessments"]
    if step >= 1800:
        raise ValueError("Sequential lifetime already completed")
    development = first_rows("dev", 2033)
    initial_step = step
    blocks = config["curriculum"]["blocks"]
    current_block = min(step // 300, 5)
    # A boundary checkpoint retains the previous stream; the next block starts fresh.
    stream_state = None if saved is None or step % 300 == 0 else saved["stream"]
    stream = RowStream(ROOT / blocks[current_block]["path"], config["seed"] + current_block,
                       buffer_size=128, state=stream_state)
    last = None
    while step < 1800 and time.perf_counter() - started < args.work_seconds:
        block = step // 300
        if block != current_block:
            current_block = block
            stream = RowStream(ROOT / blocks[block]["path"], config["seed"] + block, buffer_size=128)
        current = stream.take(16)
        replayed = reservoir.sample(current)  # Both arms advance the same sampling RNG.
        rows = current + (replayed if config["replay"] else current)
        history.append(update(owner, optimizer, rows, config["vocabulary"]))
        reservoir.admit(current)
        step += 1
        if step % 300 == 0:
            measured = measure(owner, development, config["vocabulary"])
            path = output / f"development-block-{block}.json"
            write(path, measured["records"])
            assessments.append({"block": block, "step": step, "records": path.relative_to(ROOT).as_posix(),
                                "sha256": sha(path), **{k: v for k, v in measured.items() if k != "records"}})
        if step % 100 == 0:
            last = output / f"step-{step:06d}.pt"
            save(last, owner, optimizer, stream, reservoir, config, step, history, assessments)
            print(json.dumps({"step": step, "block": block, "loss": history[-1]["loss"],
                              "seconds": time.perf_counter() - started}), flush=True)
    if last is None or step % 100:
        last = output / f"step-{step:06d}.pt"
        save(last, owner, optimizer, stream, reservoir, config, step, history, assessments)
    assert all(torch.equal(v, owner.state_dict()[n]) for n, v in old.items())
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(old_logits, owner.binding(old_x)))
    result = {"schema": "sera.stream-sequential.1", "steps": step, "new_steps": step - initial_step,
              "presentations": step * 32, "current_stream_presentations": step * 16,
              "additional_presentations": step * 16, "unique_training_examples": len(reservoir.seen),
              "replay": config["replay"], "reservoir_records": len(reservoir.rows), "seed": config["seed"],
              "old_tensors_unchanged": len(old), "retained_language_requests": len(old_x),
              "checkpoint": last.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(last),
              "assessments": assessments, "blocks": blocks,
              "numerical_worker_seconds": time.perf_counter() - started,
              "status": "planned_updates_complete" if step == 1800 else "resumable_resource_checkpoint"}
    write(output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("assessments", "blocks")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2651)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--work-seconds", type=float, default=205)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("A fresh local run path is required")
    torch.set_num_threads(1)
    train(args)


if __name__ == "__main__":
    main()
