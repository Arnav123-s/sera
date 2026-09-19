"""Staged teaching, honest measurement and genuinely resumable state.

Four of the reviewed defects live here rather than in the model:

* an evaluation that could not run returned loss 0 and perplexity 1, which reads
  as perfect prediction. Every measurement below returns an explicit
  ``UNAVAILABLE`` record with counts and a reason instead, and an unavailable
  measurement never qualifies or selects anything (finding 4);
* ragged batches were built by direct tensor construction, which fails on an
  uneven final chunk. Batches are right-padded and the loss is masked
  (finding 4);
* the declared four-stage ladder indexed its last rung twice, so mathematics and
  science were never taught, and a source that yielded nothing could spin
  forever. The schedule is derived from the protocol and validated, every
  declared source is drawn from in an explicit round-robin mix with recorded
  exposure, and a pass that makes no progress is a recorded failure (finding 6);
* checkpoints omitted the parent contract and carried no optimiser, RNG or data
  cursor, so nothing could resume. A checkpoint here carries the full baseline
  contract, a distinct descendant identity, the optimiser, all three RNG
  streams, every source cursor and the metric history, written atomically
  through a pointer (finding 7).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .corpus import load_items

LAB_ROOT = Path(__file__).resolve().parents[2]
RUNS = LAB_ROOT / "runs/owner-learning-001"


# ---------------------------------------------------------------- schedule --

def validate_schedule(stages, available_sources):
    """Derive and check the ladder instead of indexing the last rung twice."""
    if not stages:
        raise ValueError("A curriculum needs at least one stage")
    cumulative = 0
    validated = []
    seen = set()
    for position, stage in enumerate(stages):
        name, sources, steps = stage["name"], list(stage["sources"]), int(stage["steps"])
        if name in seen:
            raise ValueError(f"Duplicate stage name {name}")
        seen.add(name)
        if steps <= 0:
            raise ValueError(f"Stage {name} declares a non-positive step budget")
        if not sources:
            raise ValueError(f"Stage {name} declares no source")
        missing = [source for source in sources if source not in available_sources]
        if missing:
            raise ValueError(f"Stage {name} declares sources with no training items: {missing}")
        cumulative += steps
        if "cumulative_steps" in stage and stage["cumulative_steps"] != cumulative:
            raise ValueError(f"Stage {name} declares cumulative {stage['cumulative_steps']}, derived {cumulative}")
        validated.append({"index": position, "name": name, "sources": sources, "steps": steps,
                          "cumulative_steps": cumulative,
                          "objective": stage.get("objective", "next-token prediction on human text")})
    taught = {source for stage in validated for source in stage["sources"]}
    if taught != set(available_sources):
        raise ValueError(f"The ladder teaches {sorted(taught)} but training sources are {sorted(available_sources)}")
    return validated


# ------------------------------------------------------------------ streams --

class SourceStream:
    """A resumable, deterministic cursor over one source's training items."""

    def __init__(self, source, items, tokens, *, length=64, seed=4801):
        self.source = source
        self.tokens = tokens
        self.length = length
        self.seed = seed
        self.items = sorted(items, key=lambda item: item["id"])
        if not self.items:
            raise ValueError(f"Source {source} has no training items")
        self.epoch = 0
        self.position = 0
        self.consumed = 0
        self.token_count = 0
        self.groups = set()
        self._order = self._permutation(0)

    def _permutation(self, epoch):
        generator = np.random.default_rng(
            int(hashlib.sha256(f"{self.seed}:{self.source}:{epoch}".encode()).hexdigest()[:16], 16))
        return generator.permutation(len(self.items)).tolist()

    def cursor(self):
        return {"source": self.source, "epoch": self.epoch, "position": self.position,
                "consumed": self.consumed, "tokens": self.token_count, "groups": sorted(self.groups)}

    def restore(self, cursor):
        if cursor["source"] != self.source:
            raise ValueError("Cursor belongs to a different source")
        self.epoch = cursor["epoch"]
        self.position = cursor["position"]
        self.consumed = cursor["consumed"]
        self.token_count = cursor["tokens"]
        self.groups = set(cursor["groups"])
        self._order = self._permutation(self.epoch)

    def take(self, count):
        """Return up to ``count`` encoded sequences, advancing the cursor."""
        rows = []
        while len(rows) < count:
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._permutation(self.epoch)
            item = self.items[self._order[self.position]]
            self.position += 1
            ids = self.tokens.encode(item["text"], limit=self.length)
            if len(ids) < 4:
                continue
            rows.append(ids)
            self.consumed += 1
            self.token_count += len(ids)
            self.groups.add(item["group"])
        return rows


class MixedStream:
    """An explicit round-robin over a stage's declared sources.

    Exhausting the first source before the step budget would not establish
    exposure to every declared source, so each batch is drawn across all of
    them, and the per-source counts are recorded rather than assumed.
    """

    def __init__(self, streams, sources):
        self.streams = {source: streams[source] for source in sources}
        self.sources = list(sources)
        self.turn = 0

    def batch(self, size):
        rows, exposure = [], Counter()
        for _ in range(size):
            source = self.sources[self.turn % len(self.sources)]
            self.turn += 1
            taken = self.streams[source].take(1)
            if not taken:
                continue
            rows.append(taken[0])
            exposure[source] += 1
        if not rows:
            raise ValueError("No source in this stage produced a usable example; recorded as no progress")
        return rows, dict(exposure)


def pad_batch(rows):
    """Right-padded ragged batch; padding is position 0 and is never a target."""
    width = max(len(row) for row in rows)
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)


# -------------------------------------------------------------- measurement --

def unavailable(family, reason, counts=None):
    return {"family": family, "status": "UNAVAILABLE", "reason": reason,
            "counts": counts or {}, "qualifies": False}


def measure_next_token(owner, rows, *, batch=8, limit=None):
    """Mean next-token negative log-likelihood on held-out human passages."""
    rows = rows[:limit] if limit else rows
    if not rows:
        return unavailable("next_token", "no eligible held-out passages", {"rows": 0})
    total, counted, used, failures = 0., 0, 0, []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = [row["ids"] for row in rows[start:start + batch] if len(row["ids"]) >= 2]
            if not chunk:
                continue
            try:
                loss, tokens = owner.lex_next_token_loss(pad_batch(chunk))
            except Exception as error:
                failures.append(type(error).__name__ + ": " + str(error))
                continue
            total += float(loss) * tokens
            counted += tokens
            used += len(chunk)
    if counted == 0:
        return unavailable("next_token", "every batch failed or contained no predicted token",
                           {"rows": len(rows), "failures": len(failures), "examples": failures[:3]})
    mean = total / counted
    return {"family": "next_token", "status": "MEASURED", "passages": used, "predicted_tokens": counted,
            "mean_nll": mean, "perplexity": float(np.exp(mean)), "batch_failures": failures[:3],
            "failed_batches": len(failures), "qualifies": not failures}


def measure_choice(owner, tokens, rows, family, *, batch=16):
    """Single-token forced choice scored at the final position of the prompt."""
    if not rows:
        return unavailable(family, "no eligible held-out questions", {"rows": 0})
    prompts, answers, options, identifiers = [], [], [], []
    for row in rows:
        if family == "cloze":
            ids = tokens.encode_words(row["prefix"])
        else:
            ids = [tokens.define, *tokens.encode_words(row["definition"]), tokens.is_token]
        if len(ids) < 2:
            continue
        prompts.append(ids)
        answers.append(row["answer"])
        options.append(row["candidates"])
        identifiers.append(row["id"])
    if not prompts:
        return unavailable(family, "no question produced a usable prompt", {"rows": len(rows)})
    order = sorted(range(len(prompts)), key=lambda position: len(prompts[position]))
    correct, scored, failures = 0, 0, []
    per_source = Counter()
    per_source_correct = Counter()
    source_of = {row["id"]: row["source"] for row in rows}
    with torch.no_grad():
        start = 0
        while start < len(order):
            group = [order[start]]
            length = len(prompts[order[start]])
            start += 1
            while start < len(order) and len(prompts[order[start]]) == length and len(group) < batch:
                group.append(order[start])
                start += 1
            try:
                logits, _ = owner.lex_logits(torch.tensor([prompts[i] for i in group], dtype=torch.long))
                logprobs = F.log_softmax(logits[:, -1], -1)
            except Exception as error:
                failures.append(type(error).__name__ + ": " + str(error))
                continue
            for position, index in enumerate(group):
                candidate_ids = tokens.encode_words(options[index])
                values = logprobs[position, torch.tensor(candidate_ids)]
                chosen = options[index][int(values.argmax())]
                scored += 1
                source = source_of[identifiers[index]]
                per_source[source] += 1
                if chosen == answers[index]:
                    correct += 1
                    per_source_correct[source] += 1
    if scored == 0:
        return unavailable(family, "every batch failed", {"rows": len(rows), "failures": len(failures),
                                                          "examples": failures[:3]})
    chance = rows[0]["chance"]
    return {"family": family, "status": "MEASURED", "questions": scored, "correct": correct,
            "accuracy": correct / scored, "chance": chance,
            "by_source": {source: {"questions": per_source[source], "correct": per_source_correct[source],
                                   "accuracy": per_source_correct[source] / per_source[source]}
                          for source in sorted(per_source)},
            "batch_failures": failures[:3], "failed_batches": len(failures), "qualifies": not failures}


def evaluate(owner, tokens, evaluation, *, families=("next_token", "cloze", "definition"), limits=None):
    limits = limits or {}
    results = {}
    for family in families:
        rows = evaluation["families"].get(family, [])
        rows = rows[:limits[family]] if family in limits else rows
        started = time.perf_counter()
        if family == "next_token":
            results[family] = measure_next_token(owner, rows)
        else:
            results[family] = measure_choice(owner, tokens, rows, family)
        results[family]["seconds"] = time.perf_counter() - started
    results["split"] = evaluation["split"]
    results["all_measured"] = all(results[family]["status"] == "MEASURED" for family in families)
    return results


# -------------------------------------------------------------- checkpoints --

def atomic_write_bytes(path, data):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path, value):
    atomic_write_bytes(path, (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8"))


class CheckpointStore:
    """Append-only descendant revisions behind one atomically replaced pointer."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.revisions = self.directory / "revisions"

    def write(self, payload, metadata):
        self.revisions.mkdir(parents=True, exist_ok=True)
        number = metadata["revision"]
        stem = f"{number:06d}"
        weights = self.revisions / (stem + ".pt")
        buffer = torch.save
        temporary = Path(str(weights) + ".tmp")
        with temporary.open("wb") as handle:
            buffer(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(weights)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        metadata = {**metadata, "weights": weights.name, "weights_sha256": digest}
        meta_path = self.revisions / (stem + ".json")
        atomic_write_json(meta_path, metadata)
        atomic_write_json(self.directory / "current.json",
                          {"revision": meta_path.name,
                           "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()})
        return metadata

    def read(self):
        pointer = self.directory / "current.json"
        if not pointer.exists():
            return None
        reference = json.loads(pointer.read_text())
        meta_path = self.revisions / reference["revision"]
        raw = meta_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Descendant checkpoint metadata failed its integrity check")
        metadata = json.loads(raw)
        weights = self.revisions / metadata["weights"]
        if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Descendant checkpoint weights failed their integrity check")
        return metadata, torch.load(weights, weights_only=False)

    def history(self):
        return sorted(path.name for path in self.revisions.glob("*.json")) if self.revisions.exists() else []


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(state):
    random.setstate(tuple(state["python"]) if isinstance(state["python"], list) else state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# ------------------------------------------------------------------ trainer --

class Trainer:
    """One staged teaching run over the human corpus, resumable at any step."""

    def __init__(self, owner, tokens, schedule, *, arm="connected", learning_rate=1e-3, batch=8,
                 length=64, seed=4801, store=None, baseline_contract=None):
        self.owner = owner
        self.tokens = tokens
        self.arm = arm
        self.batch = batch
        self.length = length
        self.seed = seed
        self.baseline_contract = baseline_contract or {}
        trainable = [parameter for parameter in owner.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameter; the descendant would learn nothing")
        self.optimizer = torch.optim.Adam(trainable, lr=learning_rate)
        self.learning_rate = learning_rate
        sources = sorted({source for stage in schedule for source in stage["sources"]})
        items = load_items(split="train")
        self.streams = {source: SourceStream(source, [row for row in items if row["source"] == source],
                                             tokens, length=length, seed=seed)
                        for source in sources}
        self.schedule = schedule
        self.store = CheckpointStore(store) if store is not None else None
        self.step = 0
        self.revision = 0
        self.metrics = []
        self.stage_records = []
        self.parent_identity = None

    # -- state ------------------------------------------------------------

    def state(self):
        return {"schema": "sera.owner-language.descendant-checkpoint.1",
                "arm": self.arm, "step": self.step, "revision": self.revision,
                "seed": self.seed, "batch": self.batch, "length": self.length,
                "learning_rate": self.learning_rate,
                "parent_owner": self.parent_identity,
                "descendant_owner": self.owner.identity(),
                "baseline_contract": copy.deepcopy(self.baseline_contract),
                "language_config": copy.deepcopy(self.owner.language_config),
                "vocabulary_digest": self.tokens.digest,
                "schedule": copy.deepcopy(self.schedule),
                "cursors": {source: stream.cursor() for source, stream in self.streams.items()},
                "stage_records": copy.deepcopy(self.stage_records),
                "metrics": copy.deepcopy(self.metrics)}

    def payload(self):
        return {"language_state": self.owner.language_state(),
                "optimizer": self.optimizer.state_dict(),
                "rng": rng_state()}

    def save(self, *, note=None):
        if self.store is None:
            raise ValueError("This trainer has no checkpoint store")
        metadata = {**self.state(), "revision": self.revision, "note": note,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        written = self.store.write(self.payload(), metadata)
        self.revision += 1
        return written

    def load_latest(self):
        if self.store is None:
            return None
        found = self.store.read()
        if found is None:
            return None
        metadata, payload = found
        if metadata["arm"] != self.arm:
            raise ValueError("Checkpoint belongs to a different experimental arm")
        if metadata["vocabulary_digest"] != self.tokens.digest:
            raise ValueError("Checkpoint was written against a different frozen vocabulary")
        if self.baseline_contract and metadata["baseline_contract"] != self.baseline_contract:
            raise ValueError("Checkpoint carries a different baseline contract")
        self.owner.load_language_state(payload["language_state"])
        self.optimizer.load_state_dict(payload["optimizer"])
        set_rng_state(payload["rng"])
        for source, cursor in metadata["cursors"].items():
            self.streams[source].restore(cursor)
        self.step = metadata["step"]
        self.revision = metadata["revision"] + 1
        self.metrics = metadata["metrics"]
        self.stage_records = metadata["stage_records"]
        if self.owner.identity() != metadata["descendant_owner"]:
            raise ValueError("Restored descendant identity differs from the saved checkpoint")
        return metadata

    # -- teaching ---------------------------------------------------------

    def train_step(self, mixed):
        rows, exposure = mixed.batch(self.batch)
        ids = pad_batch(rows)
        self.optimizer.zero_grad(set_to_none=True)
        loss, counted = self.owner.lex_next_token_loss(ids)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.owner.parameters() if parameter.requires_grad], 1.)
        self.optimizer.step()
        self.step += 1
        return {"step": self.step, "loss": float(loss), "tokens": counted,
                "grad_norm": float(norm), "exposure": exposure}

    def run_stage(self, stage, *, report_every=25, checkpoint_every=None, on_checkpoint=None,
                  stop_after=None):
        mixed = MixedStream(self.streams, stage["sources"])
        started = time.perf_counter()
        before = {source: self.streams[source].cursor() for source in stage["sources"]}
        target = stage["cumulative_steps"]
        losses = []
        no_progress = 0
        while self.step < target:
            if stop_after is not None and self.step >= stop_after:
                return None
            try:
                record = self.train_step(mixed)
            except ValueError as error:
                no_progress += 1
                if no_progress >= 3:
                    failure = {"stage": stage["name"], "status": "NO_PROGRESS",
                               "reason": str(error), "step": self.step}
                    self.stage_records.append(failure)
                    raise RuntimeError(json.dumps(failure)) from error
                continue
            no_progress = 0
            losses.append(record["loss"])
            if self.step % report_every == 0:
                self.metrics.append({"step": self.step, "stage": stage["name"],
                                     "mean_loss": float(np.mean(losses[-report_every:])),
                                     "tokens": record["tokens"], "grad_norm": record["grad_norm"],
                                     "seconds": time.perf_counter() - started})
                print(json.dumps(self.metrics[-1]), flush=True)
            if checkpoint_every and self.step % checkpoint_every == 0:
                written = self.save(note=f"{stage['name']}:step-{self.step}")
                if on_checkpoint:
                    on_checkpoint(written)
        after = {source: self.streams[source].cursor() for source in stage["sources"]}
        record = {"stage": stage["name"], "status": "COMPLETE", "steps": stage["steps"],
                  "cumulative_steps": stage["cumulative_steps"], "sources": stage["sources"],
                  "objective": stage["objective"], "seconds": time.perf_counter() - started,
                  "mean_loss_first_quarter": float(np.mean(losses[:max(1, len(losses) // 4)])) if losses else None,
                  "mean_loss_last_quarter": float(np.mean(losses[-max(1, len(losses) // 4):])) if losses else None,
                  "exposure": {source: {"examples": after[source]["consumed"] - before[source]["consumed"],
                                        "tokens": after[source]["tokens"] - before[source]["tokens"],
                                        "groups_seen_total": len(after[source]["groups"])}
                               for source in stage["sources"]}}
        if any(value["examples"] == 0 for value in record["exposure"].values()):
            record["status"] = "DECLARED_SOURCE_NOT_EXPOSED"
        self.stage_records.append(record)
        return record
