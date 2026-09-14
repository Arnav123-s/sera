"""Typed observations reach learned encoders, recurrent memory and requested decoders."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.models import ModelConfig, StatefulModel
from sera.storage import digest

TASKS = ("modular_sum", "spatial_relation", "byte_sum", "patch_quadrant", "tone", "motion", "binding")
MODALITIES = ("symbolic", "numeric", "text_bytes", "image_patch", "audio_frame")
UNITS = (None, "dimensionless", "m", "cm", "s", "ms", "pixel", "amplitude")
UNIT_FACTORS = {None: 1, "dimensionless": 1, "m": 1, "cm": .01, "s": 1, "ms": .001,
                "pixel": 1, "amplitude": 1}


@dataclass(frozen=True)
class TypedExample:
    observations: tuple[Observation, ...]
    task: str
    target: int | tuple[float, float]
    evidence: Provenance
    split: str

    def __post_init__(self):
        if self.task not in TASKS or not self.observations or not isinstance(self.split, str) or not self.split:
            raise ValueError("Invalid typed example")
        if self.task == "motion":
            if not isinstance(self.target, tuple) or len(self.target) != 2 or not all(map(math.isfinite, self.target)):
                raise ValueError("Motion targets need two finite meter coordinates")
        elif type(self.target) is not int or not 0 <= self.target < 4:
            raise ValueError("Categorical targets must be in [0, 4)")

    @property
    def identifier(self):
        return digest(asdict(self))


class TypedEvidence:
    def __init__(self, records):
        self.records, self.identifiers = [], set()
        for row in records:
            if row.task not in TASKS or not row.observations or not row.split:
                raise ValueError("Typed example violates the task contract")
            if row.evidence.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}:
                raise ValueError("Typed learning requires independently admitted targets")
            if row.split.startswith(("test", "query", "evaluation", "promotion")):
                raise ValueError("Sealed typed evaluation cannot enter learning")
            if row.identifier not in self.identifiers:
                self.records.append(row)
                self.identifiers.add(row.identifier)


class TypedEncoder(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.width = width
        self.adapters = nn.ModuleDict({name: nn.Sequential(nn.Linear(42, width), nn.Tanh(),
                                                           nn.Linear(width, width)) for name in MODALITIES})

    @staticmethod
    def features(observation):
        if len(observation.values) > 16 or observation.units not in UNITS:
            raise ValueError("Observation exceeds the declared width or unit vocabulary")
        available = observation.available or (True,) * len(observation.values)
        factor = UNIT_FACTORS[observation.units] * observation.scale
        divisor = 255 if observation.modality == "text_bytes" else 8 if observation.modality in {"numeric", "symbolic"} else 1
        values = [float(value) * factor / divisor if present else 0.0
                  for value, present in zip(observation.values, available)]
        if any(abs(value) > 32 for value in values):
            raise ValueError("Observation exceeds the declared normalized dynamic range")
        mask = [float(value) for value in available]
        values += [0.0] * (16 - len(values))
        mask += [0.0] * (16 - len(mask))
        unit = [float(index == UNITS.index(observation.units)) for index in range(len(UNITS))]
        return values + mask + [observation.position / 64, math.log2(observation.scale) / 12] + unit

    def forward(self, observations):
        output = next(self.parameters()).new_zeros(len(observations), self.width)
        for modality in MODALITIES:
            rows = [index for index, value in enumerate(observations) if value.modality == modality]
            if rows:
                values = output.new_tensor([self.features(observations[index]) for index in rows])
                output = output.index_copy(0, torch.tensor(rows), self.adapters[modality](values))
        return output


class TypedReasoner(nn.Module):
    def __init__(self, width=48, memory_dim=8, programs=None):
        super().__init__()
        if type(width) is not int or not 8 <= width <= 256 or not 2 <= memory_dim <= 32:
            raise ValueError("Typed model dimensions exceed the declared bounds")
        self.width, self.memory_dim = width, memory_dim
        self.programs = copy.deepcopy(programs or {})
        from sera.typed_programs import validate_program
        for task, record in self.programs.items():
            validate_program(record)
            if record["task"] != task or task not in TASKS:
                raise ValueError("Typed program task mismatch")
        self.encoder = TypedEncoder(width)
        self.instruction = nn.Embedding(len(TASKS), width)
        self.memory = StatefulModel(ModelConfig(width=width, heads=2, memory_dim=memory_dim))
        self.memory.encoder = nn.Identity()
        self.memory.decoder = nn.Identity()
        self.categorical = nn.Linear(width, 4)
        self.numeric = nn.Linear(width, 2)

    def export_config(self):
        return {"type": "typed_reasoner", "width": self.width, "memory_dim": self.memory_dim,
                "event_schema": 1, "programs": copy.deepcopy(self.programs)}

    def validity(self):
        from sera.typed_programs import validate_program
        for task, record in self.programs.items():
            validate_program(record)
            if task != record["task"]:
                raise ValueError("Typed procedure task changed")
        return {"program_integrity_error": 0.0}

    def encode(self, observations, tasks):
        return self.encoder(observations) + self.instruction(torch.tensor([TASKS.index(task) for task in tasks]))

    def step(self, state, encoded, *, write):
        return self.memory.step(state, encoded, write=write)

    def read(self, hidden):
        return {"categorical": self.categorical(hidden), "numeric": self.numeric(hidden)}

    def forward(self, observations, tasks):
        if not observations or len(observations) != len(tasks) or any(not row for row in observations):
            raise ValueError("Typed inference needs nonempty observations and aligned task requests")
        state = self.memory.initial_state(len(tasks))
        hidden = next(self.parameters()).new_zeros(len(tasks), self.width)
        for position in range(max(map(len, observations))):
            rows = [index for index, values in enumerate(observations) if position < len(values)]
            indices = torch.tensor(rows)
            events = [observations[index][position] for index in rows]
            encoded = self.encode(events, [tasks[index] for index in rows])
            write = encoded.new_tensor([[float(not (tasks[index] == "binding" and position == len(observations[index]) - 1))]
                                         for index in rows])
            output, updated = self.step({key: value[indices] for key, value in state.items()}, encoded, write=write)
            state = {key: value.index_copy(0, indices, updated[key]) for key, value in state.items()}
            hidden = hidden.index_copy(0, indices, output)
        return self.read(hidden)

    @torch.no_grad()
    def predict(self, observations, task):
        self.eval()
        if task in self.programs:
            from sera.typed_programs import execute_rule
            try:
                answer = execute_rule(self.programs[task]["rule"], observations)
                return {"kind": EvidenceKind.PREDICTION.value, "task": task, "class": answer,
                        "procedure": self.programs[task]["identity"], "route": "verified-program"}
            except (ValueError, UnicodeError):
                pass
        output = self([observations], [task])
        if task == "motion":
            return {"kind": EvidenceKind.PREDICTION.value, "task": task, "values": output["numeric"][0].tolist(), "units": "m"}
        probabilities = output["categorical"][0].softmax(-1)
        return {"kind": EvidenceKind.PREDICTION.value, "task": task, "class": int(probabilities.argmax()),
                "probabilities": probabilities.tolist()}


def typed_examples(*, seed, count=64, split="support", family="ordinary"):
    if family not in {"ordinary", "extended", "structure"}:
        raise ValueError("Unknown typed generator family")
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("Invalid typed example count")
    rng, records = np.random.default_rng(seed), []
    for task in TASKS:
        for index in range(count):
            identity = f"{split}/{family}/{seed}/{task}/{index}"
            raw = Provenance("typed-observations", identity, EvidenceKind.OBSERVATION)
            events = []
            if task == "modular_sum":
                length = int(rng.integers(2, 5)) if family == "ordinary" else int(rng.integers(5, 8))
                numbers = rng.integers(4, size=length).tolist()
                events = [Observation("numeric", (float(value),), position, raw, units="dimensionless")
                          for position, value in enumerate(numbers)]
                target = sum(numbers) % 4
            elif task == "spatial_relation":
                limit = 2 if family == "ordinary" else 5
                first = rng.uniform(-limit, limit, size=2)
                target = int(rng.integers(4))
                displacement = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)][target]) * rng.uniform(.5, 2)
                for position, coordinate in enumerate((first, first + displacement)):
                    units = "cm" if rng.random() < .5 else "m"
                    values = tuple(float(value) / UNIT_FACTORS[units] for value in coordinate)
                    events.append(Observation("numeric", values, position, raw, units=units))
            elif task == "byte_sum":
                a, b = int(rng.integers(4)), int(rng.integers(4))
                spelling = f"{a}+{b}" if family == "ordinary" else f"{a} + {b}"
                events = [Observation("text_bytes", tuple(float(ord(c)) for c in spelling), 0, raw)]
                target = (a + b) % 4
            elif task == "patch_quadrant":
                target = int(rng.integers(4))
                x, y = target % 2, target // 2
                patch = rng.normal(0, .02 if family == "ordinary" else .08, size=(4, 4))
                patch[2 * y:2 * y + 2, 2 * x:2 * x + 2] += rng.uniform(.6, 1)
                if family == "structure":
                    patch[2 * y + int(rng.integers(2)), 2 * x + int(rng.integers(2))] = 0
                events = [Observation("image_patch", tuple(float(value) for value in patch.flatten()), 0, raw, units="pixel")]
            elif task == "tone":
                target = int(rng.integers(4))
                phase = rng.uniform(0, 2 * np.pi)
                amplitude = rng.uniform(.5, 1) if family == "ordinary" else rng.uniform(.25, .5)
                samples = amplitude * np.sin(2 * np.pi * (target + 1) * np.arange(16) / 16 + phase)
                samples += rng.normal(0, .01, size=16)
                events = [Observation("audio_frame", tuple(float(value) for value in samples), 0, raw, units="amplitude")]
            elif task == "motion":
                initial = rng.uniform(-1, 1, size=2)
                velocity = rng.uniform(-.2 if family == "ordinary" else -.4, .2 if family == "ordinary" else .4, size=2)
                events = [Observation("numeric", tuple(float(value) for value in initial + position * velocity),
                                      position, raw, units="m") for position in range(3)]
                target = tuple(float(value) for value in initial + 3 * velocity)
            else:
                values, length = {}, 6 if family == "ordinary" else 12
                for position in range(length):
                    key, value = int(rng.integers(4)), int(rng.integers(4))
                    values[key] = value
                    vector = [float(index == key) for index in range(4)] + [float(index == value) for index in range(4)]
                    events.append(Observation("symbolic", tuple(vector), position, raw))
                query = int(rng.choice(list(values)))
                events.append(Observation("symbolic", tuple([float(index == query) for index in range(4)] + [0.0] * 4),
                                          length, raw, available=(True,) * 4 + (False,) * 4))
                target = values[query]
            records.append(TypedExample(tuple(events), task, target,
                                         Provenance("typed-task-simulator", identity, EvidenceKind.SYNTHETIC), split))
    return records


@torch.no_grad()
def score_typed(model, records, *, work=None, use_programs=False, return_scores=False):
    if not records:
        raise ValueError("Typed evaluation requires examples")
    model.eval()
    results, all_scores = {}, {}
    for task in TASKS:
        selected = [row for row in records if row.task == task]
        if not selected:
            continue
        scores, losses, briers = [], [], []
        for start in range(0, len(selected), 64):
            batch = selected[start:start + 64]
            output = model([row.observations for row in batch], [task] * len(batch))
            if task == "motion":
                mse = (output["numeric"] - torch.tensor([row.target for row in batch])).square().mean(-1)
                scores.extend(torch.exp(-mse).tolist())
                losses.extend(mse.tolist())
            else:
                targets = torch.tensor([row.target for row in batch])
                probabilities = output["categorical"].softmax(-1)
                if use_programs and task in model.programs:
                    from sera.typed_programs import execute_rule
                    for index, row in enumerate(batch):
                        try:
                            answer = execute_rule(model.programs[task]["rule"], row.observations)
                            probabilities[index] = F.one_hot(torch.tensor(answer), 4).float()
                            if work is not None:
                                work.add("typed_program_evaluation_executions")
                        except (ValueError, UnicodeError):
                            pass
                scores.extend((probabilities.argmax(-1) == targets).float().tolist())
                losses.extend((-probabilities[torch.arange(len(batch)), targets].clamp_min(1e-12).log()).tolist())
                briers.extend((probabilities - F.one_hot(targets, 4)).square().sum(-1).tolist())
            if work is not None:
                work.add("typed_evaluation_events", sum(len(row.observations) for row in batch))
        results[task] = {"score": float(np.mean(scores)), "loss": float(np.mean(losses)),
                          "loss_units": "mean squared meters" if task == "motion" else "negative log likelihood",
                          "brier": float(np.mean(briers)) if briers else None, "examples": len(selected)}
        all_scores[task] = np.asarray(scores)
    report = {"tasks": results, "macro_score": float(np.mean([row["score"] for row in results.values()]))}
    return (report, all_scores) if return_scores else report


def fit_typed(model, evidence, *, validation, steps=1000, seed=0, work=None):
    if not isinstance(evidence, TypedEvidence) or not evidence.records or not validation or steps < 1:
        raise ValueError("Typed learning needs admitted support and separate validation")
    if evidence.identifiers.intersection(row.identifier for row in validation):
        raise ValueError("Typed support and validation overlap")
    if any(not row.split.startswith("validation") for row in validation):
        raise ValueError("Sealed typed evaluation cannot select a training checkpoint")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    best, best_state, history = -float("inf"), None, []
    for step in range(steps):
        rows = [evidence.records[index] for index in rng.integers(len(evidence.records), size=64)]
        model.train()
        output = model([row.observations for row in rows], [row.task for row in rows])
        categorical = [index for index, row in enumerate(rows) if row.task != "motion"]
        numeric = [index for index, row in enumerate(rows) if row.task == "motion"]
        loss = output["categorical"].sum() * 0
        if categorical:
            loss = loss + F.cross_entropy(output["categorical"][categorical], torch.tensor([rows[index].target for index in categorical]))
        if numeric:
            loss = loss + F.mse_loss(output["numeric"][numeric], torch.tensor([rows[index].target for index in numeric]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if work is not None:
            work.add("typed_optimizer_steps")
            work.add("typed_update_examples", len(rows))
            work.add("typed_update_events", sum(len(row.observations) for row in rows))
        if step % 100 == 0 or step == steps - 1:
            scored = score_typed(model, validation, work=work)
            history.append({"step": step + 1, "training_loss": float(loss.detach()), "validation": scored})
            if scored["macro_score"] > best:
                best, best_state = scored["macro_score"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"steps": steps, "examples": len(evidence.records), "validation_best_macro": best,
            "history": history, "support_ids": sorted(evidence.identifiers)}
