"""Joint pretraining and bounded retained updates of one shared R1 owner."""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sera.binding import binding_cases
from sera.contracts import EvidenceKind, Observation, Provenance
from sera.data import make_batch, seed_for
from sera.environments import collect, make_world
from sera.evaluation import evaluate
from sera.experience import EvidenceReplay
from sera.r1 import score, tensors
from sera.shared import SharedR1, SharedTypedView, make_shared_solver
from sera.storage import canonical, digest
from sera.typed_learning import TASKS, TypedEvidence, TypedExample, score_typed
from sera.typed_protocol import audit_partitions, semantic_id, typed_suite


@dataclass
class SharedEvidence:
    typed: TypedEvidence
    world: EvidenceReplay
    sequence_seed: int

    def validate(self):
        if not self.typed.records or not self.world.records:
            raise ValueError("Shared training needs typed and observed-world evidence")
        if (any(not r.split.startswith(("support", "meta-train")) for r in self.typed.records)
                or any(not r.split.startswith(("support", "meta-train", "extra-support", "program-support", "planned-support", "active-support"))
                       for r in self.world.records)):
            raise ValueError("Shared updates cannot consume validation or sealed evidence")

    def save(self, root):
        self.validate()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.world.save(root / "world.json")
        payload = {"schema_version": 1, "sequence_seed": self.sequence_seed,
                   "typed": [asdict(row) for row in self.typed.records]}
        payload["dataset_id"] = digest(payload)
        (root / "typed.json.gz").write_bytes(gzip.compress(canonical(payload).encode(), mtime=0))

    @classmethod
    def load(cls, root):
        root = Path(root)
        payload = json.loads(gzip.decompress((root / "typed.json.gz").read_bytes()))
        expected = payload.pop("dataset_id")
        if payload["schema_version"] != 1 or digest(payload) != expected:
            raise ValueError("Shared evidence integrity failure")
        evidence = cls(TypedEvidence([restore_typed(row) for row in payload["typed"]]),
                       EvidenceReplay.load(root / "world.json"), payload["sequence_seed"])
        evidence.validate()
        return evidence

    def admit_typed(self, rows):
        proposed = SharedEvidence(TypedEvidence([*self.typed.records, *rows]), self.world, self.sequence_seed)
        proposed.validate()
        seen, kept = {}, []
        for row in proposed.typed.records:
            key = semantic_id(row)
            if key in seen and seen[key] != row.target:
                raise ValueError("Conflicting corrective targets for the same observed input")
            if key not in seen:
                seen[key] = row.target
                kept.append(row)
        self.typed = TypedEvidence(kept)


def restore_typed(row):
    def provenance(value):
        return Provenance(value["source"], value["record_id"], EvidenceKind(value["kind"]))
    observations = []
    for raw in row["observations"]:
        values = dict(raw)
        values["values"] = tuple(values["values"])
        values["available"] = tuple(values["available"]) if values.get("available") is not None else None
        values["provenance"] = provenance(values["provenance"])
        observations.append(Observation(**values))
    return TypedExample(tuple(observations), row["task"], tuple(row["target"]) if row["task"] == "motion" else row["target"],
                        provenance(row["evidence"]), row["split"])


def development(seed):
    suite, manifest = typed_suite(seed=910000 + seed, test_count=128)
    latest = binding_cases(seed=920000 + seed, count=512, rule="latest")
    latest_validation = binding_cases(seed=930000 + seed, count=128, rule="latest", split="validation")
    spec = make_world(940000 + seed, family="permutation")
    worlds, _ = collect(spec, seed=seed, count=256, length=8, mask_rate=.25, split="support-shared")
    world_validation, truth = collect(spec, seed=950000 + seed, count=64, length=12, mask_rate=.4, split="validation-shared")
    evidence = SharedEvidence(TypedEvidence(suite["support"] + latest), EvidenceReplay(worlds), 960000 + seed)
    evidence.validate()
    audit_partitions({"support": evidence.typed.records, "validation": suite["validation"] + latest_validation})
    validation = {"typed": suite["validation"], "latest": latest_validation, "world": world_validation,
                  "truth": truth, "sequence_seed": 970000 + seed}
    return evidence, validation, spec, manifest


def typed_loss(model, rows, work=None):
    output = model.forward_typed([r.observations for r in rows], [r.task for r in rows])
    categorical = [i for i, r in enumerate(rows) if r.task != "motion"]
    numeric = [i for i, r in enumerate(rows) if r.task == "motion"]
    loss = output["categorical"].sum() * 0
    if categorical:
        loss = loss + F.cross_entropy(output["categorical"][categorical], torch.tensor([rows[i].target for i in categorical]))
    if numeric:
        loss = loss + F.mse_loss(output["numeric"][numeric], torch.tensor([rows[i].target for i in numeric]))
    if work is not None:
        work.add("shared_typed_example_draws", len(rows))
        work.add("shared_typed_event_evaluations", sum(len(r.observations) for r in rows))
    return loss


def world_loss(model, rows, work=None):
    obs, actions, rewards, goals, contexts = tensors(rows)
    logits, predicted = model(obs, actions, rewards, goals, contexts)
    targets = obs[:, 1:]
    visible = targets >= 0
    actual = torch.arange(actions.shape[1])[None] < torch.tensor([len(r.actions) for r in rows])[:, None]
    loss = F.cross_entropy(logits[visible], targets[visible]) if visible.any() else logits.sum() * 0
    loss = loss + .5 * F.binary_cross_entropy_with_logits(predicted, rewards, reduction="none")[actual].mean()
    if work is not None:
        work.add("shared_world_trajectory_draws", len(rows))
        work.add("shared_world_event_evaluations", int(actual.sum()))
    return loss


def old_loss(model, evidence, domain, rng, batch_size, index, work=None):
    if domain == "typed":
        # Each typed task has equal draw probability despite extra latest-binding evidence.
        grouped = {task: [r for r in evidence.typed.records if r.task == task] for task in TASKS}
        rows = []
        for task in rng.choice(TASKS, size=batch_size):
            pool = grouped[task]
            rows.append(pool[int(rng.integers(len(pool)))])
        return typed_loss(model, rows, work)
    if domain == "world":
        rows = [evidence.world.records[i] for i in rng.integers(len(evidence.world.records), size=batch_size)]
        return world_loss(model, rows, work)
    if domain != "sequence":
        raise ValueError("Unknown shared training stream")
    batch = make_batch(batch_size, 12, evidence.sequence_seed, split="support-shared-sequence", index=index)
    if work is not None:
        work.add("shared_sequence_draws", batch_size)
        work.add("shared_sequence_event_evaluations", batch_size * 12)
    return F.cross_entropy(model.forward_sequence(batch.inputs), batch.targets)


@torch.no_grad()
def validate_shared(model, validation, work=None):
    model.eval()
    typed = score_typed(SharedTypedView(model), validation["typed"], work=work)
    latest = score_typed(SharedTypedView(model), validation["latest"], work=work)["tasks"]["binding"]
    world, _ = score(model, validation["world"], validation["truth"], work=work)
    sequence, _ = evaluate(make_shared_solver(model), seed=validation["sequence_seed"], split="validation-shared-sequence", samples=128)
    if work is not None:
        work.add("shared_sequence_validation_examples", 512)
    return {"macro": float(np.mean([typed["macro_score"], latest["score"], world["accuracy"], sequence["macro_accuracy"]])),
            "typed": typed, "latest": latest, "world": world, "sequence": sequence}


def pretrain_shared(model, evidence, validation, *, seed, steps=1600, batch_size=32, learning_rate=.003, work=None):
    evidence.validate()
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 2:
        raise ValueError("Shared training needs a finite positive update budget")
    rng = np.random.default_rng(seed_for("shared-pretrain-order", seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best, best_state, history = -float("inf"), None, []
    for step in range(steps):
        model.train()
        domain = ("world", "sequence", "typed", "typed")[step % 4]
        loss = old_loss(model, evidence, domain, rng, batch_size, step, work)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if work is not None:
            work.add("shared_pretraining_updates")
        if step % 100 == 0 or step == steps-1:
            report = validate_shared(model, validation, work)
            history.append({"step": step+1, "training_loss": float(loss.detach()), "validation": report})
            if report["macro"] > best:
                best, best_state = report["macro"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
            "best_validation_macro": best, "history": history,
            "typed_support_cases": len(evidence.typed.records), "world_support_trajectories": len(evidence.world.records),
            "typed_semantic_sha256": digest(sorted(semantic_id(r) for r in evidence.typed.records)),
            "sequence_seed": evidence.sequence_seed, "schedule": ["world", "sequence", "typed", "typed"]}


def adapt_shared(parent, evidence, support, validation, *, method, seed, steps=256, batch_size=32, work=None):
    evidence.validate()
    if method not in {"none", "full", "replay", "adapter", "scratch", "scoped"}:
        raise ValueError("Unknown shared-core adaptation control")
    admitted = TypedEvidence(support)
    if not admitted.records or any(not r.split.startswith("support") for r in admitted.records):
        raise ValueError("Corrective updates require support evidence")
    if not validation or any(not r.split.startswith("validation") for r in validation):
        raise ValueError("Adaptation checkpoint selection requires separate validation")
    audit_partitions({"support": support, "validation": validation})
    if type(steps) is not int or steps < 1 or batch_size < 2:
        raise ValueError("Invalid adaptation budget")
    candidate = copy.deepcopy(parent)
    if method == "none":
        return candidate, {"method": method, "steps": 0, "support_cases": len(support)}
    if method == "scratch":
        torch.manual_seed(seed_for("shared-scratch", seed))
        settings = dict(parent.export_config())
        settings.pop("type")
        settings.pop("programs")
        candidate = SharedR1(**settings)
    if method == "adapter":
        torch.manual_seed(seed_for("shared-adapter", seed))
        candidate.add_adapter()
    if method == "scoped" and candidate.scope_adapter is None:
        torch.manual_seed(seed_for("shared-scoped-adapter", seed))
        candidate.add_scoped_adapter()
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(name.startswith("scope_adapter.") if method == "scoped" else
                                 method != "adapter" or name.startswith("adapter."))
    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(seed_for("shared-adaptation-order", seed))
    history, best, best_state = [], -float("inf"), None
    for step in range(steps):
        candidate.train()
        count = batch_size // 2 if method == "replay" else batch_size
        rows = [admitted.records[i] for i in rng.integers(len(admitted.records), size=count)]
        loss = typed_loss(candidate, rows, work)
        if method == "replay":
            retained = old_loss(candidate, evidence, ("typed", "world", "sequence")[step % 3],
                                 rng, batch_size-count, 100000 + step, work)
            loss = .5 * loss + .5 * retained
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        optimizer.step()
        if work is not None:
            work.add("shared_adaptation_updates")
            work.add("shared_novel_example_draws", count)
            work.add("shared_replay_example_draws", batch_size-count)
        if step % 64 == 0 or step == steps-1:
            report = score_typed(SharedTypedView(candidate), validation, work=work)["tasks"]["binding"]
            history.append({"step": step+1, "training_loss": float(loss.detach()), "validation": report})
            if report["score"] > best:
                best, best_state = report["score"], copy.deepcopy(candidate.state_dict())
    candidate.load_state_dict(best_state)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    candidate.eval()
    return candidate, {"method": method, "steps": steps, "batch_size": batch_size, "support_cases": len(support),
                       "history": history, "best_validation_score": best,
                       "support_semantic_sha256": digest(sorted(semantic_id(r) for r in support)),
                       "validation_semantic_sha256": digest(sorted(semantic_id(r) for r in validation)),
                       "novel_examples_per_update": batch_size//2 if method == "replay" else batch_size,
                       "replay_examples_per_update": batch_size//2 if method == "replay" else 0}
