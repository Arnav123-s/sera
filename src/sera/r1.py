"""Connected recurrent world prediction, imagined planning and learning from admitted feedback."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.contracts import EvidenceKind, Provenance, StateOwner
from sera.data import seed_for
from sera.experience import EvidenceReplay
from sera.models import ModelConfig, StatefulModel


def context_vector(identifier):
    # Public environment identity, not its transition table or hidden physical state.
    raw = hashlib.sha256(identifier.encode()).digest()[:8]
    return torch.tensor([(x / 127.5) - 1 for x in raw], dtype=torch.float32)


class RecurrentWorldModel(nn.Module):
    def __init__(self, width=48, heads=2, memory_dim=8, kind="delta", planning_horizon=3,
                 routing="all", density_rank=None, density_dimension=None, density_count=None,
                 adapter_rank=0):
        super().__init__()
        self.settings = dict(width=width, heads=heads, memory_dim=memory_dim, kind=kind)
        self.planning_horizon = planning_horizon
        if kind in {"reference", "lowrank_hybrid"}:
            from sera.lowrank import ReferenceMemory
            options = dict(routing=routing, density_rank=density_rank,
                           density_dimension=density_dimension, density_count=density_count)
            self.memory = ReferenceMemory(width, heads, memory_dim, compact=kind == "lowrank_hybrid", **options)
            self.settings.update({key: value for key, value in options.items()
                                  if value is not None and (key != "routing" or value != "all")})
        else:
            self.memory = StatefulModel(ModelConfig(**self.settings))
            self.memory.encoder = nn.Sequential(nn.Linear(22, width), nn.Tanh())
            self.memory.decoder = nn.Identity()
        self.transition = nn.Sequential(nn.Linear(width + 4, width * 2), nn.Tanh(),
                                        nn.Linear(width * 2, 4))
        self.fusion = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.reward = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh(), nn.Linear(width, 1))
        self.adapter = None
        if adapter_rank:
            if type(adapter_rank) is not int or not 1 <= adapter_rank <= width:
                raise ValueError("Invalid residual adapter rank")
            self.adapter = nn.Sequential(nn.Linear(width, adapter_rank, bias=False), nn.Tanh(),
                                          nn.Linear(adapter_rank, width, bias=False))
            nn.init.zeros_(self.adapter[-1].weight)
            self.settings["adapter_rank"] = adapter_rank

    def export_config(self):
        return {"type": "r1", **self.settings, "planning_horizon": self.planning_horizon}

    def initial(self, batch):
        return self.memory.initial_state(batch)

    def observe(self, state, colors, previous_actions, rewards, goals, contexts):
        # A missing color is zeros plus an explicit availability flag; it is not a hidden ID.
        visible = (colors >= 0).float()[:, None]
        sensor = F.one_hot(colors.clamp_min(0), 4).float() * visible
        action = F.one_hot(previous_actions.clamp_min(0), 4).float()
        action = action * (previous_actions >= 0).float()[:, None]
        features = torch.cat((sensor, action, rewards[:, None], F.one_hot(goals, 4),
                              contexts, visible), -1)
        encoded = self.memory.encoder(features)
        hidden, next_state = self.memory.step(state, encoded, write=torch.ones(len(colors), 1))
        fused = self.fusion(torch.cat((encoded, hidden), -1))
        return (fused if self.adapter is None else fused + self.adapter(fused)), next_state

    def predict(self, hidden, actions, goals):
        condition = torch.cat((hidden, F.one_hot(actions, 4)), -1)
        return self.transition(condition), self.reward(
            torch.cat((condition, F.one_hot(goals, 4)), -1)
        ).squeeze(-1)

    def forward(self, observations, actions, rewards, goals, contexts, *, reset_memory=False):
        state = self.initial(len(observations))
        previous = torch.full((len(observations),), -1, dtype=torch.long)
        received = torch.zeros(len(observations))
        predictions, reward_predictions = [], []
        for t in range(actions.shape[1]):
            if reset_memory:
                state = self.initial(len(observations))
            hidden, state = self.observe(state, observations[:, t], previous, received, goals, contexts)
            logits, reward = self.predict(hidden, actions[:, t], goals)
            predictions.append(logits)
            reward_predictions.append(reward)
            previous, received = actions[:, t], rewards[:, t]
        return torch.stack(predictions, 1), torch.stack(reward_predictions, 1)


def tensors(records):
    if not records:
        raise ValueError("A batch needs admitted trajectories")
    length = max(len(row.actions) for row in records)
    return (torch.tensor([r.observations + (-1,) * (length - len(r.actions)) for r in records]),
            torch.tensor([r.actions + (0,) * (length - len(r.actions)) for r in records]),
            torch.tensor([r.rewards + (0.0,) * (length - len(r.actions)) for r in records]),
            torch.tensor([r.goal for r in records]),
            torch.stack([context_vector(r.world_id) for r in records]))


def fit(model, evidence: EvidenceReplay, *, steps=200, batch_size=32, seed=0,
        learning_rate=0.003, replay=None, work=None, update_mode="all"):
    if not isinstance(evidence, EvidenceReplay) or not evidence.records or steps < 1:
        raise ValueError("Learning requires nonempty admitted evidence and a finite update budget")
    rng = np.random.default_rng(seed_for("r1-update-order", seed))
    if update_mode not in {"all", "adapter"} or (update_mode == "adapter" and model.adapter is None):
        raise ValueError("Adapter-only learning requires an explicit adapter")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(update_mode == "all" or name.startswith("adapter."))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=1e-4)
    losses = []
    model.train()
    for _ in range(steps):
        novel_n = batch_size if replay is None or not replay.records else batch_size // 2
        records = [evidence.records[i] for i in rng.integers(len(evidence.records), size=novel_n)]
        if novel_n != batch_size:
            records += [replay.records[i] for i in rng.integers(len(replay.records),
                                                              size=batch_size - novel_n)]
        obs, actions, rewards, goals, contexts = tensors(records)
        logits, predicted_rewards = model(obs, actions, rewards, goals, contexts)
        targets = obs[:, 1:]
        valid = targets >= 0
        actual_steps = torch.arange(actions.shape[1])[None] < torch.tensor([len(r.actions) for r in records])[:, None]
        prediction_loss = F.cross_entropy(logits[valid], targets[valid]) if valid.any() else logits.sum() * 0
        reward_loss = F.binary_cross_entropy_with_logits(predicted_rewards, rewards, reduction="none")[actual_steps].mean()
        loss = prediction_loss + 0.5 * reward_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite connected world-model loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if work is not None:
            work.add("optimizer_steps")
            work.add("update_trajectory_draws", len(records))
            work.add("update_transition_draws", sum(len(r.actions) for r in records))
            work.add("padded_transition_evaluations", actions.numel() - sum(len(r.actions) for r in records))
            work.add("novel_trajectory_draws", novel_n)
            work.add("replay_trajectory_draws", batch_size - novel_n)
    model.eval()
    return {"initial_batch_loss": losses[0], "final_batch_loss": losses[-1],
            "mean_last_20_loss": float(np.mean(losses[-20:])), "steps": steps,
            "admitted_examples": len(evidence.records), "history": losses}


@torch.no_grad()
def score(model, records, truth, *, reset_memory=False, work=None):
    model.eval()
    correct, nll, rewards, brier, entropy = [], [], [], [], []
    for start in range(0, len(records), 128):
        batch = records[start:start + 128]
        inputs = tensors(batch)
        logits, reward = model(*inputs, reset_memory=reset_memory)
        if work is not None:
            work.add("evaluation_transition_predictions", inputs[1].numel())
            work.add("evaluation_reward_predictions", inputs[1].numel())
        targets = torch.tensor(truth[start:start + len(batch), 1:])
        p = logits.softmax(-1)
        correct.extend((p.argmax(-1) == targets).float().mean(-1).tolist())
        nll.extend((-p.gather(-1, targets[..., None]).clamp_min(1e-12).log()).mean((1, 2)).tolist())
        brier.extend((p - F.one_hot(targets, 4)).square().sum(-1).mean(-1).tolist())
        rewards.extend((reward.sigmoid() - inputs[2]).square().mean(-1).tolist())
        entropy.extend((-(p * p.clamp_min(1e-12).log()).sum(-1)).mean(-1).tolist())
    return {"accuracy": float(np.mean(correct)), "nll": float(np.mean(nll)),
            "brier": float(np.mean(brier)), "reward_mse": float(np.mean(rewards)),
            "entropy": float(np.mean(entropy)), "episodes": len(records)}, np.asarray(correct)


class WorldSession:
    schema_version = 1

    def __init__(self, model, world_id, goal, initial_observation, *, owner=None,
                 model_version=None, encoder_version="world-symbolic-v1", metadata=None,
                 memory_references=(), admitted_provenance=()):
        from sera.session_state import model_identity
        if not isinstance(world_id, str) or not world_id or type(goal) is not int or not 0 <= goal < 4:
            raise ValueError("Session needs a world identifier and a valid goal")
        self.model = model.eval()
        self.model_sha256 = model_identity(model)
        self.model_version = model_version or self.model_sha256
        self.encoder_version = encoder_version
        self.owner = StateOwner(owner or str(uuid.uuid4()), self.model_version, encoder_version)
        self.owner.validate()
        if not encoder_version:
            raise ValueError("Session encoder version is required")
        self.metadata = copy.deepcopy(metadata or {"modality": "symbolic", "sensor": "four-colors", "missing": -1})
        if not isinstance(self.metadata, dict):
            raise ValueError("Session modality metadata must be a mapping")
        self.memory_references = list(memory_references)
        self.admitted_provenance = []
        for provenance in admitted_provenance:
            self.admit_provenance(provenance)
        self.events_seen = 0
        self.diagnostics = []
        self.world_id, self.goal = world_id, goal
        self.context = context_vector(world_id)[None]
        self.state = model.initial(1)
        self.hidden = None
        self.observe(initial_observation, -1, 0.0)

    def admit_provenance(self, provenance):
        from dataclasses import asdict
        if not isinstance(provenance, Provenance) or provenance.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}:
            raise ValueError("Session admitted provenance requires verified or synthetic evidence")
        row = asdict(provenance)
        if row not in self.admitted_provenance:
            self.admitted_provenance.append(row)

    def _validate_model(self):
        from sera.session_state import model_identity
        if model_identity(self.model) != self.model_sha256:
            raise ValueError("Live session model changed; create or explicitly migrate a session")

    def save(self, path):
        from sera.session_state import pack_tensors
        from sera.storage import digest, write_json
        self._validate_model()
        payload = {"schema_version": self.schema_version, "owner": self.owner.validate(),
                   "model_sha256": self.model_sha256, "world_id": self.world_id, "goal": self.goal,
                   "metadata": self.metadata, "memory_references": self.memory_references,
                   "admitted_provenance": self.admitted_provenance, "events_seen": self.events_seen,
                   "diagnostics": self.diagnostics,
                   "state": pack_tensors(self.state), "hidden": pack_tensors(self.hidden)}
        write_json(Path(path), {"payload": payload, "sha256": digest(payload)})
        return {"owner": self.owner.validate(), "events_seen": self.events_seen, "sha256": digest(payload)}

    @classmethod
    def load(cls, path, model, *, owner, model_version=None, encoder_version="world-symbolic-v1"):
        from sera.session_state import model_identity, unpack_tensors
        from sera.storage import digest
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Session file exceeds the declared serialization budget")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if digest(payload) != envelope["sha256"] or payload["schema_version"] != cls.schema_version:
            raise ValueError("Session integrity or schema mismatch")
        identity = model_identity(model)
        expected = StateOwner(owner, model_version or identity, encoder_version).validate()
        if payload["owner"] != expected or payload["model_sha256"] != identity:
            raise ValueError("Session ownership, model or encoder version mismatch")
        result = cls(model, payload["world_id"], payload["goal"], -1, owner=owner,
                     model_version=expected["model_version"], encoder_version=encoder_version,
                     metadata=payload["metadata"], memory_references=payload["memory_references"])
        result.state = unpack_tensors(payload["state"], model.initial(1))
        if model.settings["kind"] in {"reference", "lowrank_hybrid"}:
            trace = result.state["density"].abs().square().sum((-2, -1))
            if not torch.allclose(trace, torch.ones_like(trace), atol=1e-4, rtol=0):
                raise ValueError("Session density factors violate unit trace")
        result.hidden = unpack_tensors(payload["hidden"], result.hidden)
        if type(payload["events_seen"]) is not int or payload["events_seen"] < 1:
            raise ValueError("Invalid session event count")
        result.events_seen = payload["events_seen"]
        result.diagnostics = payload.get("diagnostics", [])
        if not isinstance(result.diagnostics, list) or len(result.diagnostics) > 4096:
            raise ValueError("Invalid session diagnostic history")
        for row in payload["admitted_provenance"]:
            result.admit_provenance(Provenance(row["source"], row["record_id"], EvidenceKind(row["kind"])))
        return result

    @torch.no_grad()
    def observe(self, color, action, reward):
        self._validate_model()
        if len(self.diagnostics) >= 4096:
            raise ValueError("Flush session diagnostics before observing more events")
        if (type(color) is not int or not -1 <= color < 4 or type(action) is not int
                or not -1 <= action < 4 or not math.isfinite(reward) or not 0 <= reward <= 1):
            raise ValueError("Invalid observed sensor, action or reward")
        self.hidden, self.state = self.model.observe(
            self.state, torch.tensor([color]), torch.tensor([action]), torch.tensor([reward]),
            torch.tensor([self.goal]), self.context
        )
        self.events_seen += 1
        diagnostic = getattr(self.model.memory, "last_step_diagnostics", None)
        if diagnostic is not None:
            self.diagnostics.append({"event": self.events_seen, **copy.deepcopy(diagnostic)})

    def flush_diagnostics(self, path):
        from sera.storage import digest, write_json
        record = {"owner": self.owner.validate(), "model_sha256": self.model_sha256,
                  "events": self.diagnostics}
        write_json(Path(path), record)
        self.memory_references.append({"kind": "diagnostic-history", "path": str(Path(path)), "sha256": digest(record)})
        self.diagnostics = []
        return self.memory_references[-1]

    @torch.no_grad()
    def plan(self, *, horizon=None, beam=8, work=None):
        self._validate_model()
        horizon = self.model.planning_horizon if horizon is None else horizon
        if not 1 <= horizon <= 8 or not 1 <= beam <= 64:
            raise ValueError("Invalid planning bounds")
        pending = [(self.state, self.hidden, (), 0.0)]
        best = None
        for depth in range(horizon):
            expanded = []
            for state, hidden, prefix, _ in pending:
                for action in range(4):
                    logits, reward = self.model.predict(hidden, torch.tensor([action]),
                                                        torch.tensor([self.goal]))
                    color = logits.argmax(-1)
                    probability = float(logits.softmax(-1)[0, self.goal])
                    value = (probability + float(reward.sigmoid()[0])) / 2 - 0.035 * depth
                    actions = prefix + (action,)
                    if best is None or value > best[0]:
                        best = (value, actions)
                    next_hidden, next_state = self.model.observe(
                        state, color, torch.tensor([action]), reward.sigmoid(),
                        torch.tensor([self.goal]), self.context
                    )
                    expanded.append((next_state, next_hidden, actions, value))
                    if work is not None:
                        work.add("planning_model_transitions")
            pending = sorted(expanded, key=lambda row: row[-1], reverse=True)[:beam]
        return best[1]


def updated(model, evidence, **kwargs):
    candidate = copy.deepcopy(model)
    record = fit(candidate, evidence, **kwargs)
    return candidate, record
