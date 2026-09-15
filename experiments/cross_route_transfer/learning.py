"""Matched optimizer paths; conditional targets never enter the evidence store."""

import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from sera.data import make_batch
from sera.shared_learning import old_loss
from sera.solver import Work
from sera.typed_learning import typed_examples

from .data import admit_training, state_digest, teacher_identity

ARMS = ("observed_only", "detached", "integrated", "extra_capacity", "oracle_exposure")


def anchor_bank(parent, seed):
    """Support-only inputs; targets explicitly remain frozen-model outputs."""
    rows = []
    for i, family in enumerate(("ordinary", "extended", "structure")):
        rows.extend(typed_examples(seed=seed+17*i, count=48, split="support-parent-anchor", family=family))
    with torch.no_grad():
        output = parent.forward_typed([r.observations for r in rows], [r.task for r in rows])
    return rows, {key: value.detach() for key, value in output.items()}


def retention_anchor(model, parent, bank, rng, step, count, seed, work):
    if step % 2 == 0:
        source, reference = bank
        indices = torch.tensor(rng.integers(0, len(source), count))
        rows = [source[i] for i in indices.tolist()]
        actual = model.forward_typed([r.observations for r in rows], [r.task for r in rows])
        categorical = torch.tensor([r.task != "motion" for r in rows])
        loss = actual["numeric"].sum()*0
        if categorical.any():
            loss = loss + F.kl_div(actual["categorical"][categorical].log_softmax(-1),
                                   reference["categorical"][indices][categorical].softmax(-1), reduction="batchmean")
        if (~categorical).any():
            loss = loss + F.mse_loss(actual["numeric"][~categorical], reference["numeric"][indices][~categorical])
        work.add("conditional_anchor_typed_draws", len(rows))
        work.add("conditional_anchor_typed_events", sum(len(r.observations) for r in rows))
        return loss
    batch = make_batch(count, 12 if step % 4 == 1 else 24, seed,
                       split="support-parent-anchor", index=step, task=(step//2) % 5)
    with torch.no_grad():
        target = parent.forward_sequence(batch.inputs).softmax(-1)
    work.add("conditional_anchor_sequence_draws", count)
    work.add("conditional_anchor_sequence_events", int(batch.inputs.shape[1])*count)
    work.add("conditional_anchor_teacher_sequence_events", int(batch.inputs.shape[1])*count)
    return F.kl_div(model.forward_sequence(batch.inputs).log_softmax(-1), target, reduction="batchmean")


def trainable_names(owner, mode):
    if mode == "readout":
        return [n for n, _ in owner.named_parameters() if n.startswith("typed_numeric.")]
    if mode != "shared":
        raise ValueError("Unknown trainable scope")
    prefixes = ("memory.", "fusion.", "typed_encoder.adapters.numeric.",
                "typed_instruction.", "typed_numeric.")
    # Exclude the unused legacy memory encoder/decoder and binding scope adapters.
    return [n for n, _ in owner.named_parameters() if n.startswith(prefixes)
            and not n.startswith(("memory.encoder.", "memory.decoder."))]


def configure(owner, mode):
    names = set(trainable_names(owner, mode))
    for name, value in owner.named_parameters():
        value.requires_grad_(name in names)
    return [p for p in owner.parameters() if p.requires_grad]


def motion_loss(owner, rows):
    predictions = owner.forward_typed([r.observations for r in rows], ["motion"]*len(rows))["numeric"]
    return F.mse_loss(predictions, torch.tensor([r.target for r in rows], dtype=torch.float32))


@torch.no_grad()
def motion_score(owner, rows):
    values = []
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        values.extend(owner.forward_typed([r.observations for r in batch], ["motion"]*len(batch))["numeric"].tolist())
    prediction = np.asarray(values)
    errors = np.square(prediction-np.asarray([r.target for r in rows])).mean(-1)
    return {"mse": float(errors.mean()), "rmse": float(np.sqrt(errors.mean())),
            "within_0_1": float((np.sqrt(errors) <= .1).mean()), "count": len(rows)}, {
                "predictions": prediction.tolist(), "mse": errors.tolist()}


def fit(parent, evidence, observed, dreams, oracle, development, cfg, arm, seed, checkpoint_callback=None, resume=None):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    teacher = teacher_identity(parent)
    corpus = observed if arm in {"observed_only", "extra_capacity"} else oracle if arm == "oracle_exposure" else dreams
    admit_training(observed, allow_conditional=False, teacher=teacher)
    admit_training(corpus, allow_conditional=arm in {"detached", "integrated"}, teacher=teacher)
    if any(not r.split.startswith("development-") for r in development):
        raise ValueError("Checkpoint selection requires development evidence")
    if {r.identity for r in corpus+observed}.intersection(r.identity for r in development):
        raise ValueError("Checkpoint-selection overlap")
    torch.manual_seed(seed)
    model = copy.deepcopy(parent)
    parameters = configure(model, cfg["scope"])
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0.0)
    draw_rng, replay_rng = np.random.default_rng(seed), np.random.default_rng(seed+871)
    work = Work()
    anchor_rng = np.random.default_rng(seed+1723)
    anchor = anchor_bank(parent, seed+2319) if cfg.get("anchor_weight", 0) else None
    if anchor is not None:
        work.add("conditional_anchor_teacher_typed_examples", len(anchor[0]))
        work.add("conditional_anchor_teacher_typed_events", sum(len(r.observations) for r in anchor[0]))
    initial = state_digest(model)
    history, best, best_state = [], float("inf"), None
    first_step = 0
    if resume is not None:
        if resume["training_config"] != cfg or resume["arm"] != arm or resume["seed"] != seed:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict({**parent.state_dict(), **resume["model_delta"]}, strict=True)
        best_state = {**copy.deepcopy(parent.state_dict()), **resume["best_delta"]}
        optimizer.load_state_dict(resume["optimizer"])
        first_step, history = resume["step"], copy.deepcopy(resume["history"])
        best = min(r["development"]["mse"] for r in history)
        draw_rng.bit_generator.state = resume["draw_rng"]
        replay_rng.bit_generator.state = resume["replay_rng"]
        anchor_rng.bit_generator.state = resume["anchor_rng"]
        torch.set_rng_state(resume["torch_rng"])
        work.counts = dict(resume["work"])
        if not 0 < first_step < cfg["steps"]:
            raise ValueError("Cannot restart a completed or invalid checkpoint")
    started = time.perf_counter()
    for step in range(first_step, cfg["steps"]):
        # Exactly the same index sequence, seed, batch shape, replay and optimizer
        # starts in all arms. Repeated observed rows do not count as new labels.
        selected = draw_rng.integers(0, 2**30, cfg["batch_size"])
        rows = [observed[int(i) % len(observed)] if j < cfg["batch_size"]//2
                else corpus[int(i) % len(corpus)] for j, i in enumerate(selected)]
        model.train()
        loss = motion_loss(model, rows)
        replay = old_loss(model, evidence, ("typed", "world", "sequence")[step % 3],
                          replay_rng, cfg["replay_batch"], 3800000+step, work)
        objective = loss + cfg["replay_weight"]*replay
        anchored = loss.new_zeros(())
        if anchor is not None:
            anchored = retention_anchor(model, parent, anchor, anchor_rng, step, cfg["anchor_batch"], seed+3559, work)
            objective = objective + cfg["anchor_weight"]*anchored
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        work.add("motion_example_draws", len(rows))
        work.add("motion_event_evaluations", 3*len(rows))
        work.add("optimizer_updates")
        if (step+1) % cfg["select_every"] == 0 or step == cfg["steps"]-1:
            model.eval()
            metric, _ = motion_score(model, development)
            history.append({"step": step+1, "training_mse": float(loss.detach()),
                            "replay_loss": float(replay.detach()), "anchor_loss": float(anchored.detach()), "development": metric})
            if metric["mse"] < best:
                best, best_state = metric["mse"], copy.deepcopy(model.state_dict())
            if checkpoint_callback:
                checkpoint_callback(step+1, model, optimizer, best_state, history,
                                    draw_rng.bit_generator.state, replay_rng.bit_generator.state,
                                    anchor_rng.bit_generator.state, dict(work.counts))
    model.load_state_dict(best_state)
    model.eval()
    changed = [n for n, x in model.state_dict().items() if not torch.equal(x, parent.state_dict()[n])]
    if not changed:
        raise ValueError("No numerical learning occurred")
    if any(n.startswith("generator_") for n in changed):
        raise ValueError("Frozen teacher workspace changed during distillation")
    return model, {"arm": arm, "initial_state_sha256": initial,
                   "state_sha256": state_digest(model), "history": history,
                   "best_development_mse": best, "training_seconds": time.perf_counter()-started,
                   "trainable_parameters": sum(p.numel() for p in parameters),
                   "owner_parameters": sum(p.numel() for p in parent.parameters()),
                   "extra_owner_parameters": sum(p.numel() for p in parent.parameters())
                   if arm in {"detached", "extra_capacity"} else 0,
                   "changed_tensors": changed, "work": dict(work.counts),
                   "new_observed_worlds": len(observed)+(len(oracle) if arm == "oracle_exposure" else 0),
                   "conditional_teacher_worlds": len(dreams) if arm in {"detached", "integrated"} else 0,
                   "unique_supervision_worlds": len(observed)+
                   (len(corpus) if arm not in {"observed_only", "extra_capacity"} else 0),
                   "parameter_scope": cfg["scope"], "optimizer_starts": 1,
                   "claim_boundary": "Update counts/event shapes matched; measured wall time and unique exposure are separate, not assumed equal."}
