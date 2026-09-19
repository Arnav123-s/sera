"""The owner-connected language descendant.

The draft this replaces built a fresh `ContinuationLanguageAdapter`, optimised
only that adapter, and saved it beside unchanged owner subjects. Nothing in its
forward pass touched the restored learner, so no amount of training could reach
the owner's routes (review finding 1).

What is done here instead:

* the descendant is the *same object* the owner's routes hold. `attach()`
  converts its class in place, exactly as every earlier stage does;
* the learned language signal reaches the owner through
  `SharedR1.add_adapter()`, the residual adapter that lives inside
  `shared_step`. Every route that consults the shared recurrent core —
  the typed route, the sequence route, the world route, the request route,
  the reading and book routes — passes through it, so a change there is a
  change to the owner's behaviour, not to a side-car;
* the adapter's output projection is zero-initialised, so at step 0 the
  descendant reproduces the parent exactly and every later difference is
  attributable to teaching;
* every inherited tensor is frozen. The trainable set is exactly the new
  language modules plus the adapter, and `trainable_report()` states it.

What is protected, and why it is protected by construction: the exact rational
mathematics routes, the fitted physical-field weights and the fitted
intervention coefficients are evaluated by their own code paths and never read
the shared recurrent core, so no value of the adapter can move them. That is a
claim about the code, and Stage 48 also measures it by replaying the retained
task suite through the trained descendant.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.intervention_model import InterventionR1
from sera.session_state import model_identity

LAB_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("descendant.py", "corpus.py", "training.py")
LANGUAGE_PREFIXES = ("lex_", "adapter.")


def source_digest():
    folder = Path(__file__).parent
    return hashlib.sha256(json.dumps(
        {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
         for name in SOURCE_FILES if (folder / name).is_file()}, sort_keys=True).encode()).hexdigest()


class LanguageAcquisitionR1(InterventionR1):
    """The Stage 47 owner, extended in place with a connected language route."""

    @classmethod
    def attach(cls, owner, vocabulary, *, embedding=64, adapter_rank=32, seed=4801):
        if type(owner) is not InterventionR1:
            raise ValueError("Attach to the restored Stage 47 intervention owner")
        if not isinstance(vocabulary, dict) or "words" not in vocabulary:
            raise ValueError("A frozen vocabulary record is required")
        size = len(vocabulary["words"])
        if not 64 <= size <= 65536:
            raise ValueError("Vocabulary outside the declared scope")
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        generator = torch.Generator().manual_seed(seed)
        width = owner.settings["width"]
        owner.__class__ = cls
        owner.lex_embedding = nn.Embedding(size, embedding, padding_idx=0)
        owner.lex_projection = nn.Linear(embedding, width)
        owner.lex_norm = nn.LayerNorm(width)
        owner.lex_head = nn.Linear(width, size)
        with torch.no_grad():
            owner.lex_embedding.weight.copy_(torch.randn(size, embedding, generator=generator) * .05)
            owner.lex_embedding.weight[0].zero_()
            owner.lex_projection.weight.copy_(torch.randn(width, embedding, generator=generator) * (embedding ** -.5))
            owner.lex_projection.bias.zero_()
            owner.lex_head.weight.copy_(torch.randn(size, width, generator=generator) * (width ** -.5))
            owner.lex_head.bias.zero_()
        owner.add_adapter(adapter_rank)
        with torch.no_grad():
            owner.adapter[0].weight.copy_(torch.randn(adapter_rank, width, generator=generator) * (width ** -.5))
            owner.adapter[-1].weight.zero_()
        owner.language_config = {
            "schema": "sera.owner-language.descendant.1", "seed": seed, "embedding": embedding,
            "adapter_rank": adapter_rank, "vocabulary_size": size,
            "vocabulary_digest": vocabulary["digest"], "source": source_digest(),
            "connection": "residual adapter inside SharedR1.shared_step, shared by every recurrent route",
            "adapter_output_zero_initialised": True}
        for name, parameter in owner.named_parameters():
            parameter.requires_grad_(name.startswith(LANGUAGE_PREFIXES))
        return owner

    def export_config(self):
        return {**super().export_config(), "language_acquisition": copy.deepcopy(self.language_config)}

    # ---- the connected language route -------------------------------------

    def lex_encode(self, ids):
        return self.lex_projection(self.lex_embedding(ids))

    def lex_hidden(self, ids):
        """Run tokens through the owner's own recurrent core, with gradient."""
        if ids.ndim != 2 or not ids.numel():
            raise ValueError("Token identifiers must be a non-empty batch of sequences")
        mask = ids.ne(0)
        if not bool(mask[:, 0].all()):
            raise ValueError("Every sequence must start with a real token; pad on the right only")
        encoded = self.lex_encode(ids)
        state = self.initial(len(ids))
        outputs = []
        for position in range(ids.shape[1]):
            valid = mask[:, position]
            value, updated = self.shared_step(state, encoded[:, position], valid.float()[:, None])
            state = {key: torch.where(valid.reshape(-1, *([1] * (tensor.ndim - 1))), updated[key], tensor)
                     for key, tensor in state.items()}
            outputs.append(value)
        return self.lex_norm(torch.stack(outputs, 1)), mask

    def lex_logits(self, ids):
        hidden, mask = self.lex_hidden(ids)
        return self.lex_head(hidden), mask

    def lex_next_token_loss(self, ids):
        """Masked next-token loss over ragged, right-padded batches.

        Positions whose target is padding are excluded by ``ignore_index``, so an
        uneven final chunk is ordinary rather than an exception, and the reported
        token count is the number of positions that actually contributed.
        """
        logits, _ = self.lex_logits(ids)
        predicted = logits[:, :-1].reshape(-1, logits.shape[-1])
        target = ids[:, 1:].reshape(-1)
        counted = int((target != 0).sum())
        if counted == 0:
            raise ValueError("A batch must contain at least one predicted token")
        loss = F.cross_entropy(predicted, target, ignore_index=0, reduction="sum") / counted
        return loss, counted

    @torch.no_grad()
    def lex_score_continuations(self, prefix_ids, candidate_ids):
        """Mean log-probability of each candidate continuation after one prefix.

        ``prefix_ids`` is a single sequence; ``candidate_ids`` is a right-padded
        batch of continuations. Scores are comparable across candidates of
        different lengths because each is averaged over its own real tokens.
        """
        if prefix_ids.ndim != 1 or candidate_ids.ndim != 2:
            raise ValueError("Supply one prefix and a batch of candidate continuations")
        rows = len(candidate_ids)
        joined = torch.cat((prefix_ids.expand(rows, -1), candidate_ids), 1)
        logits, _ = self.lex_logits(joined)
        start = len(prefix_ids)
        scores = []
        for row in range(rows):
            total, count = 0., 0
            for offset in range(candidate_ids.shape[1]):
                token = int(candidate_ids[row, offset])
                if token == 0:
                    break
                total += float(F.log_softmax(logits[row, start + offset - 1], -1)[token])
                count += 1
            scores.append(total / count if count else float("-inf"))
        return scores

    # ---- declared trainability --------------------------------------------

    def trainable_report(self):
        trainable, frozen = [], 0
        frozen_elements = 0
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                trainable.append({"name": name, "shape": list(parameter.shape), "elements": parameter.numel()})
            else:
                frozen += 1
                frozen_elements += parameter.numel()
        return {"trainable_tensors": len(trainable), "trainable_elements": sum(t["elements"] for t in trainable),
                "frozen_tensors": frozen, "frozen_elements": frozen_elements,
                "trainable": trainable,
                "connection": self.language_config["connection"],
                "protected_by_construction": [
                    "exact rational mathematics routes (solve_solution, what_if, apply_inquiry_rule)",
                    "fitted physical field weights (field_what_if, field_trajectory, field_plan)",
                    "fitted intervention coefficients (intervention_what_if, intervention_explain, intervention_plan)",
                    "every inherited neural tensor of the shared core"]}

    def language_state(self):
        return {name: value.detach().clone()
                for name, value in self.state_dict().items() if name.startswith(LANGUAGE_PREFIXES)}

    def load_language_state(self, values):
        expected = self.language_state()
        if set(expected) != set(values):
            raise ValueError("Saved language tensor schema differs from this descendant")
        for name, tensor in values.items():
            reference = expected[name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f"Saved tensor {name} has a different shape or dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Saved tensor {name} is not finite")
        self.load_state_dict({**self.state_dict(), **values})

    def identity(self):
        return model_identity(self)


@contextmanager
def adapter_disabled(owner):
    """Temporarily return the shared core to exactly its inherited behaviour.

    The adapter is the only channel by which teaching can reach an inherited
    route, so running a retained task with and without it says, per route,
    whether that route actually depends on the shared core.
    """
    if owner.adapter is None:
        yield False
        return
    saved = owner.adapter[-1].weight.detach().clone()
    with torch.no_grad():
        owner.adapter[-1].weight.zero_()
    try:
        yield True
    finally:
        with torch.no_grad():
            owner.adapter[-1].weight.copy_(saved)


def disconnect_inherited_core(owner, *, seed=48010):
    """Re-initialise only the shared recurrent core, for the disconnected control.

    This is a control arm, never a saved descendant: it answers "does the
    inherited memory carry anything?" by keeping the identical new language
    modules and adapter while replacing the core they read.
    """
    generator = torch.Generator().manual_seed(seed)
    touched = []
    with torch.no_grad():
        for name, parameter in owner.named_parameters():
            if not (name.startswith("memory.") or name.startswith("fusion.")):
                continue
            if parameter.ndim >= 2:
                scale = parameter.shape[-1] ** -.5
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * scale)
            else:
                parameter.zero_()
            touched.append(name)
    return touched
