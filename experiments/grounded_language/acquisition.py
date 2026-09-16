"""Bounded lessons for missing wording; sources and adaptation scopes stay explicit."""

import copy
import re

import numpy as np
import torch
from torch.nn import functional as F

from .data import TEMPLATES, WORDS, tokens
from .model import tensors, trainable
from .study import evaluate


def shape(text):
    if not isinstance(text, str) or re.fullmatch(r"[a-zA-Z0-9\s]+", text) is None:
        raise ValueError("Only the qualified plain mathematical wording is supported")
    _, words = tokens(text)
    return " ".join("#" if word in WORDS or word in {str(i) for i in range(11)} else word for word in words)


def qualified_shapes(*, novel=False, word=False):
    result = []
    for family in TEMPLATES:
        for template in family[:4 if novel else 3]:
            text = template.format(a=1, b=2, c=3)+" modulo eleven"
            result.append(shape(text))
            if word and "multiply" in text:
                result.append(shape(text.replace("multiply", "scale")))
    return sorted(set(result))


def guarded_interpret(owner, text, *, threshold, shapes):
    try:
        structure = shape(text)
    except ValueError as error:
        return {"status": "WITHHELD", "reason": str(error)}
    if structure not in shapes:
        return {"status": "NEEDS_LESSON", "reason": "This wording has not passed an acquisition check", "shape": structure}
    interpretation = owner.interpret([text])[0]
    if interpretation["missing_words"]:
        return {"status": "NEEDS_LESSON", "reason": "Missing vocabulary", **interpretation}
    if interpretation["confidence"] < threshold:
        return {"status": "WITHHELD", "reason": "Below the fixed empirical confidence filter", **interpretation}
    from .data import execute
    return {"status": "ACCEPTED", **interpretation, **execute(interpretation["labels"]),
            "boundary": "Learned translation in a supplied finite wording family; independent substitution checks its formal interpretation."}


def lesson(owner, teaching, development, *, kind, steps=120, seed=160001, checkpoint=None, resume=None):
    """The curriculum and update rule are supplied, not self-invented learning-to-learn."""
    if kind not in ("word", "wording") or not 1 <= steps <= 400 or not teaching or not development:
        raise ValueError("A declared bounded lesson is required")
    initial_state = copy.deepcopy(owner.state_dict())
    initial_flags = {name: p.requires_grad for name, p in owner.named_parameters()}
    hook = trainable(owner, "word" if kind == "word" else "interface", novel_word="scale" if kind == "word" else None)
    parameters = [p for p in owner.parameters() if p.requires_grad]
    # Zero decay is essential: the word gradient mask alone does not stop AdamW decay.
    optimizer = torch.optim.AdamW(parameters, lr=.02 if kind == "word" else .003, weight_decay=0)
    rng = np.random.default_rng(seed)
    ids, labels = tensors(teaching)
    best, best_loss, history, first_step = None, float("inf"), [], 0
    if resume is not None:
        owner.load_state_dict(resume["state"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        rng.bit_generator.state = resume["numpy_rng"]
        torch.set_rng_state(resume["torch_rng"])
        best, best_loss, history = resume["best_state"], resume["best_loss"], copy.deepcopy(resume["history"])
        first_step = resume["step"]
        if not 0 <= first_step < steps or resume["flags"] != initial_flags:
            raise ValueError("Lesson resume must match the original owner flags and extend its recorded step")
    try:
        for index in range(first_step, steps):
            owner.train()
            selection = rng.integers(len(teaching), size=16)
            outputs = owner.language(ids[selection])
            loss = sum(F.cross_entropy(value, labels[selection, i]) for i, value in enumerate(outputs))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite lesson loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            history.append({"step": index+1, "loss": float(loss.detach())})
            if (index+1) % 20 == 0 or index+1 == steps:
                score, _ = evaluate(owner, development)
                history[-1]["development"] = score
                if score["joint_cross_entropy"] < best_loss:
                    best_loss, best = score["joint_cross_entropy"], copy.deepcopy(owner.state_dict())
                if checkpoint:
                    checkpoint({"step": index+1, "state": copy.deepcopy(owner.state_dict()),
                                "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                                "torch_rng": torch.get_rng_state(), "history": history,
                                "best_state": best, "best_loss": best_loss, "flags": initial_flags})
        owner.load_state_dict(best, strict=True)
    finally:
        if hook is not None:
            hook.remove()
        for name, parameter in owner.named_parameters():
            parameter.requires_grad_(initial_flags[name])
    qualified, _ = evaluate(owner, development)
    admitted = qualified["exact_translation"] >= .99
    if kind == "word" and admitted:
        owner.language_words = sorted(set(owner.language_words) | {"scale"})
    changed = {name: int((value != initial_state[name]).sum()) for name, value in owner.state_dict().items()
               if not torch.equal(value, initial_state[name])}
    return {"kind": kind, "steps": steps, "teaching_examples": len(teaching), "teaching_draws": steps*16,
            "new_steps": steps-first_step,
            "development": qualified, "admitted": admitted, "changed_tensors": changed,
            "changed_scalars": sum(changed.values()), "history": history,
            "scope": "one vocabulary embedding" if kind == "word" else "learned language interface; same recurrent core held fixed",
            "rule": "supplied AdamW + separate labeled development check"}


def validate_shapes(owner, rows):
    """Finite labeled checks retire unsupported wording after a parameter update."""
    _, raw = evaluate(owner, rows)
    groups = {}
    for row in raw:
        groups.setdefault(shape(row["text"]), []).append(row["translation_correct"])
    metrics = {name: {"examples": len(values), "accuracy": sum(values)/len(values)} for name, values in groups.items()}
    admitted = sorted(name for name, value in metrics.items() if value["examples"] >= 24 and value["accuracy"] >= .99)
    return {"shapes": admitted, "metrics": metrics,
            "scope": "Empirical supplied-shape admission. No distribution-free semantic correctness guarantee."}
