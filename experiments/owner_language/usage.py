"""Restore a trained descendant and use it, from the laboratory path alone.

The descendant has to be usable without any promotion into production, so this
module rebuilds it the same way the trainer did — restore the real Stage 47
owner from the laboratory baseline store, attach the language route, load the
saved trainable tensors — and then checks that the identity it gets back is the
identity the checkpoint recorded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .descendant import LanguageAcquisitionR1
from .owner import BASELINE_STORE, restore_owner
from .tasks import Tokens
from .training import CheckpointStore

LAB_ROOT = Path(__file__).resolve().parents[2]
RUNS = LAB_ROOT / "runs/owner-learning-001"


def descendant_store(run, arm):
    return RUNS / "descendants" / run / arm


def restore_descendant(run, arm="connected", *, baseline=BASELINE_STORE, revision=None):
    """Rebuild a trained descendant from the laboratory baseline and a checkpoint."""
    store = CheckpointStore(descendant_store(run, arm))
    if revision is not None:
        pointer = store.revisions / f"{int(revision):06d}.json"
        metadata = json.loads(pointer.read_text())
        payload = torch.load(store.revisions / metadata["weights"], weights_only=False)
    else:
        found = store.read()
        if found is None:
            raise FileNotFoundError(f"No checkpoint for run {run} arm {arm}")
        metadata, payload = found
    tokens = Tokens()
    if metadata["vocabulary_digest"] != tokens.digest:
        raise ValueError("The checkpoint was written against a different frozen vocabulary")
    session, growth, restore_record = restore_owner(baseline, expect=metadata["baseline_contract"]["baseline_owner"])
    contract = session.state()
    if contract["contracts"] != metadata["baseline_contract"]["contracts"]:
        raise ValueError("The baseline contract changed since this descendant was trained")
    configuration = metadata["language_config"]
    owner = LanguageAcquisitionR1.attach(session.owner, tokens.record,
                                         embedding=configuration["embedding"],
                                         adapter_rank=configuration["adapter_rank"],
                                         seed=configuration["seed"])
    owner.load_language_state(payload["language_state"])
    if owner.identity() != metadata["descendant_owner"]:
        raise ValueError("The rebuilt descendant identity differs from the checkpoint")
    return {"session": session, "growth": growth, "owner": owner, "tokens": tokens,
            "metadata": metadata, "restore": restore_record}


def complete(owner, tokens, text, *, top=5, context=32):
    """The learner's own next-word distribution after a human prefix."""
    ids = tokens.encode(text)[-context:]
    if len(ids) < 1:
        raise ValueError("Supply at least one in-scope word")
    with torch.no_grad():
        logits, _ = owner.lex_logits(torch.tensor([ids], dtype=torch.long))
        probabilities = F.softmax(logits[0, -1], -1)
    values, positions = probabilities.topk(top)
    return {"kind": "LEARNED_LANGUAGE_CONTINUATION",
            "prefix_words": len(ids),
            "unknown_words": int(sum(1 for token in ids if token == tokens.unk)),
            "next": [{"word": tokens.words[int(position)], "probability": float(value)}
                     for value, position in zip(values.tolist(), positions.tolist())],
            "scope": "word distribution over the frozen 8,192-word training vocabulary",
            "evidence_kind": "learned from human-authored public-domain books; not a measurement or a proof"}


def choose(owner, tokens, prompt_ids, candidates):
    """Score a finite candidate set at the final prompt position."""
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 64:
        raise ValueError("Declare between two and 64 candidates")
    with torch.no_grad():
        logits, _ = owner.lex_logits(torch.tensor([prompt_ids], dtype=torch.long))
        logprobs = F.log_softmax(logits[0, -1], -1)
    scores = []
    for word in candidates:
        token = tokens.index.get(word, tokens.unk)
        scores.append({"candidate": word, "in_vocabulary": word in tokens.index,
                       "log_probability": float(logprobs[token])})
    scores.sort(key=lambda row: -row["log_probability"])
    return scores


def define(owner, tokens, definition, candidates):
    """Which headword does this lexicographer's definition belong to?"""
    ids = [tokens.define, *tokens.encode(definition, limit=40), tokens.is_token]
    scores = choose(owner, tokens, ids, candidates)
    margin = scores[0]["log_probability"] - scores[1]["log_probability"]
    return {"kind": "LEARNED_LEXICAL_CHOICE", "selected": scores[0]["candidate"],
            "margin_log_probability": margin, "scores": scores,
            "prompt_words": len(ids) - 2,
            "status": "ANSWERED" if margin > 0 else "UNDECIDED",
            "scope": "a forced choice among the supplied candidates, not an open-vocabulary definition",
            "source_of_the_question": "Webster's Unabridged Dictionary, Project Gutenberg 29765"}


def cloze(owner, tokens, prefix, candidates):
    """Which word did the author write here?"""
    ids = tokens.encode(prefix)[-32:]
    scores = choose(owner, tokens, ids, candidates)
    margin = scores[0]["log_probability"] - scores[1]["log_probability"]
    return {"kind": "LEARNED_CONTEXT_CHOICE", "selected": scores[0]["candidate"],
            "margin_log_probability": margin, "scores": scores, "prefix_words": len(ids),
            "status": "ANSWERED" if margin > 0 else "UNDECIDED",
            "scope": "a forced choice among the supplied candidates"}


def perform(context, request):
    """Language routes first, then every retained route of the existing owner."""
    from scripts.intervention_use import perform as retained
    owner, tokens = context["owner"], context["tokens"]
    kind = request.get("kind")
    started = time.perf_counter()
    if kind == "language_status":
        metadata = context["metadata"]
        result = {"descendant": metadata["descendant_owner"],
                  "parent_owner": metadata["baseline_contract"]["baseline_owner"],
                  "trained_steps": metadata["step"], "arm": metadata["arm"],
                  "revision": metadata["revision"], "note": metadata.get("note"),
                  "vocabulary": len(tokens), "vocabulary_digest": tokens.digest,
                  "trainable": owner.trainable_report()["trainable_elements"],
                  "frozen": owner.trainable_report()["frozen_elements"],
                  "stages": [record.get("stage") for record in metadata["stage_records"]],
                  "retained_routes": "every route of the Stage 47 owner remains reachable through this interface"}
    elif kind == "language_complete":
        result = complete(owner, tokens, request["text"], top=int(request.get("top", 5)))
    elif kind == "language_define":
        result = define(owner, tokens, request["definition"], request["candidates"])
    elif kind == "language_cloze":
        result = cloze(owner, tokens, request["prefix"], request["candidates"])
    else:
        answer = retained(context["session"], context["growth"], request)
        answer["seconds"] = time.perf_counter() - started
        return answer
    return {"id": request.get("id"), "kind": kind, "owner": owner.identity(), "result": result,
            "seconds": time.perf_counter() - started}
