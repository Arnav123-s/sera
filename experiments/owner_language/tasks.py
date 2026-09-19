"""Evaluation items cut mechanically out of the human sources.

No question here was written by a model. A cloze item removes a word the author
wrote and asks the learner to put it back; a definition item asks which headword
the lexicographer attached to a definition. The right answer is the text itself,
so there is nothing to author and nothing to leak beyond the split the item's
group already belongs to.

Distractors are drawn deterministically from the frozen vocabulary within a
frequency band, so chance accuracy is exactly 1/(distractors+1) and two runs see
identical items.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .corpus import CORPUS, load_items, load_vocabulary, tokenise

SPECIAL = ("<pad>", "<unk>", "<eos>", "<def>", "<is>")
EVALUATION = CORPUS / "evaluation"


class Tokens:
    """The frozen word vocabulary, with `<unk>` for anything it never saw."""

    def __init__(self, vocabulary=None):
        self.record = vocabulary or load_vocabulary()
        self.words = list(self.record["words"])
        self.index = {word: position for position, word in enumerate(self.words)}
        self.pad, self.unk = 0, 1
        self.eos = self.index["<eos>"]
        self.define, self.is_token = self.index["<def>"], self.index["<is>"]
        self.digest = self.record["digest"]

    def __len__(self):
        return len(self.words)

    def encode(self, text, limit=None):
        ids = [self.index.get(word, self.unk) for word in tokenise(text)]
        return ids[:limit] if limit else ids

    def encode_words(self, words):
        return [self.index.get(word, self.unk) for word in words]

    def known(self, word):
        return word in self.index and word not in SPECIAL


def _stream(seed):
    """A tiny reproducible integer stream; no global RNG is touched."""
    state = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 128)
        yield state >> 33


def _band(position, size, bands=8):
    return min(bands - 1, position * bands // size)


def _distractors(tokens, answer, count, seed, pool_by_band):
    """Same-frequency-band words, deterministic, never the answer itself."""
    band = _band(tokens.index[answer], len(tokens))
    pool = pool_by_band[band]
    picked, seen = [], {answer}
    numbers = _stream(seed)
    attempts = 0
    while len(picked) < count and attempts < count * 200:
        attempts += 1
        word = pool[next(numbers) % len(pool)]
        if word in seen:
            continue
        seen.add(word)
        picked.append(word)
    if len(picked) < count:
        return None
    return picked


def _pools(tokens, bands=8):
    pools = [[] for _ in range(bands)]
    for position, word in enumerate(tokens.words):
        if word in SPECIAL:
            continue
        pools[_band(position, len(tokens), bands)].append(word)
    return [pool for pool in pools]


def cloze_items(items, tokens, *, per_passage=12, distractors=7, context=32, minimum_length=6):
    """Remove a content word the author wrote; ask which word belongs there."""
    pools = _pools(tokens)
    built, skipped = [], {"short_passage": 0, "no_eligible_position": 0, "no_distractors": 0}
    for item in items:
        words = tokenise(item["text"])
        if len(words) < minimum_length + 4:
            skipped["short_passage"] += 1
            continue
        eligible = [position for position, word in enumerate(words)
                    if position >= minimum_length and len(word) >= 4 and word.isalpha() and tokens.known(word)]
        if not eligible:
            skipped["no_eligible_position"] += 1
            continue
        numbers = _stream("cloze:" + item["id"])
        chosen, seen = [], set()
        for _ in range(per_passage * 8):
            if len(chosen) >= per_passage:
                break
            position = eligible[next(numbers) % len(eligible)]
            if position in seen:
                continue
            seen.add(position)
            chosen.append(position)
        for position in sorted(chosen):
            answer = words[position]
            options = _distractors(tokens, answer, distractors, f"cloze:{item['id']}:{position}", pools)
            if options is None:
                skipped["no_distractors"] += 1
                continue
            prefix = words[max(0, position - context):position]
            built.append({"id": f"{item['id']}:{position}", "family": "cloze", "source": item["source"],
                          "group": item["group"], "split": item["split"], "locator": item["locator"],
                          "prefix": prefix, "answer": answer,
                          "candidates": sorted([answer, *options]),
                          "chance": 1. / (distractors + 1)})
    return built, skipped


def definition_items(items, tokens, *, distractors=7, definition_limit=40, headword_words=3):
    """Ask which headword the lexicographer attached to this definition."""
    pools = _pools(tokens)
    built, skipped = [], {"headword_out_of_vocabulary": 0, "definition_too_short": 0, "no_distractors": 0}
    for item in items:
        if item["kind"] != "definition":
            continue
        headword = tokenise(item["headword"])
        if not headword or len(headword) > headword_words or not all(tokens.known(word) for word in headword):
            skipped["headword_out_of_vocabulary"] += 1
            continue
        definition = tokenise(item["text"])
        if len(definition) < 6:
            skipped["definition_too_short"] += 1
            continue
        if len(headword) != 1:
            # Multi-word headwords have no single-word distractor band; keep the
            # family homogeneous rather than mixing scoring regimes.
            skipped["headword_out_of_vocabulary"] += 1
            continue
        options = _distractors(tokens, headword[0], distractors, "defn:" + item["id"], pools)
        if options is None:
            skipped["no_distractors"] += 1
            continue
        built.append({"id": item["id"], "family": "definition", "source": item["source"],
                      "group": item["group"], "split": item["split"], "locator": item["locator"],
                      "definition": definition[:definition_limit], "answer": headword[0],
                      "candidates": sorted([headword[0], *options]),
                      "chance": 1. / (distractors + 1)})
    return built, skipped


def language_model_items(items, tokens, *, length=64):
    """Held-out passages for the direct next-token measurement."""
    built = []
    for item in items:
        ids = tokens.encode(item["text"], limit=length)
        if len(ids) < 8:
            continue
        built.append({"id": item["id"], "family": "next_token", "source": item["source"],
                      "group": item["group"], "split": item["split"], "locator": item["locator"],
                      "ids": ids})
    return built


def build_evaluation(splits=("dev", "final", "transfer"), *, cap_per_family=None):
    """Freeze one evaluation set per split, hashed, before any teaching."""
    tokens = Tokens()
    EVALUATION.mkdir(parents=True, exist_ok=True)
    caps = cap_per_family or {"cloze": 1200, "definition": 1200, "next_token": 600}
    record = {"schema": "sera.owner-language.evaluation.1", "vocabulary_digest": tokens.digest,
              "chance_accuracy": 1. / 8, "sets": {}}
    for split in splits:
        items = load_items(split=split)
        cloze, cloze_skipped = cloze_items(items, tokens)
        definition, definition_skipped = definition_items(items, tokens)
        language = language_model_items(items, tokens)
        families = {"cloze": cloze, "definition": definition, "next_token": language}
        selected = {}
        for family, rows in families.items():
            rows = sorted(rows, key=lambda row: hashlib.sha256((family + ":" + row["id"]).encode()).hexdigest())
            selected[family] = rows[:caps.get(family, len(rows))]
        path = EVALUATION / f"{split}.json"
        payload = {"split": split, "families": selected,
                   "skipped": {"cloze": cloze_skipped, "definition": definition_skipped},
                   "available": {family: len(rows) for family, rows in families.items()}}
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        record["sets"][split] = {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "counts": {family: len(rows) for family, rows in selected.items()},
            "available": payload["available"], "skipped": payload["skipped"],
            "groups": {family: len({row["group"] for row in rows}) for family, rows in selected.items()},
            "sources": {family: sorted({row["source"] for row in rows}) for family, rows in selected.items()}}
    (EVALUATION / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_evaluation(split):
    path = EVALUATION / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set for {split} has not been frozen")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_manifest():
    return json.loads((EVALUATION / "manifest.json").read_text(encoding="utf-8"))
