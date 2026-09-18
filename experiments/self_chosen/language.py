"""Imagine request associations; keep human assessment annotations out of fitting."""

import re
from collections import Counter

import torch

from experiments.stream_curriculum.model import encode

from .common import LANGUAGES, ROOT, RUN, digest, read, sha, write


def tokens(text):
    return tuple(re.findall(r"\w+", text.lower()))


def matches(pattern, text):
    words = tokens(text)
    return any(tuple(pattern) == words[i:i+len(pattern)] for i in range(len(words)-len(pattern)+1))


def prepare(owner):
    if (RUN / "language-private.json").exists():
        raise FileExistsError("Preserve prepared human evidence")
    public, private, sources = {}, {}, []
    vocabulary = owner.stream_config["vocabulary"]["intents"]
    for locale in LANGUAGES:
        path = ROOT / ("runs/SC-data-002/train.jsonl" if locale == "en-US" else f"runs/SS-language-data/{locale}-train.jsonl")
        import json
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        partitions = {"fit": [], "check": [], "final": []}
        for row in sorted(records, key=lambda r: digest(str(r["id"]))):
            bucket = int(digest(str(row["id"]))[:8], 16) % 5
            kind = "fit" if bucket < 3 else "check" if bucket == 3 else "final"
            if len(partitions[kind]) < 768:
                partitions[kind].append({"id": str(row["id"]), "text": row["text"], "intent": row["intent"]})
        imagined = []
        with torch.no_grad():
            for start in range(0, len(partitions["fit"]), 64):
                batch = partitions["fit"][start:start+64]
                logits, _ = owner.request_logits(encode([r["text"] for r in batch]))
                for row, value in zip(batch, logits.softmax(-1), strict=True):
                    imagined.append({"id": row["id"], "text": row["text"],
                                     "prediction": vocabulary[int(value.argmax())],
                                     "confidence": float(value.max())})
        public[locale], private[locale] = imagined, partitions
        sources.append({"locale": locale, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                        "counts": {k: len(v) for k, v in partitions.items()},
                        "partition": "underlying human request id shared across translations, SHA256 mod 5; original parent training corpus"})
    write(RUN / "language-public.json", public)
    write(RUN / "language-private.json", private)
    write(RUN / "language-sources.json", {"sources": sources, "new_unique_source_documents": 0,
                                          "meaning": "prospective continuation separation within retained human corpus; parent may have seen these records"})
    return public


def questions(public):
    result = []
    for locale, rows in public.items():
        groups = Counter(r["prediction"] for r in rows)
        for intent, count in sorted(groups.items()):
            if count >= 8:
                result.append({"domain": locale, "target": intent, "missing": None,
                               "id": digest([locale, intent]),
                               "question": f"Which recurring human expression predicts {intent} in {locale}, and when is that association wrong?"})
    return result


def propose(question, rows, method, excluded=()):
    length = 1 if method == "single" else 2
    candidates = Counter()
    for row in rows:
        if row["prediction"] == question["target"]:
            words = tokens(row["text"])
            candidates.update(set(words[i:i+length] for i in range(len(words)-length+1)))
    ranked = []
    for pattern, count in candidates.items():
        if tuple(pattern) in excluded:
            continue
        if count < 4:
            continue
        covered = [r for r in rows if matches(pattern, r["text"])]
        agreement = sum(r["prediction"] == question["target"] for r in covered)/len(covered)
        if agreement >= .8:
            ranked.append((agreement*len(covered)**.5, pattern, agreement, len(covered)))
    if not ranked:
        return {"status": "OPEN", "method": method, "reason": "No repeated high-agreement imagined association"}
    _, pattern, agreement, coverage = max(ranked, key=lambda x: (x[0], x[1]))
    return {"status": "CONJECTURE", "method": method, "pattern": list(pattern),
            "target": question["target"], "imagined_agreement": agreement, "imagined_coverage": coverage,
            "canonical": [question["domain"], list(pattern), question["target"]],
            "requires": ["human_request_text"], "execution_cost": len(pattern)}


def assess(question, proposal, partition="check"):
    if proposal["status"] != "CONJECTURE":
        return {"accepted": False, "kind": "OPEN_SEARCH", "reason": proposal["reason"]}
    rows = read(RUN / "language-private.json")[question["domain"]][partition]
    covered = [r for r in rows if matches(proposal["pattern"], r["text"])]
    good = sum(r["intent"] == proposal["target"] for r in covered)
    accuracy = good/len(covered) if covered else None
    return {"accepted": len(covered) >= 10 and accuracy >= .9, "kind": "HUMAN_ANNOTATED_ASSOCIATION",
            "partition": partition, "correct": good, "count": len(covered), "accuracy": accuracy,
            "covered_ids": [r["id"] for r in covered],
            "verified_ids": [r["id"] for r in covered if r["intent"] == proposal["target"]],
            "counterexamples": [r for r in covered if r["intent"] != proposal["target"]],
            "source": sha(RUN / "language-sources.json"), "observed_event": False,
            "scope": "this retained human corpus and locale; corpus annotations, not universal semantic equivalence",
            "applicability_permission": False}
