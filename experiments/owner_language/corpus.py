"""Immutable human-authored sources, prospective source-grouped partitions.

Every byte here was written by people and published with a recorded provenance
(`runs/HB-corpus/acquisition.json` in the read-only reference). Nothing in this
module generates, paraphrases or augments text: it copies raw bytes, verifies
them against the recorded hashes, locates the publisher's own body markers, and
cuts the body into groups along structure the authors already put there.

Splits are assigned by a deterministic hash of the *group* identity, so a
dictionary entry or a prose section is wholly inside one split and can never
appear on both sides of an evaluation. `descartes` is withheld from every split
as an unseen-author transfer source and is never trained on.

`dailydialog.zip` stays quarantined: its recorded licence is research-only with
an archive README to check, and the shared source inventory already marks it
quarantined. It is listed here so the exclusion is visible, not silent.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("D:/ai/projects/sera")
CORPUS = LAB_ROOT / "runs/owner-learning-001/corpus"
RAW = CORPUS / "raw"

GUTENBERG_LICENCE = "Project Gutenberg public domain in the USA; original notice retained in the raw bytes"

SOURCES = {
    "webster": {
        "path": "runs/HB-corpus/raw/webster.txt",
        "sha256": "86fb9c28c32008ea288ca4bcf34f4f0d3d11ccf9e0898294d98b73671497f1d3",
        "title": "Webster's Unabridged Dictionary", "authors": "Various dictionary editors",
        "url": "https://www.gutenberg.org/ebooks/29765", "licence": GUTENBERG_LICENCE,
        "structure": "dictionary", "role": "train_dev_final", "human_authored": True},
    "grammar": {
        "path": "runs/HB-corpus/raw/grammar.txt",
        "sha256": "21595f1f94403d10a6a0f4ec910f27ccc8aba95da89a730bc64029a70161a614",
        "title": "An English Grammar", "authors": "W. M. Baskervill and J. W. Sewell",
        "url": "https://www.gutenberg.org/ebooks/14006", "licence": GUTENBERG_LICENCE,
        "structure": "prose", "role": "train_dev_final", "human_authored": True},
    "plato": {
        "path": "runs/HB-corpus/raw/plato.txt",
        "sha256": "917c1cb469e1a8eba6083808764d7131da8d79140b575b4214c9d02a73ec4528",
        "title": "The Republic", "authors": "Plato; Benjamin Jowett translation",
        "url": "https://www.gutenberg.org/ebooks/1497", "licence": GUTENBERG_LICENCE,
        "structure": "prose", "role": "train_dev_final", "human_authored": True},
    "calculus": {
        "path": "runs/HB-corpus/raw/calculus.txt",
        "sha256": "fb6a9195621aeba3302d2359af4ff7cd74b285d9fb57c16431c21c0b85ed5776",
        "title": "Calculus Made Easy", "authors": "Silvanus P. Thompson",
        "url": "https://www.gutenberg.org/files/33283/33283-t/33283-t.tex", "licence": GUTENBERG_LICENCE,
        "structure": "prose", "role": "train_dev_final", "human_authored": True},
    "descartes": {
        "path": "runs/HB-corpus/raw/descartes.txt",
        "sha256": "7768ba8d3895be3d6f1492733ffab983d69875cc743edd23a1840e1530d02924",
        "title": "Discourse on the Method", "authors": "Rene Descartes; John Veitch translation",
        "url": "https://www.gutenberg.org/ebooks/59", "licence": GUTENBERG_LICENCE,
        "structure": "prose", "role": "transfer_only", "human_authored": True},
}

QUARANTINED = {
    "dailydialog": {
        "path": "runs/HB-corpus/raw/dailydialog.zip",
        "sha256": "2de93416a95060057d8b95ef9e2ab9b17a317906f1e352414001a1887f081193",
        "title": "DailyDialog", "authors": "Yanran Li et al.; human-authored conversations",
        "reason": "Recorded licence is research-only CC BY-NC-SA 4.0 with an archive README to check, "
                  "and the shared source inventory marks it quarantined. Not exposed in this stage."},
}

SPLIT_WEIGHTS = (("train", 8), ("dev", 1), ("final", 1))
TOKEN_PATTERN = re.compile(r"[a-z]+(?:['\-][a-z]+)*|[0-9]+|[^\sa-z0-9]")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_raw(reference=REFERENCE):
    """Copy the raw source bytes into the laboratory and verify them both sides."""
    RAW.mkdir(parents=True, exist_ok=True)
    staged = {}
    for name, record in {**SOURCES, **QUARANTINED}.items():
        source = Path(reference) / record["path"]
        actual = sha256_file(source)
        if actual != record["sha256"]:
            raise ValueError(f"{name}: reference bytes {actual} do not match the recorded {record['sha256']}")
        target = RAW / Path(record["path"]).name
        if not target.exists():
            shutil.copyfile(source, target)
        copied = sha256_file(target)
        if copied != record["sha256"]:
            raise ValueError(f"{name}: laboratory copy {copied} does not match the recorded hash")
        staged[name] = {**{k: v for k, v in record.items() if k != "path"},
                        "reference_path": record["path"], "laboratory_path": target.as_posix(),
                        "bytes": target.stat().st_size, "verified_sha256": copied,
                        "quarantined": name in QUARANTINED}
    return staged


def body(text):
    """The publisher's own body markers, so the licence blocks are never taught."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    if start < 0:
        return text, 0, len(text)
    start = text.index("\n", start) + 1
    end = text.find(end_marker)
    if end < 0:
        end = len(text)
    return text[start:end], start, end


def split_of(source, group, weights=SPLIT_WEIGHTS):
    """Deterministic, prospective, source-grouped split assignment."""
    total = sum(weight for _, weight in weights)
    residue = int(hashlib.sha256(f"OLA-48:{source}:{group}".encode()).hexdigest(), 16) % total
    for name, weight in weights:
        if residue < weight:
            return name
        residue -= weight
    raise AssertionError("unreachable")


def _is_headword(line):
    stripped = line.strip()
    return (1 < len(stripped) <= 60 and stripped == stripped.upper()
            and any("A" <= character <= "Z" for character in stripped)
            and all(character.isalpha() or character in " ;'-" for character in stripped))


def dictionary_entries(text, *, minimum_words=6, maximum_words=120):
    """Webster's own headword / `Defn:` structure; nothing is invented here."""
    lines = text.split("\n")
    positions = [index for index, line in enumerate(lines) if _is_headword(line)]
    for order, index in enumerate(positions):
        stop = positions[order + 1] if order + 1 < len(positions) else len(lines)
        headwords = [part.strip() for part in lines[index].strip().split(";") if part.strip()]
        block = lines[index + 1:stop]
        definition = []
        capturing = False
        for line in block:
            if line.startswith("Defn:"):
                capturing = True
                definition.append(line[len("Defn:"):].strip())
            elif capturing and line.strip():
                definition.append(line.strip())
            elif capturing:
                break
        joined = " ".join(definition).strip()
        if not headwords or not joined:
            continue
        words = joined.split()
        if not minimum_words <= len(words) <= maximum_words:
            continue
        yield {"group": headwords[0], "headword": headwords[0], "variants": headwords,
               "definition": joined, "line": index}


def prose_sections(text, *, paragraphs_per_section=4, minimum_words=40, maximum_words=400):
    """Contiguous author paragraphs; the section index is the grouping unit."""
    blocks = [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", text)]
    blocks = [block for block in blocks if len(block.split()) >= 8]
    for start in range(0, len(blocks), paragraphs_per_section):
        chunk = blocks[start:start + paragraphs_per_section]
        joined = " ".join(chunk)
        words = joined.split()
        if len(words) < minimum_words:
            continue
        yield {"group": f"section-{start // paragraphs_per_section:05d}",
               "text": " ".join(words[:maximum_words]), "paragraphs": len(chunk),
               "block_start": start}


def normalise(text):
    return unicodedata.normalize("NFKC", text).lower()


def tokenise(text):
    return TOKEN_PATTERN.findall(normalise(text))


def build_items(staged):
    """One record per teachable unit, with its split already fixed."""
    items, spans = [], {}
    for name, record in staged.items():
        if record["quarantined"]:
            continue
        raw = Path(record["laboratory_path"]).read_text(encoding="utf-8", errors="strict")
        content, start, end = body(raw)
        spans[name] = {"body_character_start": start, "body_character_end": end,
                       "body_characters": end - start}
        transfer = record["role"] == "transfer_only"
        if record["structure"] == "dictionary":
            for entry in dictionary_entries(content):
                items.append({"source": name, "kind": "definition", "group": entry["group"],
                              "split": "transfer" if transfer else split_of(name, entry["group"]),
                              "headword": entry["headword"], "variants": entry["variants"],
                              "text": entry["definition"], "locator": f"line:{entry['line']}"})
        else:
            for section in prose_sections(content):
                items.append({"source": name, "kind": "passage", "group": section["group"],
                              "split": "transfer" if transfer else split_of(name, section["group"]),
                              "text": section["text"], "locator": f"block:{section['block_start']}"})
    for item in items:
        item["id"] = hashlib.sha256(
            f"{item['source']}:{item['group']}:{item['locator']}".encode()).hexdigest()[:20]
        item["tokens"] = len(tokenise(item["text"]))
    return items, spans


def build_vocabulary(items, size=8192):
    """Frozen from the training split alone; dev, final and transfer never vote."""
    counts = Counter()
    for item in items:
        if item["split"] != "train":
            continue
        counts.update(tokenise(item["text"]))
        if item["kind"] == "definition":
            counts.update(tokenise(item["headword"]))
    special = ["<pad>", "<unk>", "<eos>", "<def>", "<is>"]
    ordered = [word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    words = special + ordered[:max(0, size - len(special))]
    return {"schema": "sera.owner-language.vocabulary.1", "size": len(words), "words": words,
            "special": special, "pattern": TOKEN_PATTERN.pattern,
            "counted_items": sum(1 for item in items if item["split"] == "train"),
            "distinct_observed": len(counts),
            "coverage": round(sum(counts[w] for w in words if w in counts) / max(1, sum(counts.values())), 6),
            "digest": hashlib.sha256(json.dumps(words).encode()).hexdigest()}


def summarise(items):
    by = {}
    for item in items:
        key = (item["source"], item["split"])
        entry = by.setdefault(key, {"items": 0, "tokens": 0, "groups": set()})
        entry["items"] += 1
        entry["tokens"] += item["tokens"]
        entry["groups"].add(item["group"])
    return [{"source": source, "split": split, "items": value["items"], "tokens": value["tokens"],
             "groups": len(value["groups"])}
            for (source, split), value in sorted(by.items())]


def build(reference=REFERENCE):
    CORPUS.mkdir(parents=True, exist_ok=True)
    staged = stage_raw(reference)
    items, spans = build_items(staged)
    vocabulary = build_vocabulary(items)
    items_path = CORPUS / "items.jsonl"
    with items_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    (CORPUS / "vocabulary.json").write_text(json.dumps(vocabulary, indent=1) + "\n", encoding="utf-8")
    manifest = {"schema": "sera.owner-language.corpus.1",
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reference_root": Path(reference).as_posix(),
                "provenance": (Path(reference) / "runs/HB-corpus/acquisition.json").as_posix(),
                "provenance_sha256": sha256_file(Path(reference) / "runs/HB-corpus/acquisition.json"),
                "sources": {name: {**record, **spans.get(name, {})} for name, record in staged.items()},
                "quarantined": list(QUARANTINED),
                "split_weights": dict(SPLIT_WEIGHTS),
                "split_rule": "sha256('OLA-48:<source>:<group>') modulo 10; 0-7 train, 8 dev, 9 final",
                "transfer_rule": "sources with role transfer_only are withheld from every split",
                "generated_text": "none; no paraphrase, translation or synthetic example is admitted",
                "items": len(items), "items_path": items_path.as_posix(),
                "items_sha256": sha256_file(items_path),
                "vocabulary": {k: v for k, v in vocabulary.items() if k != "words"},
                "partitions": summarise(items)}
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_items(split=None, sources=None):
    rows = [json.loads(line) for line in (CORPUS / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if split is not None:
        wanted = {split} if isinstance(split, str) else set(split)
        rows = [row for row in rows if row["split"] in wanted]
    if sources is not None:
        wanted = {sources} if isinstance(sources, str) else set(sources)
        rows = [row for row in rows if row["source"] in wanted]
    return rows


def load_vocabulary():
    return json.loads((CORPUS / "vocabulary.json").read_text(encoding="utf-8"))
