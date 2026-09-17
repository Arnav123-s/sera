"""Strict source-labelled examples and an exactly resumable bounded shuffle."""

import copy
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/intake/massive-1.1-en-US-20260916"
PREPARED = ROOT / "runs/SC-data-002"
MAX_TOKENS = 40
BUCKETS = 8192


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_id(token):
    return 1 + int.from_bytes(hashlib.blake2b(token.lower().encode(), digest_size=8).digest(), "big") % (BUCKETS - 1)


def parse(raw):
    """Reconstruct the raw annotation; never recover labels from the request at inference."""
    annotated = raw["annot_utt"]
    parts, spans, cursor, length = [], [], 0, 0
    for match in re.finditer(r"\[([^\[\]:]+)\s+:\s+([^\[\]]+)\]", annotated):
        prefix, label, value = annotated[cursor:match.start()], match[1].strip(), match[2]
        parts.extend((prefix, value))
        start = length + len(prefix)
        spans.append((start, start + len(value), label))
        length = start + len(value)
        cursor = match.end()
    parts.append(annotated[cursor:])
    text = "".join(parts)
    if "[" in text or "]" in text or text.split() != raw["utt"].split():
        raise ValueError("Source annotation does not reconstruct the utterance")
    matches = list(re.finditer(r"\S+", text))
    if not 0 < len(matches) <= MAX_TOKENS:
        raise ValueError("Source length outside the declared 1..40 token scope")
    tags = ["O"] * len(matches)
    for start, end, label in spans:
        selected = [i for i, match in enumerate(matches) if start <= match.start() and match.end() <= end]
        if not selected or matches[selected[0]].start() != start or matches[selected[-1]].end() != end:
            raise ValueError("Annotation cuts through a token")
        for j, i in enumerate(selected):
            if tags[i] != "O":
                raise ValueError("Overlapping source annotations")
            tags[i] = ("B-" if j == 0 else "I-") + label
    return {"id": raw["id"], "partition": raw["partition"], "scenario": raw["scenario"],
            "text": raw["utt"], "intent": raw["intent"], "tokens": [m[0] for m in matches], "tags": tags}


def prepare():
    """Process train/dev only. Final test bytes are never opened here."""
    PREPARED.mkdir(parents=True, exist_ok=False)
    intents, labels, rows, rejected = set(), set(), {}, []
    for part in ("train", "dev"):
        count = 0
        with (SOURCE / f"{part}.jsonl").open(encoding="utf-8") as source, \
                (PREPARED / f"{part}.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for line in source:
                raw = json.loads(line)
                if raw["partition"] != part or raw["locale"] != "en-US":
                    raise ValueError("Source partition or locale mismatch")
                try:
                    row = parse(raw)
                except ValueError as error:
                    rejected.append({"partition": part, "id": raw["id"], "reason": str(error)})
                    continue
                if part == "train":
                    intents.add(row["intent"])
                    labels.update(t[2:] for t in row["tags"] if t != "O")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        rows[part] = count
    vocabulary = {"intents": sorted(intents), "tags": ["O"] + [prefix + name for name in sorted(labels) for prefix in ("B-", "I-")]}
    write(PREPARED / "vocabulary.json", vocabulary)
    manifest = {"source_manifest": sha(SOURCE / "manifest.json"), "rows": rows,
                "files": {n: sha(PREPARED / n) for n in ("train.jsonl", "dev.jsonl", "vocabulary.json")},
                "rejected": rejected, "final_test_opened": False, "data_source": sha(Path(__file__))}
    write(PREPARED / "manifest.json", manifest)
    return manifest


def first_rows(part, count):
    if part not in ("train", "dev"):
        raise ValueError("Development reader cannot access the final test")
    result = []
    with (PREPARED / f"{part}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            result.append(json.loads(line))
            if len(result) == count:
                break
    return result


class RowStream:
    """Buffer/cursor/RNG are part of the optimizer checkpoint, including EOF draining."""
    def __init__(self, path, seed, *, limit=0, buffer_size=128, state=None):
        self.path = Path(path)
        self.identity = sha(path)
        self.limit, self.buffer_size = limit, buffer_size
        self.rng = random.Random(seed)
        self.offset, self.seen, self.epoch, self.buffer, self.eof = 0, 0, 0, [], False
        if state is not None:
            if (state["sha256"] != self.identity or state["limit"] != limit
                    or state["buffer_size"] != buffer_size):
                raise ValueError("Changed stream data or configuration")
            for name in ("offset", "seen", "epoch", "buffer", "eof"):
                setattr(self, name, copy.deepcopy(state[name]))
            self.rng.setstate(state["rng"])

    def _fill(self):
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while len(self.buffer) < self.buffer_size and not self.eof:
                if self.limit and self.seen >= self.limit:
                    self.eof = True
                    break
                line = source.readline()
                self.offset = source.tell()
                if not line:
                    self.eof = True
                    break
                row = json.loads(line)
                if row["partition"] != "train":
                    raise ValueError("Only training rows may enter an optimizer stream")
                self.buffer.append(row)
                self.seen += 1

    def take(self, count):
        result = []
        for _ in range(count):
            self._fill()
            if not self.buffer:
                if not self.seen:
                    raise ValueError("Empty training stream")
                self.offset, self.seen, self.eof = 0, 0, False
                self.epoch += 1
                self._fill()
            result.append(self.buffer.pop(self.rng.randrange(len(self.buffer))))
        return result

    def snapshot(self):
        return {"sha256": self.identity, "limit": self.limit, "buffer_size": self.buffer_size,
                "offset": self.offset, "seen": self.seen, "epoch": self.epoch,
                "buffer": copy.deepcopy(self.buffer), "eof": self.eof, "rng": self.rng.getstate()}
