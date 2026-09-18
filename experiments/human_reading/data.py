"""Attributed human questions, exact span checks and article-disjoint partitions."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/29_human_reading"
RUN = ROOT / "runs/HR-study-001"
RAW = ROOT / "runs/HR-corpus/raw"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized(text):
    return " ".join(text.casefold().split())


def sentences(text):
    result = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|$)", text):
        a, b = match.span()
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a < b:
            result.append({"text": text[a:b], "start": a, "end": b})
    return result


def prepare(final=False):
    if final and not (OUT / "selection.json").exists():
        raise ValueError("Freeze selection before opening final labels")
    source = read(OUT / "sources.json")["records"][1 if final else 0]
    path = RAW / ("dev-v1.1.json" if final else "train-v1.1.json")
    if sha(path) != source["sha256"]:
        raise ValueError("Original dataset bytes changed")
    old = read(RUN / "data-manifest.json") if final else {}
    forbidden_titles = set(old.get("all_official_training_titles", []))
    forbidden_passages = set(old.get("all_official_training_passages", []))
    seen, rejects = set(), Counter()
    groups = {"final": []} if final else {"train": [], "dev": []}
    titles, passages = [], []
    for article in read(path)["data"]:
        title = article["title"]
        titles.append(title)
        split = "final" if final else ("dev" if int(digest("HR-001:article:" + title), 16) % 5 == 0 else "train")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_hash = digest(normalized(context))
            passages.append(context_hash)
            if final and (title in forbidden_titles or context_hash in forbidden_passages):
                rejects["overlapping_article_or_passage"] += len(paragraph["qas"])
                continue
            candidates = sentences(context)
            if not 2 <= len(candidates) <= 16 or len(context) > 6000:
                rejects["context_or_candidate_limit"] += len(paragraph["qas"])
                continue
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                key = digest(normalized(question) + "\n" + normalized(context))
                if not question or key in seen:
                    rejects["empty_or_duplicate"] += 1
                    continue
                gold, valid = set(), True
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    end = start + len(text)
                    matches = [i for i, c in enumerate(candidates) if c["start"] <= start and end <= c["end"]]
                    if not text or context[start:end] != text or len(matches) != 1:
                        valid = False
                        break
                    gold.add(matches[0])
                if not valid or not gold:
                    rejects["annotation_alignment"] += 1
                    continue
                seen.add(key)
                groups[split].append({"id": qa["id"], "title": title, "question": question,
                                      "context": context, "context_sha256": digest(context),
                                      "sentences": candidates, "gold": sorted(gold),
                                      "source": source["url"], "source_sha256": source["sha256"]})
    counts = {}
    for split, rows in groups.items():
        kept, per_article = [], Counter()
        for row in sorted(rows, key=lambda r: digest(r["id"])):
            if per_article[row["title"]] == 60:
                continue
            kept.append(row)
            per_article[row["title"]] += 1
            if len(kept) == (12000 if split == "train" else 1000):
                break
        destination = RUN / (split + ".json")
        if destination.exists():
            raise ValueError("Prepared cohorts must not be overwritten")
        write(destination, kept)
        counts[split] = {"eligible": len(rows), "selected": len(kept), "articles": len(per_article),
                         "sha256": sha(destination), "candidate_sentences": sum(len(r["sentences"]) for r in kept)}
    result = {"schema": "sera.human-reading-data.1", "counts": counts, "rejections": dict(rejects),
              "source_sha256": source["sha256"], "all_official_training_titles": titles if not final else [],
              "all_official_training_passages": sorted(set(passages)) if not final else [],
              "protocol_sha256": sha(OUT / "PROTOCOL.md"), "data_source_sha256": sha(Path(__file__)),
              "annotation_origin": "human SQuAD 1.1 annotations; no generated text"}
    write(RUN / ("final-manifest.json" if final else "data-manifest.json"), result)
    return {"counts": counts, "rejections": dict(rejects)}


if __name__ == "__main__":
    print(json.dumps(prepare()))
