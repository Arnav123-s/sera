"""Original human sources; stable group partitions and traceable pair construction."""

import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .data import OUT, RAW, ROOT, RUN, digest, normalized, read, sha, write

CORPUS = ROOT / "runs/HC-corpus-001"
DOMAINS = ("dictionary", "sentences", "conversation", "textbook")
CAPS = {"dictionary": 12000, "sentences": 12000, "conversation": 8000, "textbook": 4000}


def partition(group):
    residue = int(digest("HC-001:" + group), 16) % 10
    return "final" if residue == 0 else "dev" if residue == 1 else "train"


def pair(identifier, group, question, target, source, locator):
    return {"id": identifier, "group": group, "question": question, "target": target,
            "source": source, "locator": locator}


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError("Original human source bytes changed: " + str(path))


def dictionary(splits):
    path = RAW / "wordnet-3.0.tar.gz"
    verified(path, "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52")
    with tarfile.open(path) as archive:
        license_text = archive.extractfile("WordNet-3.0/LICENSE").read().decode()
        (CORPUS / "WORDNET-LICENSE.txt").write_text(license_text, encoding="utf-8")
        for pos in ("noun", "verb", "adj", "adv"):
            member = f"WordNet-3.0/dict/data.{pos}"
            for line in archive.extractfile(member).read().decode().splitlines():
                if not line or line[0].isspace() or " | " not in line:
                    continue
                fields, gloss = line.split(" | ", 1)
                fields = fields.split()
                group = f"wordnet:{pos}:{fields[0]}"
                if partition(group) not in splits:
                    continue
                n = int(fields[3], 16)
                lemmas = [re.sub(r"\([a-z]+\)$", "", fields[4+2*i]).replace("_", " ") for i in range(n)]
                definition = gloss.split('; "', 1)[0].strip()
                examples = re.findall(r'"([^\"]+)"', gloss)
                target = "; ".join(lemmas)
                for i, text in enumerate([definition] + examples):
                    if 3 <= len(text.split()) <= 96 and 1 <= len(target.split()) <= 48:
                        yield pair(f"{group}:{i}", group, text, target,
                                   "https://wordnet.princeton.edu/", member+":"+fields[0])


def conversation(splits):
    path = RAW / "ami-manual-1.6.2.zip"
    verified(path, "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d")
    meetings = defaultdict(list)
    ns = "{http://nite.sourceforge.net/}"
    with zipfile.ZipFile(path) as archive:
        for member in sorted(n for n in archive.namelist() if n.endswith(".segments.xml")):
            stem = Path(member).name.removesuffix(".segments.xml")
            meeting = stem.split(".")[0]
            group = "ami:" + re.sub(r"[a-d]$", "", meeting)
            if partition(group) not in splits:
                continue
            word_member = "words/" + stem + ".words.xml"
            nodes = list(ET.fromstring(archive.read(word_member)))
            index = {n.attrib[ns+"id"]: i for i, n in enumerate(nodes)}
            for segment in ET.fromstring(archive.read(member)):
                child = segment.find(ns+"child")
                if child is None:
                    continue
                ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
                if not ids or any(i not in index for i in ids):
                    raise ValueError("Manual transcript pointer did not resolve")
                selected = nodes[index[ids[0]]:index[ids[-1]]+1]
                text = " ".join(n.text for n in selected if n.tag == "w" and n.text)
                if 4 <= len(text.split()) <= 96:
                    meetings[meeting].append((float(segment.attrib["transcriber_start"]),
                                              segment.attrib[ns+"id"], text, member, group))
        for meeting, rows in sorted(meetings.items()):
            rows.sort()
            for first, second in zip(rows, rows[1:]):
                # Retain temporal overlap; recorded adjacency is an association label, not causation.
                yield pair("ami:"+meeting+":"+first[1], first[4], first[2], second[2],
                           "https://groups.inf.ed.ac.uk/ami/corpus/", [first[3]+"#"+first[1], second[3]+"#"+second[1]])


def textbook(splits):
    folder = ROOT / "runs/HR-corpus/openstax"
    receipt = read(ROOT / "runs/HR-corpus/openstax-receipt.json")
    records = {r["path"]: r for r in receipt["records"]}
    collection = folder / "collections/physics.collection.xml"
    ns = {"c": "http://cnx.rice.edu/collxml", "x": "http://cnx.rice.edu/cnxml"}
    mapping = {}
    root = ET.parse(collection).getroot()
    for chapter, section in enumerate(root.findall("./c:content/c:subcollection", ns)):
        for module in section.findall(".//c:module", ns):
            mapping[module.attrib["document"]] = "physics:chapter:"+str(chapter)
    if not mapping:
        raise ValueError("Textbook chapter partition is missing")
    def paragraphs(node):
        if node.tag.split("}")[-1] in {"exercise", "solution", "note", "figure", "table", "glossary", "footnote"}:
            return
        if node.tag == "{http://cnx.rice.edu/cnxml}para":
            # Mathematical markup is not flattened into misleading prose.
            if any("MathML" in n.tag for n in node.iter()):
                return
            text = " ".join("".join(node.itertext()).split())
            if 8 <= len(text.split()) <= 96:
                yield node.attrib.get("id", ""), text
            return
        for child in node:
            yield from paragraphs(child)
    for module, group in sorted(mapping.items()):
        if partition(group) not in splits:
            continue
        path = folder / "modules" / module / "index.cnxml"
        receipt_row = records[path.relative_to(ROOT).as_posix()]
        verified(path, receipt_row["sha256"])
        tree = ET.parse(path).getroot()
        content = tree.find("x:content", ns)
        if content is None:
            continue
        # Pair only within each top-level exposition section; never across modules.
        for section in content:
            rows = list(paragraphs(section))
            for a, b in zip(rows, rows[1:]):
                yield pair("physics:"+module+":"+a[0], group, a[1], b[1],
                           receipt_row["url"], [a[0], b[0]])


def sentence_pairs(split):
    path = RUN / (split+".json")
    for row in read(path):
        yield pair(row["id"], "squad:"+row["title"], row["question"],
                   row["sentences"][row["gold"][0]]["text"], row["source"],
                   {"question_id": row["id"], "context_sha256": row["context_sha256"]})


def prepare(final=False):
    CORPUS.mkdir(parents=True, exist_ok=True)
    if final and not (OUT / "curriculum-selection.json").exists():
        raise ValueError("Select before opening final cohorts")
    splits = ("final",) if final else ("train", "dev")
    counts, rejected, hashes = {}, Counter(), {}
    for domain, builder in (("dictionary", dictionary), ("sentences", None),
                            ("conversation", conversation), ("textbook", textbook)):
        groups = {s: [] for s in splits}
        if builder:
            for row in builder(set(splits)):
                groups[partition(row["group"])].append(row)
        else:
            for split in splits:
                groups[split] = list(sentence_pairs(split))
        forbidden, seen = set(), set()
        if final:
            for split in ("train", "dev"):
                for row in read(CORPUS / f"{domain}-{split}.json"):
                    forbidden.update((normalized(row["question"]), normalized(row["target"])))
        for split in splits:
            chosen = []
            partition_text = set()
            for row in sorted(groups[split], key=lambda x: digest(x["id"])):
                q, t = normalized(row["question"]), normalized(row["target"])
                if q == t or (q, t) in seen or q in forbidden or t in forbidden:
                    rejected[domain+":duplicate_or_overlap"] += 1
                    continue
                seen.add((q, t))
                chosen.append(row)
                partition_text.update((q, t))
                if len(chosen) == (CAPS[domain] if split == "train" else 400):
                    break
            forbidden.update(partition_text)
            destination = CORPUS / f"{domain}-{split}.json"
            if destination.exists():
                raise FileExistsError("Prepared source pairs are immutable")
            if len(chosen) < 8:
                raise ValueError(f"Insufficient {domain}/{split} source pairs")
            write(destination, chosen)
            key = domain+":"+split
            counts[key] = {"pairs": len(chosen), "groups": len({r["group"] for r in chosen})}
            hashes[key] = sha(destination)
    result = {"counts": counts, "rejections": dict(rejected), "hashes": hashes,
              "source": sha(Path(__file__)), "protocol": sha(OUT / "CURRICULUM.md")}
    write(CORPUS / ("final-manifest.json" if final else "manifest.json"), result)
    print(json.dumps({"counts": counts, "rejections": dict(rejected)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    prepare(parser.parse_args().final)
