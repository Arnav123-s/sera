"""Complete human entries and prose, with original offsets and closed final groups."""

import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

from experiments.human_reading.data import ROOT, digest, normalized, read, sha, write
from experiments.human_reading.model import words

OUT = ROOT / "research-continuation/30_grounded_books"
RUN = ROOT / "runs/HB-study-002"
DOMAINS = ("dictionary", "conversation", "grammar", "philosophy", "mathematics", "science")


def split(group):
    v = int(digest("HB-001:"+group), 16) % 10
    return "final" if v == 0 else "dev" if v == 1 else "train"


def body(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    stop = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not stop or start.end() >= stop.start():
        raise ValueError("Original book body markers are required")
    return start.end(), stop.start()


def dictionary_entries(text):
    a, b = body(text)
    headers = list(re.finditer(r"(?m)^[A-Z][A-Z ;'\-]{0,78}$", text[a:b]))
    for i, header in enumerate(headers):
        start = a+header.start()
        end = a+(headers[i+1].start() if i+1 < len(headers) else b-a)
        term = header[0].split(";")[0].strip().casefold()
        entry = text[start:end].strip()
        if len(entry.split()) >= 8:
            yield {"group": "webster:"+term, "id": "webster:"+str(start), "headword": header[0],
                   "text": entry, "start": start, "end": end,
                   "source": "https://www.gutenberg.org/ebooks/29765"}


def prose(text, name, source):
    a, b = body(text)
    for i, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text[a:b])):
        value = match[0].strip()
        if len(value.split()) >= 17:
            yield {"id": name+":"+str(a+match.start()), "group": name+":"+str(i//32),
                   "text": value, "start": a+match.start(), "end": a+match.end(), "source": source}


def records(domain, wanted):
    sources = {s["id"]: s for s in read(OUT / "sources.json")["records"]}
    for source in sources.values():
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Human source changed: "+source["id"])
    if domain == "dictionary":
        yield from dictionary_entries((ROOT / sources["webster"]["path"]).read_text(encoding="utf-8-sig"))
    elif domain in {"grammar", "philosophy"}:
        names = ("grammar",) if domain == "grammar" else ("plato", "descartes")
        for name in names:
            yield from prose((ROOT / sources[name]["path"]).read_text(encoding="utf-8-sig"), name, sources[name]["url"])
    elif domain == "conversation":
        with zipfile.ZipFile(ROOT / sources["dailydialog"]["path"]) as archive:
            # Use the supplied official partitions, with no test access during teaching.
            for official, part in (("train", "train"), ("validation", "dev"), ("test", "final")):
                if part not in wanted:
                    continue
                raw = archive.read("EMNLP_dataset/"+official+".zip")
                with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
                    name = next(n for n in bundle.namelist() if n.endswith("dialogues_"+official+".txt"))
                    for i, line in enumerate(bundle.read(name).decode().splitlines()):
                        # Speaker boundaries are retained as newlines; original utterance tokens are unchanged.
                        text = "\n".join(t.strip() for t in line.split("__eou__") if t.strip())
                        yield {"id": f"daily:{official}:{i}", "group": f"daily:{official}:{i}", "partition": part,
                               "text": text, "source": sources["dailydialog"]["url"], "locator": name+":"+str(i+1)}
    elif domain == "mathematics":
        text = (ROOT / sources["calculus"]["path"]).read_text(encoding="utf-8-sig")
        text = text.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
        for i, paragraph in enumerate(re.split(r"\n\s*\n", text)):
            paragraph = re.sub(r"(?m)%[^\n]*", "", paragraph)
            pieces = re.split(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$]*\$", paragraph, flags=re.S)
            for j, piece in enumerate(pieces):
                if any(marker in piece for marker in (r"\begin", r"\def", r"\newcommand", r"\documentclass")):
                    continue
                value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?", " ", piece).replace("{", "").replace("}", "")
                if len(value.split()) >= 17:
                    yield {"id": f"calculus:{i}:{j}", "group": "calculus:"+str(i//32), "text": value,
                           "source": sources["calculus"]["url"], "locator": f"paragraph:{i}:prose-part:{j}"}
    else:
        receipt = read(ROOT / "runs/HR-corpus/openstax-receipt.json")
        for item in receipt["records"]:
            if not item["path"].endswith(".cnxml"):
                continue
            path = ROOT / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError("Physics source changed")
            root = ET.parse(path).getroot()
            for i, node in enumerate(root.iter("{http://cnx.rice.edu/cnxml}para")):
                if any("MathML" in n.tag for n in node.iter()):
                    continue
                text = " ".join("".join(node.itertext()).split())
                if len(text.split()) >= 17:
                    yield {"id": path.parent.name+":"+str(i), "group": "physics:"+path.parent.name,
                           "text": text, "source": item["url"], "locator": node.attrib.get("id", str(i))}


def prepare(final=False):
    if final and not (OUT / "selection.json").exists():
        raise ValueError("Model selection must precede final preparation")
    RUN.mkdir(parents=True, exist_ok=True)
    wanted = ("final",) if final else ("train", "dev")
    counts, identities, rejected = {}, {}, Counter()
    for domain in DOMAINS:
        groups = {s: [] for s in wanted}
        for record in records(domain, set(wanted)):
            part = record.get("partition", split(record["group"]))
            if part not in wanted:
                continue
            token = words(record["text"])
            for start in range(0, len(token)-16, 17):
                chunk = token[start:start+17]
                groups[part].append({"id": record["id"]+":"+str(start), "group": record["group"],
                                     "context": chunk[:16], "target": chunk[16], "source": record["source"],
                                     "record_id": record["id"], "text_sha256": digest(record["text"]),
                                     "token_offset": start, "headword": record.get("headword")})
        forbidden = set()
        if final:
            for part in ("train", "dev"):
                forbidden.update(" ".join(r["context"]+[r["target"]]) for r in read(RUN / f"{domain}-{part}.json"))
        for part in wanted:
            kept, seen = [], set()
            for row in sorted(groups[part], key=lambda r: digest(r["id"])):
                key = normalized(" ".join(row["context"]+[row["target"]]))
                if key in seen or key in forbidden:
                    rejected[domain] += 1
                    continue
                seen.add(key)
                kept.append(row)
                if len(kept) == (2000 if part == "train" else 160):
                    break
            forbidden.update(seen)
            path = RUN / f"{domain}-{part}.json"
            if path.exists():
                raise FileExistsError("Prepared teaching/evaluation data are immutable")
            if len(kept) < 32:
                raise ValueError("Insufficient human text windows")
            write(path, kept)
            counts[domain+":"+part] = {"windows": len(kept), "groups": len({r["group"] for r in kept})}
            identities[domain+":"+part] = sha(path)
    write(RUN / ("final-manifest.json" if final else "manifest.json"),
          {"counts": counts, "hashes": identities, "rejections": dict(rejected), "protocol": sha(OUT / "PROTOCOL.md")})
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    prepare(parser.parse_args().final)
