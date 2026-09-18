"""Original human definitions, explicit type labels and a closed transfer cohort."""

import concurrent.futures
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

from experiments.human_reading.data import ROOT, digest, normalized, read, sha, write

from .data import OUT, dictionary_entries

GROUND = ROOT / "runs/GD-study-001"
COLLEGE = ROOT / "runs/HB-grounding-sources"
NS = {"c": "http://cnx.rice.edu/cnxml"}
ROLES = ("position", "velocity", "acceleration", "force", "mass", "other")
TERMS = {
    "position": {"position", "equilibrium position"},
    "velocity": {"velocity", "average velocity", "instantaneous velocity", "tangential velocity"},
    "acceleration": {"acceleration", "average acceleration", "instantaneous acceleration", "negative acceleration",
                     "constant acceleration", "acceleration due to gravity", "tangential acceleration"},
    "force": {"force", "net force", "net external force", "external force", "normal force", "restoring force",
              "weight", "friction", "static friction", "kinetic friction", "tension", "gravitational force",
              "electric force", "magnetic force", "centripetal force"},
    "mass": {"mass", "rest mass"},
}


def acquire():
    destination = COLLEGE / "receipt.json"
    if destination.exists():
        raise FileExistsError("Keep the original acquisition")
    commit = read(COLLEGE / "tree.json")["commit"]
    if commit != "fd1b25dfd5d8c6580c6e2b2b34a19e29cc69ada9":
        raise ValueError("College Physics pin changed")
    started = time.perf_counter()
    def fetch(name):
        url = f"https://raw.githubusercontent.com/openstax/osbooks-college-physics-bundle/{commit}/{name}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SERA-research"}), timeout=25) as response:
            data = response.read(8*1024*1024+1)
        if len(data) > 8*1024*1024:
            raise ValueError("Source exceeds acquisition bound")
        path = COLLEGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Pinned source changed")
        path.write_bytes(data)
        return {"path": path.relative_to(ROOT).as_posix(), "url": url, "sha256": sha(path), "bytes": len(data)}
    collection = fetch("collections/college-physics-2e.collection.xml")
    tree = ET.parse(ROOT / collection["path"])
    modules = list(dict.fromkeys(n.attrib["document"] for n in tree.iter() if n.tag.endswith("}module")))[:60]
    records = [collection, fetch("LICENSE"), fetch("README.md")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(fetch, [f"modules/{m}/index.cnxml" for m in modules]))
    receipt = {"title": "College Physics 2e", "publisher": "OpenStax / Rice University", "commit": commit,
               "records": records, "seconds": time.perf_counter()-started,
               "protocol_sha256": sha(OUT / "GROUNDING_PROTOCOL.md"), "definitions_opened": False}
    write(destination, receipt)
    print(json.dumps({"modules": len(modules), "bytes": sum(r["bytes"] for r in records), "seconds": receipt["seconds"]}))


def glossary(receipt):
    for item in receipt["records"]:
        if not item["path"].endswith(".cnxml"):
            continue
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("Definition source changed")
        root = ET.parse(path).getroot()
        for node in root.findall(".//c:glossary/c:definition", NS):
            term = " ".join(node.findtext("c:term", default="", namespaces=NS).casefold().split())
            meaning = node.find("c:meaning", NS)
            if meaning is None:
                continue
            text = " ".join("".join(meaning.itertext()).split())
            if len(text.split()) < 3:
                continue
            label = next((k for k, v in TERMS.items() if term in v), "other")
            yield {"id": path.parent.name+":"+node.attrib.get("id", digest(text)[:16]), "term": term,
                   "text": text, "label": label, "source": item["url"], "source_sha256": item["sha256"],
                   "text_sha256": digest(text), "module": path.parent.name}


def webster():
    source = next(s for s in read(OUT / "sources.json")["records"] if s["id"] == "webster")
    path = ROOT / source["path"]
    if sha(path) != source["sha256"]:
        raise ValueError("Original dictionary changed")
    text = path.read_text(encoding="utf-8-sig")
    # Original contiguous spans, never paraphrases. The labels are a supplied interface.
    selections = [
        ("position", "The spot where a person or thing is placed", "; as, the position"),
        ("velocity", "Rate of motion; the relation of motion to time", " See\nthe Note under Speed."),
        ("velocity", "Uniform velocity, velocity in which", "\n -- Variable velocity"),
        ("acceleration", "The act of accelerating, or the state of being accelerated;", "\nA period of social"),
        ("force", "Any action between two bodies which changes", "; as, the force of gravity"),
        ("mass", "The quantity of matter which a body contains", "\n\nNote: Mass and weight"),
        ("other", "A period of social improvement, or of intellectual advancement,", "\n(Astr. & Physics.)"),
        ("other", "To stuff; to lard; to farce.", "\nWit larded"),
        ("other", "A waterfall; a cascade.", "\nTo see the falls"),
        ("other", "Strength or energy of body or mind; active power;", "\nHe was, in the full force"),
        ("other", "Power exerted against will or consent; compulsory power;", "\nWhich now they hold"),
        ("other", "Strength or power war; hence, a body of land or naval combatants,", "\nIs Lucius general"),
        ("other", "The sacrifice in the sacrament of the Eucharist, or the", "\n\n2. (Mus.)"),
        ("other", "The portions of the Mass usually set to music, considered as a", " Canon of the Mass."),
        ("other", "To celebrate Mass.", "\n\nMASS"),
        ("other", "The ground which any one takes in an argument or", "\nLet not the proof"),
        ("other", "Relative place or standing; social or official rank;", "\n\n5. (Arith.)"),
    ]
    entries = list(dictionary_entries(text))
    for label, start, end in selections:
        a = text.index(start)
        b = text.index(end, a)
        value = text[a:b].strip()
        parent = next(e for e in entries if e["start"] <= a and e["end"] >= b)
        yield {"id": "webster:"+str(a), "term": parent["headword"].casefold(), "text": value,
               "label": label, "source": source["url"], "source_sha256": source["sha256"],
               "text_sha256": digest(value), "entry": parent, "span": [a, a+len(value)]}


def prepare(final=False):
    GROUND.mkdir(parents=True, exist_ok=True)
    destination = GROUND / ("final.json" if final else "teaching.json")
    if destination.exists():
        raise FileExistsError("Definition partitions are immutable")
    if final:
        if not (OUT / "ground-selection.json").exists():
            raise ValueError("Freeze the learned binding before opening transfer definitions")
        old = read(GROUND / "teaching.json")
        forbidden = {normalized(r["text"]) for p in ("train", "dev") for r in old[p]}
        rows, seen, counts = [], set(), defaultdict(int)
        for row in sorted(glossary(read(COLLEGE / "receipt.json")), key=lambda r: digest(r["id"])):
            key = normalized(row["text"])
            if key in forbidden or key in seen or (row["label"] == "other" and counts["other"] >= 40):
                continue
            seen.add(key)
            rows.append(row)
            counts[row["label"]] += 1
        write(destination, {"rows": rows, "counts": dict(counts), "selection": sha(OUT / "ground-selection.json"),
                            "source": sha(COLLEGE / "receipt.json")})
    else:
        groups = defaultdict(list)
        seen = set()
        for row in sorted(glossary(read(ROOT / "runs/HR-corpus/openstax-receipt.json")), key=lambda r: digest(r["id"])):
            key = normalized(row["text"])
            if key not in seen:
                seen.add(key)
                groups[row["label"]].append(row)
        train, dev = [], []
        for role in ROLES:
            entries = groups[role][:80] if role == "other" else groups[role]
            terms = sorted({r["term"] for r in entries}, key=lambda t: digest("GD-001:"+t))
            n = max(1, len(terms)//4) if len(terms) >= 2 else 0
            held = set(terms[:n])
            for row in entries:
                (dev if row["term"] in held else train).append(row)
        train += list(webster())
        write(destination, {"train": train, "dev": dev, "protocol": sha(OUT / "GROUNDING_PROTOCOL.md"),
                            "labels": {k: sorted(v) for k, v in TERMS.items()}})
        counts = {p: {k: sum(r["label"] == k for r in rows) for k in ROLES} for p, rows in (("train", train), ("dev", dev))}
    print(json.dumps(counts), flush=True)
