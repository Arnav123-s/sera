"""Static source intake and frozen splits; no dataset scripts are executed."""
import hashlib
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from experiments.stream_curriculum.data import parse
from workbench.storage import Store
from .sources import ROOT

OUT = ROOT / "research-continuation/27_self_study"
DATA = ROOT / "runs/SS-language-data"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parent = OUT / "parent-request.json"
    if not parent.exists():
        write(parent, Store(ROOT / "runs/sera-requests-live").read())
    DATA.mkdir(exist_ok=False)
    started = time.perf_counter()
    # Direct publisher archive identified by the Hugging Face dataset card.
    url = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    locale_names = ("es-ES", "fr-FR", "de-DE")
    wanted = {f"1.1/data/{locale}.jsonl": locale for locale in locale_names}
    wanted["1.1/LICENSE"] = "license"
    source_files = {}
    with urlopen(Request(url, headers={"User-Agent": "SERA-Research/0.1"}), timeout=30) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                if not member.isfile() or member.size > 20 * 1024 * 1024:
                    raise ValueError("Unexpected teaching source size/type")
                raw = archive.extractfile(member).read()
                name = wanted[member.name]
                sha = hashlib.sha256(raw).hexdigest()
                source_files[name] = {"member": member.name, "bytes": len(raw), "sha256": sha}
                if name == "license":
                    (DATA / "LICENSE.txt").write_bytes(raw)
                    continue
                partitions = {p: [] for p in ("train", "development", "final")}
                rejected = []
                # Official test records are discarded without interpreting their contents.
                # New-language dev IDs are split deterministically before any model scoring.
                for line in raw.splitlines():
                    item = json.loads(line)
                    if item["partition"] not in {"train", "dev"}:
                        continue
                    partition = "train" if item["partition"] == "train" else (
                        "development" if int(item["id"]) % 2 == 0 else "final")
                    if len(partitions[partition]) >= (1200 if partition == "train" else 160):
                        continue
                    try:
                        row = parse(item)
                        row["locale"], row["source_id"] = name, row["id"]
                        row["id"] = name + ":" + str(row["id"])
                        row["partition"] = partition
                        partitions[partition].append(row)
                    except ValueError as error:
                        rejected.append({"id": item["id"], "reason": str(error)})
                for partition, rows in partitions.items():
                    with (DATA / f"{name}-{partition}.jsonl").open("x", encoding="utf-8") as stream:
                        for row in rows:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                source_files[name].update(rows={p: len(r) for p, r in partitions.items()}, rejected=rejected)
    if set(source_files) != set(locale_names) | {"license"}:
        raise ValueError("Incomplete language acquisition")
    manifest = {"url": url, "discovery": "https://huggingface.co/datasets/AmazonScience/massive",
                "license": "CC BY 4.0", "utc": datetime.now(timezone.utc).isoformat(),
                "source_files": source_files, "seconds": time.perf_counter() - started,
                "prepared": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                      "bytes": p.stat().st_size} for p in sorted(DATA.glob("*.jsonl"))},
                "scope": "Translation learning on supplied MASSIVE intent/entity labels. Parallel semantic IDs can have been seen in prior English training/development; this is not new-domain semantic transfer. Official test records unused."}
    write(DATA / "manifest.json", manifest)
    write(OUT / "language-sources.json", manifest)
    # Small primary source snapshots remain local; publish only provenance and derived lessons.
    for key, source_url in (
        ("calculus", "https://openstax.org/books/calculus-volume-1/pages/4-10-antiderivatives"),
        ("rh", "https://www.claymath.org/wp-content/uploads/2022/05/riemann.pdf"),
    ):
        with urlopen(Request(source_url, headers={"User-Agent": "SERA-Research/0.1"}), timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("Primary source exceeds cap")
        (DATA / f"{key}.source").write_bytes(raw)
        write(OUT / f"{key}-source.json", {"url": source_url, "sha256": hashlib.sha256(raw).hexdigest(),
                                          "bytes": len(raw), "utc": datetime.now(timezone.utc).isoformat(),
                                          "reading": "Engineering source review; separate from learned SERA capability"})
    print(json.dumps({"language_rows": {k: v.get("rows") for k, v in source_files.items()},
                      "source_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
