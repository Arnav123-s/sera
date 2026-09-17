"""Read one English member from the original public MASSIVE tar stream."""

import hashlib
import json
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/intake/massive-1.1-en-US-20260916"
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"


class BoundedReader:
    def __init__(self, response):
        self.response, self.count = response, 0
        self.hash = hashlib.sha256()

    def read(self, size):
        value = self.response.read(min(size, 1048576))
        self.count += len(value)
        if self.count > 80 * 1024 * 1024:
            raise ValueError("Finite 80 MiB compressed intake limit reached")
        self.hash.update(value)
        return value


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = {}
    with urllib.request.urlopen(URL, timeout=30) as response:
        reader = BoundedReader(response)
        headers = {k: response.headers.get(k) for k in ("ETag", "Content-Length", "Last-Modified")}
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                basename = member.name.rsplit("/", 1)[-1]
                if basename not in ("LICENSE", "LICENSE.txt", "en-US.jsonl"):
                    continue
                if not member.isfile() or member.size > 16 * 1024 * 1024:
                    raise ValueError("Unexpected selected archive member")
                destination = "LICENSE.txt" if basename.startswith("LICENSE") else basename
                if destination in files:
                    raise ValueError("Duplicate selected archive member")
                data = archive.extractfile(member).read()
                (OUT / destination).write_bytes(data)
                files[destination] = {"member": member.name, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}
                if "LICENSE.txt" in files and "en-US.jsonl" in files:
                    break
        received = reader.count
        prefix_hash = reader.hash.hexdigest()
    if "en-US.jsonl" not in files or "LICENSE.txt" not in files:
        raise ValueError("Missing English data or original license; retain partial intake")
    counts = Counter()
    outputs = {part: (OUT / f"{part}.jsonl").open("xb") for part in ("train", "dev", "test")}
    try:
        with (OUT / "en-US.jsonl").open("rb") as source:
            for raw in source:
                row = json.loads(raw)
                if row["locale"] != "en-US" or row["partition"] not in outputs:
                    raise ValueError("Unexpected source locale or partition")
                outputs[row["partition"]].write(raw)
                counts[row["partition"]] += 1
    finally:
        for output in outputs.values():
            output.close()
    for partition in outputs:
        data = (OUT / f"{partition}.jsonl").read_bytes()
        files[f"{partition}.jsonl"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"schema": "sera.massive-intake.1", "url": URL, "headers": headers,
                "source_name": "MASSIVE 1.1; English seed utterances from SLURP",
                "license": "CC BY 4.0", "files": files, "rows": dict(counts),
                "downloaded_compressed_prefix_bytes": received,
                "downloaded_compressed_prefix_sha256": prefix_hash,
                "whole_archive_downloaded": received == int(headers["Content-Length"]),
                "dataset_commands_executed": False,
                "cost_boundary": "Static source retrieval and partitioning, outside numerical worker; wall measured below.",
                "seconds": time.perf_counter() - started,
                "test_access": "Quarantined by source partition; no test examples printed or supplied to pilot."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
