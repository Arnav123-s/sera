"""Apply the saved request learner to a local JSONL task inbox."""

import argparse
import json
from pathlib import Path

import torch

from workbench.storage import Store

from .data import ROOT, sha, write
from .runtime import RequestSession


def annotate(session, source, destination):
    source, destination = Path(source), Path(destination)
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Use an input chunk of at most 2 MiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    identities, predictions, errors = set(), 0, 0
    with source.open(encoding="utf-8") as file, destination.open("x", encoding="utf-8", newline="\n") as output:
        for index, line in enumerate(file, 1):
            if index > 10000:
                raise ValueError("Use at most 10,000 requests per chunk; completed rows remain preserved")
            try:
                row = {"text": line.rstrip("\r\n")} if source.suffix.lower() == ".txt" else json.loads(line)
                if not isinstance(row, dict) or set(row) - {"id", "text"}:
                    raise ValueError("Use a JSON object containing text and optional id")
                identifier = row.get("id", str(index))
                if not isinstance(identifier, str) or not identifier or identifier in identities:
                    raise ValueError("Request IDs must be distinct nonempty strings")
                if not isinstance(row.get("text"), str):
                    raise ValueError("Request text must be a string")
                frame = session.interpret(row["text"])
                result = {"id": identifier, "status": "predicted", **frame}
                identities.add(identifier)
                predictions += 1
            except (ValueError, TypeError) as error:
                result = {"line": index, "status": "input_error", "message": str(error)}
                errors += 1
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {"predictions": predictions, "input_errors": errors,
            "input_format": "one request per line" if source.suffix.lower() == ".txt" else "JSONL",
            "source_sha256": sha(source),
            "output_sha256": sha(destination), "checkpoint": session.checkpoint,
            "actions_executed": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-requests-live")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Write annotation output inside the local runs directory")
    record = Store(args.store).read()
    if record is None:
        raise ValueError("Initialize a saved request learner first")
    session = RequestSession(record["checkpoint"], record)
    before = session.snapshot()
    result = annotate(session, args.input, args.output)
    if session.snapshot() != before:
        raise AssertionError("Annotation unexpectedly changed the learner")
    write(args.output.with_suffix(".receipt.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
