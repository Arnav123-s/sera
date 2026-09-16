"""One supervised application transaction. Never mutates a research checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .model import transact
from .storage import encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.workspace.mkdir(parents=True, exist_ok=True)
    lock = args.workspace/"transaction.lock"
    with lock.open("x") as file:
        file.write(str(args.request.resolve()))
    try:
        result = transact(args.workspace, json.loads(args.request.read_text()))
        with args.response.open("xb") as file:
            file.write(encoded(result))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
