"""Freeze the held-out evaluation sets cut from the human sources."""

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate

activate()

from experiments.owner_language.tasks import build_evaluation  # noqa: E402


def main():
    print(json.dumps(build_evaluation(), indent=2))


if __name__ == "__main__":
    main()
