"""Stage the human sources, cut them into groups and freeze the partitions."""

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate

activate()

from experiments.owner_language.corpus import build  # noqa: E402


def main():
    manifest = build()
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
