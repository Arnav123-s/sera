"""Static train/dev preparation without loading the numerical libraries."""

import json

from workbench.storage import Store

from .data import PREPARED, ROOT, prepare, read, write


def main():
    path = ROOT / "research-continuation/26_stream_curriculum/parent.json"
    current = Store(ROOT / "runs/sera-constraints").read()
    if path.exists():
        if read(path) != current:
            raise ValueError("Pinned and current parents differ; reconcile before preparation")
    else:
        write(path, current)
    manifest = prepare()
    print(json.dumps({"rows": manifest["rows"], "rejected": len(manifest["rejected"]),
                      "intents": len(read(PREPARED / "vocabulary.json")["intents"]),
                      "slot_tags": len(read(PREPARED / "vocabulary.json")["tags"])}))


if __name__ == "__main__":
    main()
