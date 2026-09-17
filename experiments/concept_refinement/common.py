"""Stage-local lineage, artifact and source contracts."""

import hashlib
import json
from pathlib import Path

from experiments.self_study.runtime import StudySession

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/28_concept_refinement"
EXPERIMENT = OUT / "r2"
RUNS = ROOT / "runs/CR-study-r2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contracts():
    folder = Path(__file__).parent
    return {
        "parent": sha(OUT / "parent-study.json"),
        "protocol": sha(OUT / "PROTOCOL.md"),
        "repair": sha(OUT / "F01-simple-initial-condition.md"),
        **{name: sha(folder / name) for name in ("data.py", "model.py", "physical.py")},
    }


def restore_parent():
    saved = read(OUT / "parent-study.json")
    return StudySession(saved["parent"], saved)
