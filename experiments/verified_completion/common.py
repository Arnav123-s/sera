"""Stable paths and canonical identities for the versioned completion extension."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/32_verified_completion"
RUN = ROOT / "runs/VC-study-001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_hash(model):
    value = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value.update(key.encode())
        value.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def contracts():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            if p.name in {"model.py", "task.py", "credit.py", "runtime.py"}} | {"protocol": sha(OUT / "PROTOCOL.md")}
