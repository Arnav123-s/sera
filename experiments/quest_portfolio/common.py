"""Versioned quest contracts and exact resource-independent records."""

from pathlib import Path

from experiments.verified_completion.common import ROOT, digest, model_hash, read, sha, write

OUT = ROOT / "research-continuation/33_capability_portfolio"
RUN = ROOT / "runs/QP-study-001"


def contracts():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            if p.name in {"model.py", "methods.py", "board.py", "runtime.py", "assessor.py"}} | {"protocol": sha(OUT / "PROTOCOL.md")}


__all__ = ["ROOT", "OUT", "RUN", "digest", "model_hash", "read", "sha", "write", "contracts"]
