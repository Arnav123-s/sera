from pathlib import Path

from experiments.verified_completion.common import ROOT, digest, read, sha, write

OUT = ROOT / "research-continuation/34_learning_progress"
RUN = ROOT / "runs/LP-study-001"
SKILLS = ("dictionary", "conversation", "mathematics", "science", "reading", "methods")
PREFIXES = ("book_heads.shared.", "reading_head.", "quest_policy.")
RATES = (.0005, .002, .008)


def contracts():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            if p.name in {"common.py", "data.py", "model.py", "runtime.py"}} | {
        "protocol": sha(OUT / "PROTOCOL.md")}


__all__ = ["ROOT", "OUT", "RUN", "SKILLS", "PREFIXES", "RATES", "digest", "read", "sha", "write", "contracts"]
