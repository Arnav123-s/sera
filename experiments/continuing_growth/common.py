from pathlib import Path

from experiments.verified_completion.common import ROOT, digest, read, sha, write

OUT = ROOT / "research-continuation/38_continuing_growth"
RUN = ROOT / "runs/CG-study-001"
BOOKS = ("dictionary", "conversation", "grammar", "philosophy", "mathematics", "science")
LANGUAGES = ("en-US", "es-ES", "fr-FR", "de-DE")
SKILLS = (*BOOKS, "reading", *LANGUAGES, "methods")
PREFIXES = ("book_heads.shared.", "reading_head.", "stream_intent.", "quest_policy.")
RATES = (.0005, .002, .008)
ARMS = ("balanced", "self_directed")
SEEDS = (3801, 3802)
DECISIONS = 1536
BLOCK = 128
PRACTICE_STEPS = 4
BATCH = 16


def contracts():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))} | {
        "protocol": sha(OUT / "PROTOCOL.md")}


__all__ = ["ROOT", "OUT", "RUN", "BOOKS", "LANGUAGES", "SKILLS", "PREFIXES", "RATES",
           "ARMS", "SEEDS", "DECISIONS", "BLOCK", "PRACTICE_STEPS", "BATCH",
           "contracts", "digest", "read", "sha", "write"]
