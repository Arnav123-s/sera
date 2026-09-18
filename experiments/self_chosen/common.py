from pathlib import Path

from experiments.verified_completion.common import ROOT, digest, read, sha, write

OUT = ROOT / "research-continuation/41_self_chosen_discovery"
RUN = ROOT / "runs/SD-study-001"
SEEDS = (4101, 4102)
STEPS = 256
BLOCK = 32
ARMS = ("learned", "balanced")
LANGUAGES = ("en-US", "es-ES", "fr-FR", "de-DE")
DOMAINS = ("motion_0", "motion_1", "motion_2", "polynomials", *LANGUAGES)


def contracts():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))} | {
        "protocol": sha(OUT / "PROTOCOL.md"), "finite": sha(OUT / "FINITE.md")}


__all__ = ["ROOT", "OUT", "RUN", "SEEDS", "STEPS", "BLOCK", "ARMS", "LANGUAGES", "DOMAINS",
           "contracts", "digest", "read", "sha", "write"]
