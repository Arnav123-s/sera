"""Explicit bounded entry point for the frozen independent applicability audit."""

import os
from pathlib import Path

from .applicability_audit import audit


def test_independent_audit():
    root = Path(__file__).resolve().parents[2]
    run = Path(os.environ["SERA_GUARD_RUN"]).resolve()
    output = Path(os.environ["SERA_GUARD_AUDIT_OUTPUT"]).resolve()
    if not run.is_relative_to(root/"runs") or not output.is_relative_to(root/"runs"):
        raise ValueError("Local audit paths must stay within runs")
    audit(run, output)
