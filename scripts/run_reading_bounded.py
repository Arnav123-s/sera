"""Charge human-reading acquisition and verification to the existing allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.human_reading.data",
    "experiments.human_reading.study",
    "experiments.human_reading.runtime",
    "experiments.human_reading.audit",
    "experiments.human_reading.curriculum",
    "experiments.human_reading.corpus",
    "scripts.reading_artifacts",
    "pytest",
    "ruff",
}

if __name__ == "__main__":
    supervisor.main()
