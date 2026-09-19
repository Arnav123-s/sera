"""Restore the current SERA learner and its verified predecessor artifacts."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [
        ["scripts/restore_test_artifacts.py"],
        ["scripts/reading_artifacts.py", "--restore"],
        ["scripts/books_artifacts.py", "--restore"],
        ["scripts/completion_artifacts.py", "--restore"],
        ["scripts/quest_artifacts.py", "--restore"],
        ["scripts/quest_retention_artifacts.py"],
        ["scripts/learning_artifacts.py"],
        *[["-m", "scripts." + name] for name in (
            "discovery_artifacts", "autonomous_artifacts", "acquisition_artifacts", "growth_artifacts")],
        ["-m", "scripts.refinement_artifacts", "--stage", "39"],
        ["-m", "scripts.refinement_artifacts", "--stage", "40"],
        ["-m", "scripts.self_chosen_artifacts"],
        ["-m", "scripts.frontier_artifacts"],
        ["-m", "scripts.gap_artifacts", "--restore"],
        ["-m", "scripts.solution_artifacts", "--restore"],
        ["-m", "scripts.counterfactual_artifacts", "--restore"],
        ["-m", "scripts.structural_artifacts", "--restore"],
        ["-m", "scripts.intervention_artifacts", "--restore"],
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    for index, command in enumerate(commands, 1):
        print(f"Restore {index}/{len(commands)}: {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)
    print("The current learner is restored. Run scripts/sera_current.py with a task JSON and a fresh output path.")


if __name__ == "__main__":
    main()
