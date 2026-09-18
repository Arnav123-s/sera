"""One owned bounded CPU job for complete self-chosen solution portfolios."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"scripts.portable_gap_use", "scripts.solution_study", "scripts.solution_use",
                      "scripts.solution_artifacts", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
