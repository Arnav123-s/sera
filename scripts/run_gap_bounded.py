"""Standing-approved, single-worker investigation of retained knowledge gaps."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"scripts.gap_study", "scripts.gap_artifacts", "scripts.gap_use", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
