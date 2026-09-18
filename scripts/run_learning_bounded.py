"""Learning-progress jobs under the existing owned-worker allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.learning_progress.study", "experiments.learning_progress.runtime",
    "experiments.learning_progress.audit", "scripts.learning_artifacts", "pytest", "ruff",
    "experiments.learning_progress.bridge",
}

if __name__ == "__main__":
    supervisor.main()
