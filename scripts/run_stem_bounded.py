"""Textbook capability acquisition within the continuing numerical allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.stem_learning.intake", "experiments.stem_learning.study",
    "experiments.stem_learning.runtime", "experiments.stem_learning.audit",
    "scripts.stem_artifacts", "pytest", "ruff",
}

if __name__ == "__main__":
    supervisor.main()
