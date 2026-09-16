"""Owned grounded-language jobs use the explicitly renewed cumulative allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.grounded_language.study"}

if __name__ == "__main__":
    supervisor.main()
