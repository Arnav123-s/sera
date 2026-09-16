"""Use the existing aggregate allowance for application and update-learning jobs."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "workbench.worker", "experiments.update_learning.study"}

if __name__ == "__main__":
    supervisor.main()
