"""Run the acquisition intervention audit under the existing owned-job allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.acquisition_dependence.study"}

if __name__ == "__main__":
    supervisor.main()
