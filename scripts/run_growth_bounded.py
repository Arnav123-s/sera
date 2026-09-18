"""Use the existing one-worker, time-reserved, 2 GiB continuation supervisor."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.continuing_growth.study", "experiments.continuing_growth.audit",
                      "experiments.continuing_growth.use", "scripts.growth_artifacts", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
