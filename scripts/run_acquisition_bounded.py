"""Follow the existing owned-worker ledger for prerequisite acquisition."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.autonomous_discovery.acquire", "scripts.acquisition_artifacts", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
