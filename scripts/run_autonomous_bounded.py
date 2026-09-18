"""Autonomous discovery under the same remaining local resource allowance."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.autonomous_discovery.runtime", "scripts.autonomous_artifacts", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
