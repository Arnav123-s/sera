"""Owned internal-discovery work within the existing numerical allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.operator_discovery.runtime", "pytest", "ruff", "scripts.discovery_artifacts"}

if __name__ == "__main__":
    supervisor.main()
