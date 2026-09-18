"""Standing-approved local continuation of the self-chosen discovery frontier."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.discovery_frontier", "scripts.frontier_audit", "scripts.frontier_artifacts",
                      "scripts.frontier_report", "experiments.discovery_observation", "scripts.continuing_use", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
