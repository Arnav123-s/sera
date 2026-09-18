"""Standing-approved owned CPU discovery jobs, with full resource accounting."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.self_chosen.study", "scripts.self_chosen_audit",
                      "experiments.discovery_observation",
                      "scripts.self_chosen_report",
                      "scripts.self_chosen_artifacts", "scripts.self_chosen_use", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()

