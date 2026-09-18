"""Use the existing owned-worker allowance for verified step resolution."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.refinement_resolution", "scripts.resolution_audit",
                      "scripts.refinement_artifacts", "scripts.continuing_use", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
