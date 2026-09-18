"""Reserve existing local numerical allowance for sustained retained practice."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.sustained_refinement", "scripts.refinement_audit",
                      "scripts.refinement_artifacts", "scripts.continuing_use", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
