"""Independent arithmetic/output audit uses the same cumulative resource allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.grounded_audit", "pytest", "workbench.qualify_language", "workbench.math_cycle", "workbench.audit_contract"}

if __name__ == "__main__":
    supervisor.main()
