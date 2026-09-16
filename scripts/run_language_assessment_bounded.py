"""Assess language successors and targeted lessons inside the same resource ledger."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.grounded_language.assessment"}

if __name__ == "__main__":
    supervisor.main()
