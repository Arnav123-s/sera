"""Qualification uses the same original numerical allowance and process-tree cap."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "workbench.qualification"}

if __name__ == "__main__":
    supervisor.main()
