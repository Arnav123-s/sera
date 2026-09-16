"""Continue using the original cumulative local allowance and exclusive job lock."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.continuing_control.study", "experiments.continuing_control.audit",
                      "experiments.continuing_control.integration"}

if __name__ == "__main__":
    supervisor.main()
