"""Postflight parameter audit under the same preserved worker allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.transfer_postflight"}

if __name__ == "__main__":
    supervisor.main()
