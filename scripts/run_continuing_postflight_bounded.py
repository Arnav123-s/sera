"""Read-only postflight under the same original aggregate resource allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.continuing_postflight"}

if __name__ == "__main__":
    supervisor.main()
