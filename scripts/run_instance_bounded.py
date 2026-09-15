"""Use the original local allowance for acquisition-dependent transfer."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.instance_transfer.study", "experiments.instance_transfer.audit"}

if __name__ == "__main__":
    supervisor.main()
