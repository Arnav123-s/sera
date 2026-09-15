"""Use the existing budget for qualification-gated experimental state migration."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.transfer_integration.study", "scripts.resume_transfer"}

if __name__ == "__main__":
    supervisor.main()
