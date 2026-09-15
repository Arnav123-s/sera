"""Account for output-compatible replay of the original frozen checker."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.audit_compatibility.transfer"}

if __name__ == "__main__":
    supervisor.main()
