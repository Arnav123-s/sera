"""Verified-completion jobs under the existing one-worker resource contract."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.verified_completion.packet", "experiments.verified_completion.study",
    "experiments.verified_completion.runtime", "experiments.verified_completion.audit",
    "scripts.completion_artifacts", "pytest", "ruff",
}

if __name__ == "__main__":
    supervisor.main()
