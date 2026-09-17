"""Account for all constraint integration work under the existing allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.constraint_inquiry.packet", "experiments.constraint_inquiry.study",
    "experiments.constraint_inquiry.runtime", "experiments.constraint_inquiry.evaluate",
    "experiments.constraint_inquiry.audit", "pytest", "ruff",
    "experiments.constraint_inquiry.followup",
    "experiments.constraint_inquiry.report",
}

if __name__ == "__main__":
    supervisor.main()
