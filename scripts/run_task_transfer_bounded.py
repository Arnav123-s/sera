"""Reserve every task-transfer numerical job against the existing local grant."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.task_transfer.study", "experiments.task_transfer.audit",
                      "experiments.task_transfer.runtime", "experiments.task_transfer.integration",
                      "experiments.task_transfer.report", "experiments.task_transfer.certify",
                      "experiments.task_transfer.migration_check", "pytest"}

if __name__ == "__main__":
    supervisor.main()
