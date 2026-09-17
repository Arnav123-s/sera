"""Run request learning within the existing finite resource ledger."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.stream_curriculum.study", "experiments.stream_curriculum.prepare",
                      "experiments.stream_curriculum.audit", "experiments.stream_curriculum.runtime",
                      "experiments.stream_curriculum.facts", "experiments.stream_curriculum.evaluate",
                      "experiments.stream_curriculum.mixed", "experiments.stream_curriculum.sequential",
                      "experiments.stream_curriculum.batch", "experiments.stream_curriculum.report",
                      "experiments.stream_curriculum.integrate", "pytest", "ruff"}

if __name__ == "__main__":
    supervisor.main()
