"""Quest jobs under the existing one-worker resource contract."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.quest_portfolio.packet", "experiments.quest_portfolio.study",
    "experiments.quest_portfolio.runtime", "experiments.quest_portfolio.audit",
    "scripts.quest_artifacts", "pytest", "ruff",
}

if __name__ == "__main__":
    supervisor.main()
