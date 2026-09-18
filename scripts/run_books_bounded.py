"""Human-book jobs within the existing numerical allowance."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.book_learning.data", "experiments.book_learning.study",
    "experiments.book_learning.grounding", "experiments.book_learning.audit",
    "experiments.book_learning.grounding_v2", "experiments.book_learning.runtime",
    "scripts.books_artifacts",
    "pytest", "ruff",
}

if __name__ == "__main__":
    supervisor.main()
