"""Charge concept-refinement work to the existing local resource ledger."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {
    "experiments.concept_refinement.packet_verify",
    "experiments.concept_refinement.packet_finish",
    "experiments.concept_refinement.packet_portability",
    "experiments.concept_refinement.study",
    "experiments.concept_refinement.runtime",
    "experiments.concept_refinement.audit",
    "experiments.concept_refinement.reporting",
    "experiments.concept_refinement.validation",
    "pytest",
    "ruff",
}

if __name__ == "__main__":
    supervisor.main()
