"""Materialise and verify the laboratory copy of the current owner's lineage."""

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate, identity

activate()

from experiments.owner_language.labstore import materialise  # noqa: E402

STORES = ("runs/sera-acquisition-live", "runs/sera-autonomous-live", "runs/sera-completion-live",
          "runs/sera-composed-discovery-live", "runs/sera-constraints", "runs/sera-counterfactual-live",
          "runs/sera-counterfactual-qualified-live", "runs/sera-discovery-live", "runs/sera-gap-inquiry-live",
          "runs/sera-grounded-live", "runs/sera-growth-live", "runs/sera-inquiry", "runs/sera-learning-live",
          "runs/sera-observed-discovery-live", "runs/sera-quests-live", "runs/sera-reading-live",
          "runs/sera-refinement-live", "runs/sera-requests", "runs/sera-requests-live",
          "runs/sera-self-discovery-live", "runs/sera-solution-progress-live", "runs/sera-solutions-live",
          "runs/sera-structural-development", "runs/sera-structural-live", "runs/sera-study-live",
          "runs/sera-task-transfer", "runs/sera-workbench")

# Immutable non-store artefacts the restore chain reads.  Each was discovered by
# executing the restore against the laboratory root and copying only the file the
# loader actually asked for, so this list is evidence of what the chain touches.
AUXILIARY = ("runs/A06-integrated-001/solver/current.json",
             "runs/A06-integrated-001/solver/versions/v0.json",
             "runs/A06-integrated-001/solver/versions/v0.pt",
             "runs/CF-study-001/language-public.json",
             "runs/CI-fits-final-001/selected.pt",
             "runs/CI-study-001/frontier.json",
             "runs/GD-study-001/teaching.json",
             "runs/KG-study-001/development.json",
             "runs/GD-study-002/step-0160.pt",
             "runs/HB-study-002/science/step-0120.pt",
             "runs/HC-study-002/textbook/step-0300.pt",
             "runs/HR-study-001/data-manifest.json",
             "runs/HR-study-001/lexical-2902/step-0420.pt",
             "runs/LI-fits-final-001/shared-2402/step-0100.pt",
             "runs/SC-data-002/manifest.json",
             "runs/SC-exposure-shared-2642/step-005400.pt",
             "runs/SD-study-001/language-public.json",
             "runs/SD-study-001/observation.csv",
             "runs/SS-rehearsal-2721/step-002000.pt",
             "runs/VC-study-001/3212/step-0800.pt")


def main():
    record = materialise(STORES, AUXILIARY)
    print(json.dumps({"stores": len(record["copied_stores"]), "auxiliary": len(record["copied_files"]),
                      "total_bytes": record["total_bytes"],
                      "baseline_current": record["baseline"]["current"],
                      "process": identity()["modules"]["sera"]}, indent=2))


if __name__ == "__main__":
    main()
