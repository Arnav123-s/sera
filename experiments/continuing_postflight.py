"""Fresh paired behavior check after the entire continuing owner is restored."""

import gzip
import json

import torch

from experiments.continuing_control.integration import restore
from experiments.continuing_control.study import RELEASE, parent, write
from experiments.cross_route_transfer.study import evaluate_retention
from sera.storage import canonical


def main():
    torch.set_num_threads(1)
    original = parent()
    successor, interaction, _ = restore()
    old_id, new_id = original.identity(), successor.identity()
    config = {"retention_samples": 128, "typed_samples": 128}
    before = evaluate_retention(original, 1200161, config)
    after = evaluate_retention(successor, 1200161, config)
    same = before["groups"] == after["groups"] and before["scores"] == after["scores"]
    if not same or original.identity() != old_id or successor.identity() != new_id:
        raise ValueError("Retained behavior changed or evaluation mutated an owner")
    output = RELEASE/"retention"
    output.mkdir(exist_ok=False)
    for name, value in (("before", before), ("after", after)):
        (output/(name+".json.gz")).write_bytes(gzip.compress(canonical(value).encode(), mtime=0))
    report = {"status": "PASS", "seed": 1200161, "config": config, "groups": len(before["groups"]),
              "group_scores_and_all_saved_case_scores_exact": same, "maximum_measured_drop": 0.,
              "original_solver": old_id, "new_solver": new_id, "interaction_observations": len(interaction.events),
              "new_registered_state_bytes": sum(t.numel()*t.element_size() for n, t in successor.components["r1"].state_dict().items()
                                                if n.startswith("interaction_")),
              "added_trainable_parameters": 0, "scope": "Fresh paired retained inputs; exact scores on this finite bank, not a population-wide guarantee"}
    write(output/"result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
