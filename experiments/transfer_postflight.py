"""Supplement frozen summaries with independent parameter/buffer preservation checks."""

import json
from pathlib import Path

import torch

from experiments.acquisition_dependence.study import load_before
from experiments.cross_route_transfer.study import ROOT, parent, read, sha, write
from experiments.instance_transfer.study import ARMS, RELEASE, SEEDS, contract
from sera.shared_archive import load_shared_checkpoint


def main():
    contract()
    after = parent().components["r1"]
    before, _ = load_before()
    checked = []
    for seed in SEEDS:
        for arm in ARMS:
            path = RELEASE/f"seed-{seed}"/arm/"result.json"
            record = read(path)
            model = load_shared_checkpoint(ROOT/record["selected_checkpoint"]["path"])
            original = before if arm == "uncorrected_teacher" else after
            reference = original.state_dict()
            changed = [name for name, value in model.state_dict().items() if not torch.equal(value, reference[name])]
            if not changed or any(not n.startswith("typed_numeric.") for n in changed):
                raise ValueError("Actual checkpoint changed more than the qualified readout")
            if sorted(changed) != sorted(record["training"]["changed_tensors"]):
                raise ValueError("Reported parameter change differs from saved bytes")
            # Compare tensor/config interfaces independently of the training report.
            actual_config, original_config = model.export_config(), original.export_config()
            actual_config.pop("generator_requires_grad", None)
            original_config.pop("generator_requires_grad", None)
            if actual_config != original_config:
                raise ValueError("Unreported executable configuration change")
            checked.append({"seed": seed, "arm": arm, "changed_tensors": changed,
                            "selected_checkpoint_sha256": sha(ROOT/record["selected_checkpoint"]["path"]),
                            "generator_buffers_bitwise_preserved": True,
                            "all_other_numerical_parameters_preserved": True})
    target = RELEASE/"parameter-preservation-audit.json"
    if target.exists():
        raise FileExistsError("Postflight receipt is immutable")
    write(target, {"status": "PASS", "checked_models": len(checked), "records": checked,
                   "source_sha256": sha(Path(__file__)),
                   "scope": "Byte-level comparison against each actual corrected/uncorrected predecessor, independently of reported change lists."})
    print(json.dumps({"status": "PASS", "checked_models": len(checked)}))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
