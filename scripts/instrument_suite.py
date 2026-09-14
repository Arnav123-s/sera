"""Replicate the real/complex instrument experiment across three seeds."""

from pathlib import Path

import torch

from sera.experiments import instrument_experiment
from sera.storage import write_json

torch.set_num_threads(1)
root = Path("runs/instrument-suite-v1")
if root.exists():
    raise FileExistsError("Use a fresh experiment directory")
root.mkdir(parents=True)
results = []
for complex_valued in (False, True):
    for seed in (0, 1, 2):
        model, result = instrument_experiment(seed=seed, steps=300, complex_valued=complex_valued)
        name = f"{'complex' if complex_valued else 'real'}-{seed}"
        write_json(root / f"{name}.json", result)
        torch.save(
            {"state_dict": model.state_dict(), "complex": complex_valued}, root / f"{name}.pt"
        )
        results.append(result)
        print(f"{name}: length-24 NLL {result['length24_nll']:.5f}", flush=True)
write_json(root / "summary.json", {"runs": results})
