"""Measure the packet regeneration discrepancy without fitting any model."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/intake/v18-understanding/SERA_v18"
sys.path.insert(0, str(PACKET / "src"))
from core import FAMILIES, SEEDS, dataset  # noqa: E402 -- isolated preserved packet import

rows = []
for family in FAMILIES:
    for seed in SEEDS:
        data, _ = dataset(family, seed)
        with np.load(PACKET / f"results/operators/{family}-{seed}-gru/data.npz") as archive:
            for key, value in data.items():
                saved = archive[key]
                rows.append({"family": family, "seed": seed, "array": key,
                             "differing_elements": int(np.count_nonzero(value != saved)),
                             "elements": value.size, "max_abs_difference": float(np.max(np.abs(value-saved)))})
out = Path(sys.argv[sys.argv.index("--output")+1])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(rows, indent=2) + "\n")
print(json.dumps([r for r in rows if r["max_abs_difference"] > 1e-14], indent=2))
