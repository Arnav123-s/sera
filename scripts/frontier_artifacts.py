"""Preserve the composed discovery successor and all comparison evidence."""

import argparse
import json

import torch

from experiments.discovery_frontier import OUT, ROOT, RUN, read, write
from scripts.refinement_artifacts import costs, seal, verify
from workbench.storage import Store


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seal", action="store_true")
    p.add_argument("--costs", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.seal:
        saved = Store(ROOT / "runs/sera-composed-discovery-live").read()
        if saved:
            path = RUN / "admitted-owner.json"
            if path.exists() and read(path) != saved:
                raise ValueError("Preserve the earlier exported frontier")
            write(path, saved)
        seal(42)
    if args.costs:
        costs(42)
    else:
        verify(42)
        final, audit = read(OUT / "final.json"), read(OUT / "audit.json")
        if not final["admitted"] or not audit["passed"] or audit["proposal_checks"] != 3456:
            raise ValueError("Frontier qualification disagrees with the publication")
        print(json.dumps({"selected": final["selected"], "independent_checks": audit["proposal_checks"]}))


if __name__ == "__main__":
    main()
