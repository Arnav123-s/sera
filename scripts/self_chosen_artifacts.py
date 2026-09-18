"""Verify the preserved self-chosen discovery and observation evidence."""

import argparse
import json

import torch

from experiments.self_chosen.common import OUT, ROOT, RUN, read, write
from scripts.refinement_artifacts import costs, seal, verify
from workbench.storage import Store


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seal", action="store_true")
    p.add_argument("--costs", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.seal:
        for name in ("sera-self-discovery-live", "sera-observed-discovery-live"):
            saved = Store(ROOT / "runs" / name).read()
            if saved:
                path = RUN / (name+".json")
                if path.exists() and read(path) != saved:
                    raise ValueError("Preserve the previously exported live owner")
                write(path, saved)
        seal(41)
    if args.costs:
        costs(41)
    else:
        verify(41)
        final, audit = read(OUT / "final.json"), read(OUT / "audit.json")
        if not final["admitted"] or not audit["passed"] or audit["proposal_checks"] != 2048:
            raise ValueError("Discovery publication disagrees with independent qualification")
        print(json.dumps({"selected": final["selected"], "audited_proposals": audit["proposal_checks"],
                          "checked_routes": final["results"][final["selected"]]["exact_routes"]}))


if __name__ == "__main__":
    main()
