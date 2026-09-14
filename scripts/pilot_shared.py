"""Development-only numerical/learning pilot; no sealed query results are used."""

import argparse
from pathlib import Path

import torch

from sera.accounting import Costs
from sera.binding import binding_cases
from sera.shared import SharedR1, SharedTypedView
from sera.shared_learning import adapt_shared, development, pretrain_shared
from sera.solver import Work
from sera.storage import write_json
from sera.training import environment
from sera.typed_learning import score_typed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("delta", "reference"), default="delta")
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--adapt-steps", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep old development pilots")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    costs, work = Costs(), Work()
    result = {"environment": environment(), "role": "validation-only development pilot", "kind": args.kind}
    evidence, validation, _, manifest = development(41)
    torch.manual_seed(41)
    model = SharedR1(kind=args.kind)
    with costs.phase("joint-pretraining", work):
        result["pretraining"] = pretrain_shared(model, evidence, validation, seed=41, steps=args.pretrain_steps, work=work)
    support = binding_cases(seed=982041, count=128)
    checks = binding_cases(seed=983041, count=128, split="validation")
    result["before_binding"] = score_typed(SharedTypedView(model), checks)
    result["adaptation"] = []
    torch.save({"config": model.export_config(), "state": model.state_dict()}, args.output / "base.pt")
    for method in ("full", "replay"):
        with costs.phase(method, work):
            candidate, training = adapt_shared(model, evidence, support, checks, method=method,
                                                 seed=41, steps=args.adapt_steps, work=work)
            torch.save({"config": candidate.export_config(), "state": candidate.state_dict()}, args.output / f"{method}.pt")
            row = {"training": training, "binding_validation": score_typed(SharedTypedView(candidate), checks)}
            result["adaptation"].append(row)
            print(method, row["binding_validation"]["tasks"]["binding"]["score"], flush=True)
    result.update(costs=costs.record(), work=work.record(), typed_manifest=manifest)
    write_json(args.output / "pilot.json", result)
    print("Pretraining validation macro", result["pretraining"]["best_validation_macro"], "seconds", result["costs"]["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
