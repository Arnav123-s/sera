"""Preserve frozen variants, train a new typed cohort, and reassess retention."""

import argparse
import copy
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs
from sera.capability_evaluation import evaluate_capabilities
from sera.environments import WorldSpec
from sera.evaluation import AdmissionPolicy, assess
from sera.session_state import model_identity
from sera.solver import SolverStore, Work
from sera.storage import write_json
from sera.training import environment, source_hash
from sera.typed_learning import (
    TASKS,
    TypedEvidence,
    TypedReasoner,
    fit_typed,
    score_typed,
    typed_examples,
)
from sera.typed_programs import induce
from sera.typed_protocol import typed_suite


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def baselines(support, tests):
    result = {}
    for partition, rows in tests.items():
        if not partition.startswith("test"):
            continue
        result[partition] = {}
        for task in TASKS:
            examples = [r for r in rows if r.task == task]
            if task == "motion":
                mse = [float(np.mean((np.asarray(r.observations[-1].values) - np.asarray(r.target))**2)) for r in examples]
                result[partition][task] = {"last_position_mse": float(np.mean(mse))}
            else:
                counts = np.bincount([r.target for r in support if r.task == task], minlength=4)
                chosen = int(counts.argmax())
                result[partition][task] = {"support_majority_class": chosen,
                                           "accuracy": float(np.mean([r.target == chosen for r in examples])),
                                           "test_class_counts": np.bincount([r.target for r in examples], minlength=4).tolist()}
    return result


def typed_trial(study, root, seed, steps, count, costs, work):
    frozen = SolverStore(study / str(seed) / "solver").load().components["typed"] if study is not None else None
    original_id = model_identity(frozen) if frozen is not None else None
    old_support = typed_examples(seed=210000 + seed, count=192) if frozen is not None else []
    old_validation = typed_examples(seed=220000 + seed, count=48, split="validation") if frozen is not None else []
    with costs.phase(f"typed/{seed}/data", work):
        suite, manifest = typed_suite(seed=810000 + seed, test_count=count,
                                      exclude=old_support + old_validation)
        serialized = json.dumps({p: [asdict(r) for r in rows] for p, rows in suite.items()}, sort_keys=True).encode()
        (root / "typed-data.json.gz").write_bytes(gzip.compress(serialized, mtime=0))
        write_json(root / "typed-manifest.json", manifest)
        work.add("typed_generated_cases", manifest["unique_semantic_cases"])
    with costs.phase(f"typed/{seed}/training", work):
        torch.manual_seed(820000 + seed)
        fresh = TypedReasoner()
        training = fit_typed(fresh, TypedEvidence(suite["support"]), validation=suite["validation"],
                             steps=steps, seed=seed, work=work)
        programs = {task: induce(suite["support"], suite["validation"], task=task, work=work)
                    for task in ("modular_sum", "byte_sum")}
        fresh.programs = {task: r["record"] for task, r in programs.items() if r["accepted"]}
        torch.save({"config": fresh.export_config(), "state": fresh.state_dict()}, root / "typed-v2.pt")
    results, scores = {}, {}
    with costs.phase(f"typed/{seed}/sealed-evaluation", work):
        routes = [("typed-v2-neural", fresh, False), ("typed-v2-procedural", fresh, True)]
        if frozen is not None:
            routes = [("typed-v1-neural", frozen, False), ("typed-v1-procedural", frozen, True)] + routes
        for name, model, use_programs in routes:
            results[name], scores[name] = {}, {}
            for partition, rows in suite.items():
                if partition.startswith("test"):
                    report, vectors = score_typed(model, rows, work=work, use_programs=use_programs, return_scores=True)
                    results[name][partition] = report
                    scores[name][partition] = {task: vector.tolist() for task, vector in vectors.items()}
    (root / "typed-scores.json.gz").write_bytes(gzip.compress(json.dumps(scores).encode(), mtime=0))
    if frozen is not None and model_identity(frozen) != original_id:
        raise ValueError("Frozen variant changed during evaluation")
    return {"seed": seed, "manifest": manifest, "frozen_model_identity": original_id,
            "new_model_identity": model_identity(fresh), "training": training, "program_induction": programs,
            "results": results, "baselines": baselines(suite["support"], suite),
            "parameters": sum(p.numel() for p in fresh.parameters()),
            "scope": ("Four routes on identical v2 tests: frozen v1 and freshly trained v2 weights, each with/without its acquired arithmetic procedures. Same 26,219-parameter architecture; different training data/initialization. This is not a causal architecture comparison."
                      if frozen is not None else "Fresh typed-v2 training and procedure induction only; no frozen checkpoint or retained-world comparison. This run has its own dataset identity.")}


def retention_trial(study, root, seed, costs, work):
    store = SolverStore(study / str(seed) / "solver")
    worlds = {row["identifier"]: WorldSpec(row["identifier"], tuple(tuple(r) for r in row["table"]),
                                          tuple(row["colors"]), row["resettable"])
              for row in read(store.root / "worlds.json")}
    results = []
    for path in sorted((store.root / "rounds").glob("*.json"), key=lambda p: int(p.stem)):
        original = read(path)
        if original["status"] not in {"promoted", "rejected"}:
            continue
        specs = [worlds[key] for key in original["incumbent"]["tasks"]]
        samples = next(iter(original["incumbent"]["tasks"].values()))["prediction"]["episodes"]
        with costs.phase(f"retention/{seed}/{path.stem}", work):
            extra = Work()
            before_report, before = evaluate_capabilities(store.load(original["parent"]), specs,
                                                          seed=original["seed"], samples=samples, work=extra)
            after_report, after = evaluate_capabilities(store.load(original["version"]), specs,
                                                        seed=original["seed"], samples=samples, work=extra)
            policy = AdmissionPolicy(**original["decision"]["policy"])
            old = assess(dict(after), dict(before), round_index=original["round_index"], invariants_ok=True,
                         candidate_cost=original["decision"]["candidate_cost"], policy=policy)
            if abs(old["mean_gain"] - original["decision"]["mean_gain"]) > 1e-7 or old["admitted"] != original["decision"]["admitted"]:
                raise ValueError("Historical world objective or decision did not reproduce")
            overhead = sum(v for k, v in extra.counts.items() if k.startswith(("typed_", "legacy_retention_")))
            new = assess(after, before, round_index=original["round_index"], invariants_ok=True,
                         candidate_cost=original["decision"]["candidate_cost"] + overhead, policy=policy)
            for key, value in extra.counts.items():
                work.add(key, value)
            result = {"seed": seed, "round": int(path.stem), "parent": original["parent"], "candidate": original["version"],
                      "original_status": original["status"], "old_contract_reproduced": True,
                      "decision_v2": new, "extra_counted_operations": overhead,
                      "incumbent": before_report, "candidate_report": after_report}
            write_json(root / f"retention-{path.stem}.json", result)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("runs/stage-three-complete"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-retention", action="store_true")
    parser.add_argument("--fresh-only", action="store_true", help="Train typed-v2 from scratch without requiring historical local checkpoints")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; preserve all previous trials")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    costs, work = Costs(), Work()
    summary = {"protocol": "preserved-variants-evaluation-v2", "environment": environment(),
               "frozen_source": None if args.fresh_only else read(args.study / "manifest.json")["environment"]["source_sha256"],
               "configuration": {"seeds": args.seeds, "typed_steps": args.steps, "typed_test_count": args.test_count},
               "typed": [], "retention": [],
               "retention_scope": "Retrospective reassessment, not new admissions. Frozen historical proposals and world samples; additional legacy/typed capability tests and overhead. No saved pointer or historical decision is changed."}
    write_json(args.output / "manifest.json", copy.deepcopy(summary))
    try:
        for seed in args.seeds:
            root = args.output / str(seed)
            root.mkdir()
            typed = typed_trial(None if args.fresh_only else args.study, root, seed, args.steps, args.test_count, costs, work)
            summary["typed"].append(typed)
            write_json(root / "typed-results.json", typed)
            print(f"seed {seed}: typed variants evaluated; semantic overlap 0", flush=True)
            if not args.skip_retention and not args.fresh_only:
                summary["retention"].extend(retention_trial(args.study, root, seed, costs, work))
                print(f"seed {seed}: all historical proposals reassessed with separate capabilities", flush=True)
        if source_hash() != summary["environment"]["source_sha256"]:
            raise ValueError("Executable source changed during the study")
        summary["status"] = "completed"
    except Exception as error:
        summary.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(costs=costs.record(), work=work.record())
        summary["artifacts"] = [file_record(p, args.output) for p in sorted(args.output.rglob("*")) if p.is_file()]
        write_json(args.output / "summary.json", summary)
    for row in summary["typed"]:
        print(f"seed {row['seed']}: v2 structural neural scores " + str({k: round(v['score'], 4) for k, v in row['results']['typed-v2-neural']['test-composition']['tasks'].items()}))
    print(f"Completed in {summary['costs']['wall_seconds']:.1f}s; all prior artifacts retained", flush=True)


if __name__ == "__main__":
    main()
