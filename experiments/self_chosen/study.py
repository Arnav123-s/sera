"""Freeze, teach, investigate, evaluate, and retain a self-chosen discovery portfolio."""

import argparse
import copy
import json

import torch

from experiments.continuing_growth.runtime import GrowthSession
from experiments.verified_completion.credit import decode, encode
from workbench.storage import Store

from . import equations, language, teaching
from .common import ARMS, BLOCK, DOMAINS, OUT, ROOT, RUN, SEEDS, STEPS, contracts, read, sha, write
from .runtime import Session


def save(path, value):
    if path.exists():
        raise FileExistsError("Preserve an existing exact discovery checkpoint")
    torch.save(value, path)


def prepare():
    if (RUN / "freeze.json").exists():
        raise FileExistsError("Discovery sources already frozen")
    RUN.mkdir(parents=True, exist_ok=True)
    parent = Store(ROOT / "runs/sera-growth-live").read()
    growth = GrowthSession(parent["parent"], saved=decode(parent["state"]))
    write(RUN / "parent.json", parent)
    language.prepare(growth.owner)
    session = Session(parent)
    report = session.teach()
    report["actual_example_execution"] = teaching.run_examples(session.owner)
    write(OUT / "teaching.json", report)
    save(RUN / "taught.pt", session.state())
    write(RUN / "freeze.json", {"contracts": contracts(), "parent_owner": parent["owner"],
                               "parent": sha(RUN / "parent.json"), "human_sources": sha(RUN / "language-sources.json"),
                               "question_inventory": session.inventory, "seeds": list(SEEDS), "steps": STEPS,
                               "external_observation_sources_opened": False, "final_opened": False})
    write(OUT / "freeze.json", read(RUN / "freeze.json"))
    print(json.dumps({"teaching_transfer": report["transfer_correct"], "questions": len(session.inventory), "parent": parent["owner"]}))


def train(arm, seed, limit):
    frozen = read(RUN / "freeze.json")
    if frozen["contracts"] != contracts() or (OUT / "selection.json").exists():
        raise ValueError("Preserve frozen sources and completed selection")
    paths = sorted(RUN.glob(f"{arm}-{seed}-*.pt"))
    initial = torch.load(paths[-1] if paths else RUN / "taught.pt", weights_only=True)
    if len(initial["events"]) >= STEPS:
        raise FileExistsError("This investigator already completed")
    initial = copy.deepcopy(initial)
    if not paths:
        initial["seed"] = seed
        import numpy as np
        initial["rng"] = np.random.default_rng(seed).bit_generator.state
    session = Session(read(RUN / "parent.json"), seed=seed, saved=initial)
    def commit(pending):
        path = RUN / "proposals" / f"{arm}-{seed}-{pending['sequence']:04d}.json"
        if path.exists():
            # Recovery after a crash between commitment and checkpoint must
            # produce exactly the same previously committed conjecture.
            if read(path) != pending:
                raise ValueError("Resumed proposal changed after its evidence was opened")
        else:
            write(path, pending)
    stop = min(STEPS, len(initial["events"])+limit)
    while len(session.events) < stop:
        session.step(arm, commit)
        if len(session.events) % BLOCK == 0 or len(session.events) == stop:
            path = RUN / f"{arm}-{seed}-{len(session.events):04d}.pt"
            save(path, session.state())
            print(json.dumps({"arm": arm, "seed": seed, "decisions": len(session.events),
                              "retained": len(session.records), "reward": sum(e["reward"] for e in session.events)}), flush=True)


def summary(saved):
    events, records = saved["events"], saved["records"]
    return {"decisions": len(events), "candidate_attempts": sum(len(e["proposals"]) for e in events),
            "retained": len(records), "reward": sum(e["reward"] for e in events),
            "unique_connections": len({r["relation"] for r in records}),
            "unique_routes": len({r["route"] for r in records}),
            "rejected": sum(not r["receipt"]["accepted"] for e in events for r in e["receipts"]),
            "domains": {d: {"decisions": sum(e["question"]["domain"] == d for e in events),
                            "retained": sum(r["question"]["domain"] == d for r in records)} for d in DOMAINS}}


def select():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Preserve completed selection")
    candidates = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = RUN / f"{arm}-{seed}-{STEPS:04d}.pt"
            candidates[f"{arm}-{seed}"] = {**summary(torch.load(path, weights_only=True)), "checkpoint": sha(path)}
    selected = max(candidates, key=lambda k: (candidates[k]["unique_routes"], candidates[k]["reward"], k))
    write(OUT / "selection.json", {"contracts": contracts(), "candidates": candidates, "selected": selected,
                                    "criterion": "independently supported distinct routes, then total checked reward", "final_opened": False})
    print(json.dumps({"selected": selected, "candidates": candidates}, indent=2))


def final():
    if (OUT / "final.json").exists():
        raise FileExistsError("Preserve the sealed final")
    selection = read(OUT / "selection.json")
    results = {}
    for name in selection["candidates"]:
        saved = torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True)
        checks = []
        for r in saved["records"]:
            if r["question"]["domain"] in DOMAINS[:4]:
                checked = equations.check_execution(r["proposal"], equations.reference_rows(r["question"]["domain"], 41981, 96))
            else:
                checked = language.assess(r["question"], r["proposal"], "final")
            checks.append({"id": r["id"], "domain": r["question"]["domain"], "result": checked})
        algebra = [c for c in checks if c["domain"] in DOMAINS[:4]]
        empirical = [c for c in checks if c["domain"] not in DOMAINS[:4]]
        results[name] = {"checks": checks, "exact_pass": all(c["result"]["accepted"] for c in algebra),
                         "exact_routes": len(algebra), "exact_cases": sum(c["result"]["checked"] for c in algebra),
                         "language_confirmed": sum(c["result"]["accepted"] for c in empirical), "language_attempted": len(empirical)}
    chosen = results[selection["selected"]]
    write(OUT / "final.json", {"selection": sha(OUT / "selection.json"), "selected": selection["selected"],
                               "results": results, "admitted": chosen["exact_pass"],
                               "empirical_policy": "retain original development receipts and final outcomes; no unconditional request override or factual promotion"})
    print(json.dumps({k: {a: b for a, b in v.items() if a != "checks"} for k, v in results.items()}, indent=2))


def integrate():
    report = read(OUT / "audit.json")
    final = read(OUT / "final.json")
    if not report["passed"] or not final["admitted"]:
        raise ValueError("Independent replay and admission are required")
    saved = torch.load(RUN / f"{final['selected']}-{STEPS:04d}.pt", weights_only=True)
    session = Session(read(RUN / "parent.json"), saved=saved)
    live = Store(ROOT / "runs/sera-self-discovery-live")
    if live.read() is not None:
        raise FileExistsError("Preserve an existing discovery owner")
    snapshot = {"schema": "sera.self-chosen-discovery.1", "parent": session.parent,
                "state": encode(saved), "owner": session.identity(), "audit": sha(OUT / "audit.json"),
                "final": sha(OUT / "final.json")}
    live.commit(snapshot, None)
    write(OUT / "integration.json", {"owner": snapshot["owner"], "store": "runs/sera-self-discovery-live",
                                     "same_actual_owner": session.owner is session.base.owner,
                                     "records": len(session.records), "parent": session.parent["owner"]})
    print(json.dumps(read(OUT / "integration.json"), indent=2))


def future():
    if (OUT / "future.json").exists():
        raise FileExistsError("Preserve future procedure comparison")
    selected = read(OUT / "selection.json")["selected"]
    old = torch.load(RUN / "taught.pt", weights_only=True)
    new = torch.load(RUN / f"{selected}-{STEPS:04d}.pt", weights_only=True)
    parent = read(RUN / "parent.json")
    session = Session(parent, saved=old)
    trials = []
    import numpy as np
    for seed in (41401, 41402):
        for knowledge in ("old", "new"):
            for policy in ("old", "new", "balanced"):
                name = f"{seed}-{knowledge}-{policy}"
                saved = copy.deepcopy(old)
                saved["rng"] = np.random.default_rng(seed).bit_generator.state
                if knowledge == "new":
                    saved["records"] = copy.deepcopy(new["records"])
                if policy == "new":
                    saved["policy"] = copy.deepcopy(new["policy"])
                session.restore(saved)
                before = len(session.records)
                def commit(pending):
                    path = RUN / "future-proposals" / f"{name}-{pending['sequence']:04d}.json"
                    if path.exists() and read(path) != pending:
                        raise ValueError("Future proposal changed on recovery")
                    if not path.exists():
                        write(path, pending)
                for _ in range(32):
                    session.step("balanced" if policy == "balanced" else "learned", commit, phase="future", learn=False)
                path = RUN / f"future-{name}.pt"
                save(path, session.state())
                result = {"seed": seed, "knowledge": knowledge, "policy": policy, "new_retained": len(session.records)-before,
                          "reward": sum(e["reward"] for e in session.events), "checkpoint": sha(path),
                          "decisions": len(session.events), "policy_frozen": True}
                trials.append(result)
                print(json.dumps(result), flush=True)
    comparisons = []
    for seed in (41401, 41402):
        for knowledge in ("old", "new"):
            row = {r["policy"]: r for r in trials if r["seed"] == seed and r["knowledge"] == knowledge}
            comparisons.append(row["new"]["new_retained"] > max(row["old"]["new_retained"], row["balanced"]["new_retained"]))
    write(OUT / "future.json", {"trials": trials, "all_seed_knowledge_improvement": all(comparisons),
                                "scope": "32 fresh question choices, frozen procedures, independently assessed discoveries; supplied investigator algorithms"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "select", "final", "future", "integrate"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--limit", type=int, default=STEPS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "train":
        if args.arm is None or args.seed is None or not 1 <= args.limit <= STEPS:
            raise ValueError("An arm, seed and finite work limit are required")
        train(args.arm, args.seed, args.limit)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
