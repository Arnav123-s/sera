"""Disjoint development and sealed continuing lifetimes; no privileged learner input."""

import argparse
import copy
import gzip
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from experiments.acquisition_dependence.study import load_before
from experiments.guarded_consolidation.study import ReadOnlyStore
from sera.session_state import model_identity
from sera.shared import replace_shared_owner
from sera.storage import canonical, digest

from .control import HORIZON, decide
from .core import ContinuingSession, extend_solver, sensor
from .environment import World, assess_plan, specification

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/19_continuing"
INTEGRATION = ROOT/"research-continuation/17_transfer/integration/result.json"
ARMS = ("adaptive_mpc", "full_history_mpc", "frozen_mpc", "adaptive_one_step", "adaptive_reactive", "incorrect_geometry", "analytic_mpc")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent():
    record = read(INTEGRATION)
    solver = ReadOnlyStore(ROOT/record["store"]).load()
    if solver.identity() != record["solver_record"]["solver_sha256"]:
        raise ValueError("Qualified experimental predecessor changed")
    return solver


def sources():
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((ROOT/"src/sera").glob("*.py"))
    files += [ROOT/"scripts/run_continuing_bounded.py", ROOT/"scripts/run_v3_bounded.py",
              ROOT/"scripts/windows_job_v3.py", ROOT/"tests/test_continuing_control.py"]
    # Imported predecessor entry points and their source closure are already sealed.
    files += [ROOT/name for name in read(ROOT/"research-continuation/18_instance_transfer/source-manifest.json")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}


def freeze(directory, config, *, final):
    directory.mkdir(parents=True, exist_ok=False)
    payload = {"experiment": directory.name, "final": final, "config": config, "sources": sources(),
               "parent_solver": read(INTEGRATION)["solver_record"]["solver_sha256"],
               "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
               "primary": "Mean physical tracking/action cost per continuing lifetime after the three-observation prefix; paired world is the uncertainty unit.",
               "gate": {"nominal_cost_reduction_vs_reactive": .10, "nominal_cost_reduction_vs_one_step": .10,
                        "both_paired_97_5_percent_lower_positive": True, "post_change_gain_vs_frozen": .10,
                        "priced_work_gain_vs_reactive": .10, "conditional_particle_price": .00001,
                        "posterior_fit_price": .001, "measurement_price": .001,
                        "old_tensors_unchanged": True, "exact_replay_and_no_factual_imagination": True},
               "cost_boundary": "All controls share observations and available goal schedules. Equal maximum decision allowance, actual work reported. Analytic dynamics are privileged and separate. No exact-FLOP or same-total-compute superiority claim.",
               "scope": "One acquired circle geometry, supplied angular-dynamics grammar, fixed engineered eta. New observed action dynamics, not new neural weights. Finite continuous lifetimes, not successor generations.",
               "separation": "Development seeds 6001,6011,6029 and 6043,6047,6053; final seeds begin 7001 and 8009; original A06 final banks unused."}
    protocol = {"payload": payload, "sha256": digest(payload)}
    write(directory/"protocol.json", protocol)
    with zipfile.ZipFile(directory/"frozen-sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in payload["sources"]:
            archive.writestr(name, (ROOT/name).read_bytes())
    return protocol


def check_contract(directory):
    protocol = read(directory/"protocol.json")
    if digest(protocol["payload"]) != protocol["sha256"]:
        raise ValueError("Protocol changed")
    for name, expected in protocol["payload"]["sources"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return protocol


def choose_owner(original, incorrect, arm, window):
    base = copy.deepcopy(original)
    if arm == "incorrect_geometry":
        with torch.no_grad():
            for name, tensor in base.components["r1"].state_dict().items():
                if name.startswith("generator_"):
                    tensor.copy_(incorrect.state_dict()[name])
    selected_window = 10000 if arm == "full_history_mpc" else window
    solver = extend_solver(base, selected_window, arm not in ("full_history_mpc", "frozen_mpc"))
    return base, solver, ContinuingSession(solver.components["r1"], adapt=arm != "frozen_mpc")


def run_lifetime(original, incorrect, spec, arm, window, checkpoint, *, resume=False):
    started = time.perf_counter()
    base, solver, session = choose_owner(original, incorrect, arm, window)
    world = World(spec)
    rows = []
    # Prefix measurements and coast actions are paid and never repeated mid-lifetime.
    if checkpoint.exists():
        saved = read(checkpoint)
        if not resume or saved["spec"] != spec or saved["arm"] != arm:
            raise ValueError("Pending checkpoint must be reconciled with its own lifetime")
        session = ContinuingSession.restore(base.components["r1"], saved["session"])
        replace_shared_owner(solver, session.owner)
        world.angle, world.velocity, world.time = (saved["world"][k] for k in ("angle", "velocity", "time"))
        rows = saved["rows"]
        if saved["next_index"] != len(rows):
            raise ValueError("Pending history length differs")
    else:
        session.admit(sensor(world.measure(), 0, f"{spec['seed']}:0"))
        for i in range(1, 3):
            world.step(0)
            session.admit(sensor(world.measure(), i, f"{spec['seed']}:{i}"), 0)
    for index in range(len(rows), spec["steps"]):
        goals = world.goals(HORIZON)
        policy = {"adaptive_reactive": "reactive", "adaptive_one_step": "one_step", "analytic_mpc": "analytic"}.get(arm, "mpc")
        kwargs = {"oracle_coefficients": world.coefficients()} if arm == "analytic_mpc" else {}
        plan = decide(session, goals, policy, **kwargs)
        session.validate_plan(plan)
        assumed = assess_plan(world, plan["actions"])
        before = world.assessment()
        actual = world.step(plan["action"])
        cost = float(np.square(actual-goals[0]).sum()+.005*plan["action"]**2+.05*world.velocity**2)
        observed = world.measure()
        session.admit(sensor(observed, len(session.events), f"{spec['seed']}:{world.time}"), plan["action"])
        row = {"index": index, "before_assessor": before, "goals": goals.tolist(), "plan": plan,
               "counterfactual_assessor_positions": assumed.tolist(), "observed": observed.tolist(),
               "true_position": actual.tolist(), "cost": cost, "after_assessor": world.assessment(),
               "posterior_after_observation": session.owner.interaction_mean.tolist()}
        rows.append(row)
        # Every completed transition has an exact local continuation checkpoint.
        pending = {"spec": spec, "arm": arm, "next_index": index+1, "world": world.assessment(),
                   "session": session.snapshot(), "rows": rows, "state": "IN_PROGRESS"}
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(canonical(pending)+"\n", encoding="utf-8", newline="\n")
        temporary.replace(checkpoint)
    unchanged = all(torch.equal(tensor, session.owner.state_dict()[name]) for name, tensor in base.components["r1"].state_dict().items())
    snapshot = session.snapshot()
    restored = ContinuingSession.restore(base.components["r1"], snapshot)
    reload_equal = all(torch.equal(t, restored.owner.state_dict()[n]) for n, t in session.owner.state_dict().items())
    rebound = copy.deepcopy(base)
    replace_shared_owner(rebound, restored.owner)
    if solver.identity() != rebound.identity() or not reload_equal or not unchanged:
        raise ValueError("Whole-owner replay or old tensor preservation failed")
    checkpoint.unlink()  # Only this study's replaced scratch checkpoint; final snapshot follows.
    return {"spec": spec, "arm": arm, "rows": rows, "snapshot": snapshot,
            "restored_work": restored.work, "old_tensors_unchanged": unchanged,
            "exact_reload": reload_equal, "solver_sha256": solver.identity(),
            "seconds": time.perf_counter()-started, "base_owner_sha256": model_identity(base.components["r1"])}


def execute(directory, *, resume=False):
    protocol = check_contract(directory)
    config = protocol["payload"]["config"]
    results = directory/"lifetimes"
    if (directory/"summary.json").exists():
        raise ValueError("Completed cohorts cannot restart")
    results.mkdir(exist_ok=resume)
    local = ROOT/"runs"/directory.name
    local.mkdir(exist_ok=resume)
    original, incorrect = parent(), load_before()[0]
    original_identity = original.identity()
    summary = []
    started = time.perf_counter()
    for family, seeds in config["families"].items():
        for seed in seeds:
            spec = specification(seed, family, config["steps"])
            for arm in config["arms"]:
                identity = f"{family}-{seed}-{arm}"
                raw_path = results/(identity+".json.gz")
                if raw_path.exists():
                    if not resume:
                        raise ValueError("Completed lifetime exists")
                    record = json.loads(gzip.decompress(raw_path.read_bytes()))
                else:
                    record = run_lifetime(original, incorrect, spec, arm, config["window"], local/"in-progress.json", resume=resume)
                    raw_path.write_bytes(gzip.compress(canonical(record).encode(), mtime=0))
                selected = local/(identity+".json")
                if selected.exists():
                    if read(selected) != record["snapshot"]:
                        raise ValueError("Preserved checkpoint differs from completed evidence")
                else:
                    write(selected, record["snapshot"])
                costs = np.array([r["cost"] for r in record["rows"]])
                item = {"identity": identity, "family": family, "seed": seed, "arm": arm,
                        "cost": float(costs.mean()), "post_change_cost": float(np.mean([r["cost"] for r in record["rows"]
                            if r["before_assessor"]["time"] >= spec["change_step"]])),
                        "seconds": record["seconds"], "work": record["snapshot"]["work"],
                        "old_tensors_unchanged": record["old_tensors_unchanged"], "exact_reload": record["exact_reload"],
                        "checkpoint": str((local/(identity+".json")).relative_to(ROOT).as_posix()),
                        "checkpoint_sha256": sha(local/(identity+".json")), "raw_sha256": sha(results/(identity+".json.gz"))}
                summary.append(item)
                write(directory/"progress.json", {"status": "RUNNING", "completed": summary,
                      "next": "Continue only uncompleted cells from the same protocol; in-progress.json is the exact last completed transition."})
    if original.identity() != original_identity:
        raise ValueError("Immutable parent was altered")
    write(directory/"summary.json", {"status": "COMPLETE", "protocol": protocol["sha256"], "records": summary,
                                    "seconds": time.perf_counter()-started, "parent_unchanged": True})
    write(directory/"progress.json", {"status": "COMPLETE", "completed_lifetimes": len(summary), "restart_allowed": False})
    aggregate = {}
    for family in config["families"]:
        aggregate[family] = {arm: {"cost": float(np.mean([r["cost"] for r in summary if r["arm"] == arm and r["family"] == family])),
                                   "seconds": sum(r["seconds"] for r in summary if r["arm"] == arm and r["family"] == family)}
                              for arm in config["arms"]}
    print(json.dumps({"status": "COMPLETE", "lifetimes": len(summary), "seconds": time.perf_counter()-started,
                      "aggregate": aggregate}, indent=2))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "freeze", "run"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--pilot-generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.startswith("A08-"):
        raise ValueError("A unique A08 experiment name is required")
    directory = RELEASE/args.name
    if args.mode == "pilot":
        seeds = [6001, 6011, 6029] if args.pilot_generation == 1 else [6043, 6047, 6053]
        config = {"window": args.window, "steps": 60,
                  "families": {"gain_reversal": seeds}, "arms": list(ARMS)}
        freeze(directory, config, final=False)
        execute(directory)
    elif args.mode == "freeze":
        config = {"window": args.window, "steps": 72,
                  "families": {"gain_reversal": list(range(7001, 7049, 4)), "omitted_torque": list(range(8009, 8057, 4))},
                  "arms": list(ARMS)}
        freeze(directory, config, final=True)
        print(json.dumps({"status": "FROZEN", "lifetimes": 168, "transitions_per_lifetime": 74}))
    else:
        execute(directory, resume=args.resume)


if __name__ == "__main__":
    main()
