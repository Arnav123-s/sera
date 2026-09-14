"""Independent release artifact audit, including fresh-process ordinary behavior."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from sera.belief import aliased_trajectories, score_predictor
from sera.connected import evaluate_worlds, restore_component
from sera.environments import WorldSpec, collect
from sera.experience import EvidenceReplay
from sera.r1 import RecurrentWorldModel, WorldSession, fit, score
from sera.session_state import pack_tensors
from sera.solver import SolverStore
from sera.storage import digest, write_json
from sera.training import source_hash


def world(row):
    return WorldSpec(row["identifier"], tuple(tuple(value) for value in row["table"]),
                     tuple(row["colors"]), row["resettable"])


def verify_decision(row):
    decision = row["decision"]
    before, after = row["incumbent"], row["candidate"]
    gain = after["macro_score"] - before["macro_score"]
    policy = decision["policy"]
    log_alpha = math.log(policy["global_alpha"]) - (row["round_index"] + 1) * math.log(2)
    bound = gain - math.sqrt(-2 * log_alpha / decision["samples"])
    retention = {key: before["tasks"][key]["score"] - after["tasks"][key]["score"] for key in before["tasks"]}
    assert abs(gain - decision["mean_gain"]) < 1e-7
    assert abs(bound - decision["gain_lower_bound"]) < 1e-7
    assert all(abs(value - decision["retention_loss_by_task"][key]) < 1e-7 for key, value in retention.items())
    passed = bound > policy["minimum_gain"] and max(retention.values()) <= policy["max_retention_loss"] and decision["candidate_cost"] <= policy["max_candidate_cost"]
    assert passed == decision["admitted"] == (row["status"] == "promoted")
    assert before["dataset_id"] == after["dataset_id"] == row["dataset_id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.study / "summary.json").read_text(encoding="utf-8"))
    if summary["manifest"]["environment"]["source_sha256"] != source_hash():
        raise ValueError("Study source differs from the release being audited")
    results = {"source_sha256": source_hash(), "seeds": [], "proposal_decisions_checked": 0}
    for run in summary["runs"]:
        root = args.study / str(run["seed"])
        store = SolverStore(root / "solver")
        predictor_checks = []
        for name, record in run["predictors"]["models"].items():
            payload = torch.load(root / f"predictor-{name}.pt", weights_only=True, map_location="cpu")
            predictor = restore_component(payload["config"])
            predictor.load_state_dict(payload["state"])
            for label, structure, length in (("ordinary", "ordinary", 12), ("repeated", "repeated", 24),
                                               ("long-random", "random", 40)):
                examples = aliased_trajectories(seed=46000 + run["seed"], count=256, length=length,
                                                split=f"test-{label}", structure=structure)
                measured = score_predictor(predictor, examples)
                error = max(abs(value - record["tests"][label][key]) for key, value in measured.items())
                assert error < 1e-7
                predictor_checks.append({"model": name, "split": label, "maximum_metric_difference": error})
        known = {row["identifier"]: world(row) for row in json.loads((store.root / "worlds.json").read_text())}
        known[run["programs"]["world"]["identifier"]] = world(run["programs"]["world"])
        versions, program_checks = [], 0
        for path in sorted((store.root / "versions").glob("v*.json")):
            solver = store.load(path.stem)
            versions.append(path.stem)
            for record in solver.skills.values():
                if record["kind"] != "action_program":
                    continue
                spec = known[record["world_id"]]
                assert spec.execute(record["start"], record["actions"])[-1] == record["goal"]
                for check in record["tests"]:
                    assert spec.execute(check["input"], record["actions"])[-1] == check["expected"]
                program_checks += 1
        rounds = [json.loads(path.read_text()) for path in (store.root / "rounds").glob("*.json")]
        for row in rounds:
            verify_decision(row)
        results["proposal_decisions_checked"] += len(rounds)
        store.journal.verify()
        replay = EvidenceReplay.load(store.root / "experience.json")
        assert len(replay.records) == run["persistence"]["replay_records"]
        assert digest(sorted(replay.identifiers)) == run["persistence"]["replay_dataset_id"]
        # Reexecute one complete saved paired admission per seed; check all other
        # decisions algebraically and all version/skill identities above.
        chosen = run["persistence"]["rounds"][-1]["result"]
        for label, version in (("incumbent", chosen["parent"]), ("candidate", chosen["version"])):
            specs = [known[name] for name in chosen[label]["tasks"]]
            samples = next(iter(chosen[label]["tasks"].values()))["prediction"]["episodes"]
            fresh, _ = evaluate_worlds(store.load(version), specs, seed=chosen["seed"], samples=samples)
            assert fresh["dataset_id"] == chosen["dataset_id"]
            assert abs(fresh["macro_score"] - chosen[label]["macro_score"]) < 1e-7
        request = Path(__file__).resolve().parents[1] / "examples/typed-addition.json"
        process = subprocess.run([sys.executable, "-m", "sera", "typed-solve", str(store.root), str(request)],
                                 capture_output=True, text=True, check=True, timeout=60)
        prediction = json.loads(process.stdout)
        assert prediction["class"] == 2 and prediction["route"] == "verified-program"
        model = store.load("v0").components["r1"]
        session = WorldSession.load(root / "live-session.json", model, owner="stage-three-session")
        session.observe(1, 2, 0.)
        expected = digest({"state": pack_tensors(session.state), "hidden": pack_tensors(session.hidden)})
        code = """import sys,torch
from pathlib import Path
from sera.solver import SolverStore
from sera.r1 import WorldSession
from sera.session_state import pack_tensors
from sera.storage import digest
torch.set_num_threads(1)
root=Path(sys.argv[1]);m=SolverStore(root/'solver').load('v0').components['r1']
s=WorldSession.load(root/'live-session.json',m,owner='stage-three-session');s.observe(1,2,0.)
print(digest({'state':pack_tensors(s.state),'hidden':pack_tensors(s.hidden)}))
"""
        process = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True,
                                 check=True, timeout=60)
        assert process.stdout.strip() == expected
        # Reconstruct the full all-branch reference checkpoint from its declared
        # seed/support, then persist an actual trained diagnostic trajectory.
        reference_started = time.perf_counter()
        spec = world(run["world"]["world"])
        support, _ = collect(spec, seed=run["seed"], count=256, length=8, mask_rate=.3)
        torch.manual_seed(350000 + run["seed"])
        reference = RecurrentWorldModel(width=256, heads=8, memory_dim=32, kind="reference")
        teaching = fit(reference, EvidenceReplay(support), steps=summary["manifest"]["config"]["reference_steps"],
                       batch_size=8, seed=run["seed"])
        original_reference = next(row for row in run["world"]["reference"] if row["routing"] == "all" and row["rank"] == 4)
        history_error = max(abs(a - b) for a, b in zip(teaching["history"], original_reference["training"]["history"]))
        assert history_error < 1e-5
        test, truth = collect(spec, seed=run["seed"], count=summary["manifest"]["config"]["evaluation_samples"],
                               length=12, mask_rate=.4, split="test-retention")
        reproduced, _ = score(reference, test, truth)
        assert abs(reproduced["accuracy"] - original_reference["prediction"]["accuracy"]) < 1e-7
        audit_root = args.study.parent / "stage-three-audit" / str(run["seed"])
        audit_root.mkdir(parents=True, exist_ok=True)
        torch.save({"config": reference.export_config(), "state": reference.state_dict()}, audit_root / "reference-all.pt")
        probe = collect(spec, seed=990000 + run["seed"], count=1, length=16, mask_rate=.4, split="diagnostic-reference")[0][0]
        live = WorldSession(reference, spec.identifier, probe.goal, probe.observations[0],
                             owner="trained-reference-audit", admitted_provenance=[probe.provenance])
        for observation, action, reward in zip(probe.observations[1:], probe.actions, probe.rewards):
            live.observe(observation, action, reward)
        live.save(audit_root / "reference-session.json")
        restored = WorldSession.load(audit_root / "reference-session.json", reference, owner="trained-reference-audit")
        assert digest(pack_tensors(live.state)) == digest(pack_tensors(restored.state))
        result = {"seed": run["seed"], "versions_checked": versions, "programs_reexecuted": program_checks,
                  "predictor_checkpoint_checks": predictor_checks,
                  "decision_checks": len(rounds), "paired_admission_reexecuted": chosen["version"],
                  "replay_records": len(replay.records), "ordinary_typed_fresh_process": prediction,
                  "live_session_fresh_process_digest": expected, "journal_verified": True,
                  "full_reference_reproduction": {"maximum_training_loss_difference": history_error,
                                                   "prediction": reproduced,
                                                   "wall_seconds": time.perf_counter() - reference_started,
                                                   "observations": probe.observations, "actions": probe.actions,
                                                   "session_restored": True, "event_diagnostics": live.diagnostics}}
        results["seeds"].append(result)
        write_json(args.output, results)
        print(f"seed {run['seed']}: versions, programs, decisions, replay and ordinary fresh-process behavior verified", flush=True)
    originals = json.loads((Path(__file__).resolve().parents[1] / "research/reference-catalog.json").read_text())
    results["reference_catalog_sha256"] = digest(originals)
    results["study_summary_sha256"] = hashlib.sha256((args.study / "summary.json").read_bytes()).hexdigest()
    results["passed"] = True
    write_json(args.output, results)


if __name__ == "__main__":
    main()
