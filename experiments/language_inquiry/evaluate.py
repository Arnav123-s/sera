"""Prospective interpretation and independent-world consequence evaluation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.continuing_control.core import PRECISION, PRIOR, ContinuingSession, sensor
from experiments.continuing_control.environment import World, specification
from sera.session_state import model_identity

from .data import WITHHELD, queries, sequences
from .graph import Investigation, advance, answer, interpret, situation, start
from .study import OUT, ROOT, read, restore_fit, sha, write


def forecast(owner, frame):
    model = situation(owner)
    states = start(model)
    for action in frame["actions"]:
        states = advance(model, states, action)
    return answer(model, states, frame)


def independent_target(world, frame):
    assessor = copy.deepcopy(world)
    for action in frame["actions"]:
        assessor.step(action)
    return assessor.position() if frame["field"] == "position" else np.array([assessor.velocity])


def new_lifetime(parent, seed, family):
    owner = copy.deepcopy(parent)
    # Separate experimental lifetime; neither saved live store is reset.
    for name, value in owner.state_dict().items():
        if name.startswith("interaction_"):
            value.zero_()
    owner.interaction_mean.copy_(torch.from_numpy(PRIOR.copy()))
    owner.interaction_covariance.copy_(torch.from_numpy(np.linalg.inv(PRECISION)))
    session = ContinuingSession(owner)
    spec = specification(seed, family, 48)
    world = World(spec)
    session.admit(sensor(world.measure(), 0, f"LI-evaluation:{seed}:0"))
    for i in range(24):
        action = (-1, 1, 0, 1, -1, 0)[i % 6]
        world.step(action)
        session.admit(sensor(world.measure(), i+1, f"LI-evaluation:{seed}:{i+1}"), action)
    return session, world


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    records = read(ROOT/"runs/LI-fits-final-001/summary.json")
    selected = read(ROOT/"runs/LI-fits-final-001/selected.json")
    frozen = {"protocol": sha(OUT/"protocol.md"), "checkpoints": {r["checkpoint"]: sha(ROOT/r["checkpoint"]) for r in records},
              "sources": {f.name: sha(f) for f in Path(__file__).parent.glob("*.py")},
              "world_seeds": list(range(24100, 24112)), "selection": selected,
              "assessment_scope": "No final result selects weights, thresholds or a seed"}
    write(a.output/"frozen.json", frozen)
    all_queries = queries(sequences(), templates=(0, 1, 2), entities=("orbit", "comet"))
    language_results, summaries = [], []
    for row in records:
        session, _ = restore_fit(ROOT/row["checkpoint"])
        for q in all_queries:
            parsed = interpret(session.owner, q["text"], q["entity"])
            language_results.append({"kind": row["kind"], "seed": row["seed"], **q,
                "frame": parsed["frame"], "exact": parsed["frame"]["actions"] == q["actions"] and parsed["frame"]["field"] == q["field"],
                "withheld_composition": tuple(q["actions"]) in WITHHELD,
                "new_surface": q["text"].startswith("after ")})
        local = language_results[-len(all_queries):]
        summaries.append({"kind": row["kind"], "seed": row["seed"], "count": len(local),
                          "exact": sum(r["exact"] for r in local)/len(local),
                          **{part: sum(r["exact"] for r in local if r[part])/sum(r[part] for r in local)
                             for part in ("withheld_composition", "new_surface")}})
    write(a.output/"language-records.json", language_results)
    write(a.output/"language-summary.json", summaries)
    chosen, _ = restore_fit(ROOT/selected["checkpoint"])
    world_rows = []
    for seed in range(24100, 24112):
        family = "gain_reversal" if seed < 24106 else "omitted_torque"
        session, world = new_lifetime(chosen.owner, seed, family)
        graph = Investigation(session.owner)
        graph.add(session.owner, "live", "predict the position of orbit after push left then wait")
        graph.add(session.owner, "historical", "predict the position of orbit after push left then wait", anchor="historical")
        graph.run(session.owner, 1)
        graph = Investigation.restore(session.owner, graph.snapshot())
        graph.run(session.owner, 3)
        history = []
        for stage in ("before_new_evidence", "after_12_new_observations"):
            if stage.startswith("after"):
                for i in range(12):
                    action = (1, -1, 0, -1, 1, 0)[i % 6]
                    world.step(action)
                    values = world.measure()
                    session.admit(sensor(values, len(session.events), f"LI-evaluation:{seed}:{world.time}"), action)
                graph.sync(session.owner)
                assert graph.jobs["live"]["status"] == "stale"
                assert graph.jobs["historical"]["status"] == "complete"
                graph.rebase(session.owner, "live")
                graph.run(session.owner, 2)
            identity = model_identity(session.owner)
            for q in queries(sequences(), templates=(0,)):
                # Wording variants are reported as correlated requests, not new worlds.
                parsed = interpret(session.owner, q["text"], q["entity"])
                oracle = {"entity": q["entity"], "actions": q["actions"], "field": q["field"]}
                ignored = {**oracle, "actions": [0]*len(q["actions"])}
                for method, frame in (("learned", parsed["frame"]), ("supplied", oracle), ("ignored", ignored)):
                    prediction = forecast(session.owner, frame)
                    target = independent_target(world, oracle)
                    error = float(np.linalg.norm(np.array(prediction["value"])-target))
                    world_rows.append({"seed": seed, "family": family, "stage": stage, "method": method,
                        "text": q["text"], "frame": frame, "expected_frame": oracle,
                        "prediction": prediction, "target": target.tolist(), "error": error,
                        "interpretation_exact": frame == oracle, "within_005": error <= .05})
            assert model_identity(session.owner) == identity
            history.append({"stage": stage, "observations": copy.deepcopy(session.events),
                            "situation": situation(session.owner), "graph": graph.snapshot(),
                            "world_assessor_only": {"spec": world.spec, "state": world.assessment()}})
        write(a.output/f"world-{seed}.json", history)
    write(a.output/"world-records.json", world_rows)
    world_summary = []
    for family in ("gain_reversal", "omitted_torque"):
        for stage in ("before_new_evidence", "after_12_new_observations"):
            for field in ("position", "velocity"):
                for method in ("learned", "supplied", "ignored"):
                    rows = [r for r in world_rows if (r["family"], r["stage"], r["frame"]["field"], r["method"]) == (family, stage, field, method)]
                    world_summary.append({"family": family, "stage": stage, "field": field, "method": method,
                        "independent_worlds": 6, "correlated_requests": len(rows), "mean_error": float(np.mean([r["error"] for r in rows])),
                        "within_005": float(np.mean([r["within_005"] for r in rows]))})
    write(a.output/"world-summary.json", world_summary)
    write(a.output/"completion.json", {"language_fits": len(records), "language_queries": len(language_results),
          "independent_worlds": 12, "world_method_records": len(world_rows), "paid_sensor_observations": 12*37,
          "executed_learning_controls": 12*36, "history_restores": 12,
          "interpretation_errors_and_forecast_errors_separated": True,
          "final_data_used_for_learning_or_selection": False})
    print(json.dumps({"language": summaries, "world_rows": len(world_rows)}), flush=True)


if __name__ == "__main__":
    main()
