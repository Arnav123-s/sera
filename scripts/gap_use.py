"""Use SERA's retained explanations and earlier abilities through the same owner."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gap_consequences import residual_consequences
from experiments.gap_inquiry import OUT, ROOT, RUN, GapSession, predict, read, sha, write
from scripts.continuing_use import perform as earlier_perform
from sera.session_state import model_identity
from workbench.storage import Store


def restore():
    stored = Store(ROOT / "runs/sera-gap-inquiry-live").read()
    if stored is None:
        raise ValueError("A qualified gap inquiry successor is required")
    parent = Store(ROOT / stored["parent_store"]).read()
    if parent["owner"] != stored["parent_owner"] or stored["audit"] != sha(OUT / "audit.json") or stored["final"] != sha(OUT / "final.json"):
        raise ValueError("Changed source, parent or independent admission")
    session = GapSession(parent, stored["state"])
    if session.identity() != stored["owner"]:
        raise ValueError("Changed restored owner")
    growth = session.base.base.base
    growth.base.base.base.base.refresh()
    return session, growth


def empirical_scope(session):
    public = read(RUN / "development.json")
    original = next(iter(session.base.evidence.values()))
    ev = original["evaluation"]
    origin = ev["origin_seconds"]
    return {"new_source_range": [min(public["time"]), max(public["time"])],
            "preserved_original_range": [min(ev["times"]) + origin, max(ev["times"]) + origin],
            "source": original["source"], "origin": origin}


def perform(session, growth, request):
    kind = request.get("kind")
    if kind == "gap_findings":
        result = {"goal": read(OUT / "gap.json"), "investigation": read(OUT / "selection.json"),
                  "open_goals": session.open_goals, "scope": empirical_scope(session),
                  "models": [{"id": key, "program": c["program"], "weights": session.owner.gap_models[key].detach().tolist(),
                              "basis": c["basis"], "coordinate_center": c["center"], "coordinate_scale": c["scale"]}
                             for key, c in session.models.items()],
                  "credit": session.credits, "status": "VERIFIED_EMPIRICAL_EXPLANATION_PORTFOLIO"}
    elif kind == "gap_predict":
        times = request.get("times")
        if not isinstance(times, list) or not 1 <= len(times) <= 256 or not np.isfinite(times).all():
            raise ValueError("Supply one to 256 finite source-clock times")
        scope = empirical_scope(session)
        outputs = []
        for t in times:
            if not scope["new_source_range"][0] <= t <= scope["new_source_range"][1]:
                outputs.append({"time": t, "status": "NEW_OBSERVATION_NEEDED", "goal_preserved": True})
                continue
            if scope["preserved_original_range"][0] <= t <= scope["preserved_original_range"][1]:
                old = earlier_perform(session.base, growth, {"kind": "observed_motion", "time": t - scope["origin"]})
                outputs.append({"time": t, "route": "preserved_original_scope", "prediction": old["result"]})
                continue
            alternatives = [{"model": key, "value": float(predict(candidate, [t], session.owner.gap_models[key].detach().numpy())[0]),
                             "assumptions": candidate["program"]} for key, candidate in session.models.items()]
            values = [v["value"] for v in alternatives]
            outputs.append({"time": t, "status": "SCOPED_EMPIRICAL_PREDICTION", "alternatives": alternatives,
                            "model_disagreement_range": [min(values), max(values)],
                            "range_is_confidence_interval": False, "measured_event": False})
        result = {"scope": scope, "predictions": outputs}
    elif kind == "gap_consequences":
        time = float(request["time"])
        scope = empirical_scope(session)
        if not scope["new_source_range"][0] <= time <= scope["new_source_range"][1]:
            raise ValueError("Consequences require a time within the investigated source range")
        if scope["preserved_original_range"][0] <= time <= scope["preserved_original_range"][1]:
            raise ValueError("Use the preserved original model inside its previously verified interval")
        result = residual_consequences(session, time)
    elif kind == "gap_next_observation":
        scope = empirical_scope(session)
        grid = np.linspace(*scope["new_source_range"], 1025)
        grid = grid[(grid < scope["preserved_original_range"][0]) | (grid > scope["preserved_original_range"][1])]
        values = np.array([predict(c, grid, session.owner.gap_models[key].detach().numpy()) for key, c in session.models.items()])
        disagreement = np.ptp(values, axis=0)
        index = int(np.argmax(disagreement))
        result = {"status": "PROPOSED_DISCRIMINATING_OBSERVATION", "source_clock_time": float(grid[index]),
                  "model_disagreement": float(disagreement[index]),
                  "predicted_alternatives": values[:, index].tolist(), "observed": False,
                  "original_goal": read(OUT / "gap.json")["id"],
                  "purpose": "Distinguish retained predictive explanations using new independent measurement"}
    else:
        return earlier_perform(session.base, growth, request)
    return {"id": request.get("id"), "kind": kind, "owner": model_identity(session.owner), "result": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh output within runs")
    tasks = read(args.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("Use a finite batch of one to 256 tasks")
    session, growth = restore()
    before = session.identity()
    results = []
    for request in tasks:
        try:
            results.append(perform(session, growth, request))
        except (ValueError, KeyError, IndexError) as error:
            results.append({"id": request.get("id"), "status": "RETAINED_OPEN", "reason": str(error)})
        write(args.output, results)
    if session.identity() != before:
        raise ValueError("Inference altered the learner")
    print(json.dumps({"owner": before, "tasks": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
