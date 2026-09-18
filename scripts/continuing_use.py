"""Batch useful tasks through the latest independently admitted continuing owner."""

import argparse
import json
from pathlib import Path

import torch

from experiments.continuing_growth.runtime import GrowthSession
from experiments.self_study.algebra import check, independent
from experiments.stream_curriculum.model import encode as request_tokens
from experiments.verified_completion.common import ROOT, read, write
from experiments.verified_completion.credit import decode
from sera.session_state import model_identity
from workbench.storage import Store


def restore(store=None):
    preferred = next((p for p in ("runs/sera-observed-discovery-live", "runs/sera-composed-discovery-live", "runs/sera-self-discovery-live",
                                 "runs/sera-step-resolution-live", "runs/sera-sustained-growth-live", "runs/sera-growth-live")
                      if (ROOT / p / "current.json").exists()), "runs/sera-growth-live")
    path = ROOT / (store or preferred)
    if not path.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use an owned retained learner store")
    saved = Store(path).read()
    if saved is None:
        raise ValueError("No independently admitted continuing owner")
    if saved["schema"] == "sera.observed-discovery.1":
        from experiments.discovery_observation import ObservationSession
        session = ObservationSession(saved["parent"], saved["evidence"])
        growth = session.base.base
    elif saved["schema"] == "sera.composed-discovery.1":
        from experiments.discovery_frontier import FrontierSession
        session = FrontierSession(saved["parent"], saved=decode(saved["state"]))
        growth = session.base
    elif saved["schema"] == "sera.self-chosen-discovery.1":
        from experiments.self_chosen.runtime import Session
        session = Session(saved["parent"], saved=decode(saved["state"]))
        growth = session.base
    elif saved["schema"] == "sera.step-resolution.1":
        from experiments.refinement_resolution import ResolutionSession
        session = ResolutionSession(saved["parent"], saved=decode(saved["state"]))
        growth = session.base.base
    elif saved["schema"] == "sera.sustained-refinement.1":
        from experiments.sustained_refinement import Session
        session = Session(saved["parent"], saved=decode(saved["state"]))
        growth = session.base
    elif saved["schema"] == "sera.continuing-growth.1":
        session = GrowthSession(saved["parent"], saved=decode(saved["state"]))
        growth = session
    else:
        raise ValueError("Unknown continuing owner schema")
    if model_identity(session.owner) != saved["owner"]:
        raise ValueError("Restored owner identity changed")
    # Rebind inherited typed views after continuing weights. Existing signed
    # portfolio qualifications remain bound to their original owner identities.
    growth.base.base.base.base.refresh()
    return session, growth


def perform(session, growth, request):
    owner = session.owner
    kind = request.get("kind")
    if kind == "request":
        with torch.no_grad():
            intents, slots = owner.request_logits(request_tokens([request["text"]]))
        vocabulary = owner.stream_config["vocabulary"]
        result = {"status": "INTERPRETED_REQUEST", "intent": vocabulary["intents"][int(intents.argmax(-1)[0])],
                  "tags": [vocabulary["tags"][i] for i in slots.argmax(-1)[0].tolist()]}
    elif kind == "read":
        if not request.get("source"):
            raise ValueError("Source attribution is required")
        if not 1 <= len(request["sentences"]) <= 16:
            raise ValueError("Use one to sixteen source sentences")
        with torch.no_grad():
            x, mask = owner.reading_features([request])
            choice = int(owner.reading_logits(x, mask).argmax(-1)[0])
        result = {"status": "ATTRIBUTED_SENTENCE_SELECTION", "sentence": request["sentences"][choice]["text"],
                  "sentence_index": choice, "source": request["source"], "factual_updates": 0}
    elif kind in ("sum", "integral"):
        p = request["coefficients"]
        proposed = owner.study_maps[kind].propose(p)
        certified = check(kind, p, proposed)["accepted"] and independent(kind, p, proposed)
        result = {"status": "CERTIFIED_ALGEBRA" if certified else "NEEDS_ACQUISITION",
                  "operator": kind, "coefficients": proposed, "independent_check": certified}
    elif kind == "imagine":
        grounded = growth.base.base.base.base.base.base.grounded
        result = grounded.imagine(request["request"], acquire=False)
    elif kind == "discovered_route":
        acquisition = growth.base
        if not acquisition.base.state["records"]:
            raise ValueError("No retained independently certified relationship")
        record = acquisition.base.state["records"][int(request.get("route", 0))]
        result = acquisition.base.use(record, request["coefficients"])
    elif kind in {"discoveries", "solve_discovery"}:
        from experiments.self_chosen.runtime import Session as DiscoverySession
        discovery = session
        while not isinstance(discovery, DiscoverySession) and hasattr(discovery, "base"):
            discovery = discovery.base
        if not isinstance(discovery, DiscoverySession):
            raise ValueError("This owner has no admitted self-chosen portfolio")
        if kind == "solve_discovery":
            result = discovery.solve(request["domain"], request["target"], request["observations"])
        else:
            final = read(ROOT / "research-continuation/41_self_chosen_discovery/final.json")
            assessed = {r["id"]: r["result"] for r in final["results"][final["selected"]]["checks"]}
            frontier_final = ROOT / "research-continuation/42_composed_discovery/final.json"
            if frontier_final.exists():
                extra = read(frontier_final)
                assessed.update({r["id"]: r["result"] for r in extra["results"][extra["selected"]]["checks"]})
            result = {"goal": discovery.goal, "records": [{**r, "reserved_evaluation": assessed.get(r["id"])} for r in discovery.records]}
    elif kind == "observed_motion":
        from fractions import Fraction

        from experiments.discovery_observation import point
        if not getattr(session, "evidence", None):
            raise ValueError("No independently checked measured-motion parameters on this owner")
        value = str(request["time"])
        if len(value) > 40:
            raise ValueError("Use a bounded elapsed time")
        t = Fraction(value)
        if not 0 < t <= 1000:
            raise ValueError("Use a positive finite model time up to 1000 seconds")
        key = request.get("evidence_id") or next(iter(session.evidence))
        record = session.evidence[key]
        theta = session.owner.observed_parameters[key].detach().tolist()
        selected = read(ROOT / "research-continuation/41_self_chosen_discovery/observation-selection.json")
        measured_times = record["evaluation"]["times"]
        result = {"status": "EMPIRICAL_MODEL_PREDICTION" if min(measured_times) <= float(t) <= max(measured_times) else "CONDITIONAL_EXTRAPOLATION",
                  "time_since_model_origin_seconds": float(t), "position_metres": point(selected["record"]["proposal"], float(t), theta),
                  "source": record["source"], "evidence_id": key, "measured_event": False,
                  "observed_interval_seconds": [min(measured_times), max(measured_times)]}
    else:
        raise ValueError("Supported tasks: request, read, sum, integral, imagine, discovered_route, discoveries, solve_discovery, observed_motion")
    return {"id": request.get("id"), "kind": kind, "owner": model_identity(owner), "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of attributed/typed tasks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        raise ValueError("Use a fresh output file inside runs")
    tasks = read(args.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("Use a finite batch of one to 256 tasks")
    session, growth = restore(args.store)
    results = []
    for request in tasks:
        try:
            results.append(perform(session, growth, request))
        except (ValueError, KeyError, IndexError) as error:
            results.append({"id": request.get("id"), "kind": request.get("kind"),
                            "status": "RETAINED_OPEN", "reason": str(error), "original_task": request})
        write(output, results)
    print(json.dumps({"owner": model_identity(session.owner), "tasks": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
