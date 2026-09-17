"""Independent scalar outcome audit and an actual persistent-owner demonstration."""

import argparse
import copy
import math

import numpy as np

from experiments.language_inquiry.study import ROOT, read, write
from sera.session_state import model_identity
from workbench.storage import Store

from .runtime import ConstraintRuntime
from .study import OUT


def scalar(model, controls, parameters, torque=0.):
    angle, velocity = model["angle"], model["velocity"]
    a, gain, bias = parameters
    for action in controls:
        velocity = max(-.65, min(.65, a * velocity + gain * action + bias + torque * math.sin(2 * angle)))
        angle += velocity
    return [model["center"][0] + model["radius"] * math.cos(angle),
            model["center"][1] + model["radius"] * math.sin(angle)]


def verify():
    rows = read(OUT / "evaluation/numerical-records.json")
    cases = read(OUT / "evaluation/cases.json")
    offset = len(cases)
    extra_cases = read(OUT / "followup/cases.json")
    cases.extend({**c, "assessor_torque": .08 if c["family"] == "omitted_torque" else 0.} for c in extra_cases)
    rows.extend({**r["answer"], "case": r["case"] + offset, "assessed_distance_m": r["assessed_distance_m"],
                 "assessed_success": r["assessed_success"]} for r in read(OUT / "followup/records.json"))
    maximum = 0.
    for row in rows:
        case = cases[row["case"]]
        model, controls = case["model"], row["controls"]
        if any(abs(v) > 1 for v in controls) or row["occurrence"] != "NOT_ESTABLISHED":
            raise AssertionError("Control or evidence contract changed")
        values = [scalar(model, controls, p) for p in model["parameters"]]
        errors = [math.dist(v, model["target"]) for v in values]
        assessed = math.dist(scalar(model, controls, model["mean"], case["assessor_torque"]), model["target"])
        maximum = max(maximum, max(abs(a - b) for x, y in zip(values, row["parameter_endpoints"]) for a, b in zip(x, y)),
                      abs(errors[0] - row["nominal_distance_m"]), abs(assessed - row["assessed_distance_m"]))
        if (errors[0] <= .03) != (row["status"] == "CONDITIONAL_WITNESS") or (assessed <= .03) != row["assessed_success"]:
            raise AssertionError("Independent witness or assessment differs")
    if maximum > 1e-12:
        raise AssertionError("Independent numerical replay differs")
    write(OUT / "independent-audit.json", {"records": len(rows), "max_absolute_difference": maximum,
                                          "scope": "Scalar recurrence, static parameters, endpoints, threshold decisions and assessor law; not a second training study"})
    print({"verified": len(rows), "max_error": maximum}, flush=True)


def integrate():
    directory = ROOT / "runs/sera-constraints"
    directory.mkdir(parents=True, exist_ok=False)
    store = Store(directory)
    selected = read(ROOT / "runs/CI-fits-final-001/selected.json")
    runtime = ConstraintRuntime(selected)
    previous = None
    transcript = []

    def save(label, result):
        nonlocal previous
        previous = store.commit(runtime.snapshot(), previous)
        transcript.append({"operation": label, "response": copy.deepcopy(result),
                           "revision": previous["revision"], "owner": previous["owner"]})

    save("initialize", runtime.status())
    save("ask with missing position", runtime.ask("reach", "can orbit reach beacon"))
    save("ask reversed roles", runtime.ask("reverse", "can beacon reach orbit"))
    save("ask historical occurrence", runtime.ask("event", "did orbit reach beacon"))
    save("acquire grounded position", runtime.acquire("beacon"))
    save("work four steps", runtime.work_on("reach", 4))
    first = runtime.snapshot()
    expected = runtime.work_on("reach", 8)
    restored = ConstraintRuntime(selected, first)
    actual = restored.work_on("reach", 8)
    if actual != expected or model_identity(runtime.owner) != model_identity(restored.owner):
        raise AssertionError("Interrupted reasoning differs from uninterrupted execution")
    runtime = restored
    save("restore and finish eight steps", actual)
    save("tested passive wording", runtime.ask("wording", "please beacon can be reached by orbit"))
    save("solve with fixed residual gate", runtime.solve("wording"))
    save("unqualified new wording stays missing", runtime.ask("unqualified", "please can orbit reach beacon"))
    save("new observed transition", runtime.advance(0))
    if runtime.jobs["reach"]["status"] != "stale":
        raise AssertionError("Old plan survived changed physical evidence")
    save("rebase after observation", runtime.rebase("reach"))
    save("finish corrected situation", runtime.work_on("reach", 12))
    # Leave a genuine partially completed task for further investigation.
    save("preserve unfinished imagination", runtime.ask("unfinished", "imagine orbit reaching beacon"))
    save("partial work", runtime.work_on("unfinished", 2))
    restored = ConstraintRuntime(selected, store.read())
    if model_identity(restored.owner) != model_identity(runtime.owner):
        raise AssertionError("Persistent learner changed on restoration")
    transcript.append({"operation": "final restore", "response": restored.status(), "history_revisions": store.verify_history()})
    write(OUT / "integration-transcript.json", transcript)
    print({"store": str(directory), "owner": model_identity(runtime.owner), "revisions": store.verify_history()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "integrate"))
    args = parser.parse_args()
    np.seterr(all="raise")
    verify() if args.mode == "verify" else integrate()


if __name__ == "__main__":
    main()
