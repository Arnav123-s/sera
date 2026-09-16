"""Actual-owner acquisition, persistence, withdrawal and fresh relearning."""

import argparse
import csv
import json

import numpy as np
import torch

from experiments.grounded_language.data import examples
from experiments.sparse_mechanisms.owner_weights import capture
from sera.session_state import model_identity

from .runtime import initialize, load, read_csv, save
from .study import RELEASE, ROOT, freeze, sha, write


def generators(x):
    x = np.asarray(x)
    a, b, c = x.T
    return np.column_stack((.5+1.2*a-.4*b*c, b*b+.3*a*c, .6*c-.2*a**3))


def values(x, weights, drift=False):
    x = np.asarray(x)
    return generators(x)@weights+(.6*x[:, 0]**2 if drift else 0)


def material(path, seed, weights, count=48, drift=False, off_family=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (count+80, 3))
    y = values(x, weights, drift)+(np.sin(6*x[:, 0]) if off_family else 0)
    roles = ["fit"]*count+["selection"]*16+["calibration"]*64
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "x3", "y", "role"])
        writer.writerows([*coordinate, output, role] for coordinate, output, role in zip(x, y, roles, strict=True))
    return read_csv(path, "teach")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    folder = RELEASE/args.name
    freeze(folder, {"id": args.name, "directory": args.directory,
                    "parent": "Exact current SERA owner, restored and extended on an isolated descendant",
                    "sequence": "Four observed source tasks; new task with 12 fitting labels; restart; contradictory current observation; withdrawal; fresh 20-label fit with 16 selection and 64 calibration; restart; unchanged old tasks; off-family abstention.",
                    "claims": "Useful numerical readouts on the actual persistent owner. Explicit supplied input interface, polynomial vocabulary and observed synthetic calibration functions. No new language skill, shared-recurrent transfer or learned eta."})
    inputs = folder/"examples"
    inputs.mkdir()
    pointer_path = ROOT/"runs/sera-workbench/current.json"
    pointer_before = pointer_path.read_bytes()
    directory = ROOT/args.directory
    session = initialize(directory)
    base_state = {key: value.detach().clone() for key, value in session.owner.state_dict().items()}
    initial_owner = model_identity(session.owner)
    language_prompts = examples(981011, 64, "final")
    _, old_logits = capture(session.owner, language_prompts)
    query = [[-.5, .2, .8], [0., 0., 0.], [.8, -.3, .1], [-.2, -.4, -.7]]
    steps = []
    weights = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., -.5, .8]]
    for index, coefficient in enumerate(weights):
        task = f"calibration_source_{index+1}"
        batch = material(inputs/f"{task}.csv", 981101+101*index, coefficient)
        outcome = session.teach(task, "calibration", batch)
        assert outcome["status"] == "ACCEPTED_EMPIRICAL"
        save(directory, session)
        steps.append(outcome)
    coefficient = [.7, -1.3, .4]
    name = "calibration_estimator"
    batch = material(inputs/"new_task.csv", 981911, coefficient, count=12)
    outcome = session.teach(name, "calibration", batch)
    assert outcome["status"] == "ACCEPTED_EMPIRICAL"
    save(directory, session)
    learned_identity = model_identity(session.owner)
    prediction = session.predict(name, query)
    before_drift = prediction["outputs"]
    expected = values(query, coefficient)
    error = float(np.max(abs(np.array(before_drift)-expected)))
    assert error < 1e-8
    steps.append(outcome)
    session = load(directory)
    assert model_identity(session.owner) == learned_identity
    assert session.predict(name, query)["outputs"] == before_drift
    repeat_rejected = False
    try:
        session.teach(name, "calibration", batch)
    except ValueError as exc:
        repeat_rejected = "fresh calibration" in str(exc)
    assert repeat_rejected
    assert session.predict(name, [[1.1, 0., 0.]])["status"] == "OUTSIDE_DOMAIN"
    observed_x = [[.9, -.25, .35]]
    observed_y = values(observed_x, coefficient, drift=True).tolist()
    drift = session.observe(name, observed_x, observed_y)
    assert drift["status"] == "WITHDRAWN"
    save(directory, session)
    assert session.predict(name, query)["outputs"] == []
    duplicate_rejected = False
    try:
        session.observe(name, observed_x, observed_y)
    except ValueError as exc:
        duplicate_rejected = "already recorded" in str(exc)
    assert duplicate_rejected
    updated = material(inputs/"changed_task.csv", 982013, coefficient, count=20, drift=True)
    correction = session.teach(name, "calibration", updated)
    assert correction["status"] == "ACCEPTED_EMPIRICAL" and correction["generation"] == 2
    save(directory, session)
    steps.append(correction)
    final_predictions = session.predict(name, query)
    corrected_error = float(np.max(abs(np.array(final_predictions["outputs"])-values(query, coefficient, drift=True))))
    assert corrected_error < 1e-8
    unrelated = material(inputs/"outside_family.csv", 982111, [0., 0., 0.], count=48, off_family=True)
    rejected = session.teach("outside_family", "calibration", unrelated)
    assert rejected["status"] == "WITHHELD"
    save(directory, session)
    assert session.predict("outside_family", query)["outputs"] == []
    final_owner = model_identity(session.owner)
    session = load(directory)
    assert model_identity(session.owner) == final_owner
    assert session.predict(name, query)["outputs"] == final_predictions["outputs"]
    retention = 0.
    for index, coefficient_ in enumerate(weights):
        prediction_ = session.predict(f"calibration_source_{index+1}", query)["outputs"]
        retention = max(retention, float(np.max(abs(np.array(prediction_)-values(query, coefficient_)))))
    assert retention < 1e-8
    _, new_logits = capture(session.owner, language_prompts)
    assert all(np.array_equal(a, b) for a, b in zip(old_logits, new_logits, strict=True))
    assert all(torch.equal(value, session.owner.state_dict()[key]) for key, value in base_state.items())
    formal = session.learner.solve(3, 2, 4)
    assert formal["solutions"] == [2]
    assert pointer_path.read_bytes() == pointer_before
    assert initial_owner != final_owner
    # Human-usable inputs and persistent state, plus exact portable descendant records.
    with (inputs/"queries.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "x3"])
        writer.writerows(query)
    with (inputs/"changed_observation.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "x3", "y"])
        writer.writerows([*x, y] for x, y in zip(observed_x, observed_y, strict=True))
    import zipfile
    with zipfile.ZipFile(folder/"session-records.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(directory.rglob("*.json")):
            z.write(path, path.relative_to(directory).as_posix())
    output = {"status": "PASS", "parent_owner": session.parent["owner_sha256"], "initial_descendant": initial_owner,
              "final_owner": final_owner, "steps": steps, "drift": drift, "off_family": rejected,
              "new_task_max_error": error, "corrected_max_error": corrected_error,
              "old_task_max_error": retention, "parent_tensors_unchanged": True,
              "old_language_logits_bitwise_equal": True, "shared_owner_aliases": True,
              "formal_route": formal, "repeat_calibration_rejected": repeat_rejected,
              "duplicate_observations_rejected": duplicate_rejected, "outside_domain_withheld": True,
              "withdrawn_task_returns_no_output": True, "saved_owner_identity_exact": True,
              "live_pointer_unchanged": True, "revisions": len(list((directory/"revisions").glob("*.json"))),
              "queries": query, "before_change": before_drift, "after_change": final_predictions["outputs"],
              "current_pointer_sha256": sha(directory/"current.json"), "directory": str(directory)}
    write(folder/"result.json", output)
    print(json.dumps({k: v for k, v in output.items() if k not in ("steps", "off_family", "formal_route")}))


if __name__ == "__main__":
    main()
