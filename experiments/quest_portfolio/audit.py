"""Independent numerical replay, actual-owner lifecycle and qualification audit."""

import argparse
import copy

import numpy as np
import torch

from experiments.verified_completion.credit import encode
from experiments.verified_completion.task import PracticeSource
from sera.session_state import model_identity
from workbench.storage import Store

from .assessor import challenge
from .common import OUT, ROOT, RUN, contracts, digest, model_hash, read, write
from .methods import contexts, deploy, features
from .runtime import QuestSession, protected_hash
from .study import train_steps


def replay():
    destination = OUT / "replay.json"
    if destination.exists():
        raise FileExistsError("Keep the completed replay")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Selected implementation changed")
    session = QuestSession(parent=read(RUN / "parent.json"), authority=RUN / "unused-assessor")
    profiles = np.asarray(read(RUN / "practice-screen.json")["profiles"])
    weights = contexts(2048, 33001)
    records = []
    for record in selection["all_development"]:
        path = ROOT / record["checkpoint"]
        middle = torch.load(path.with_name("step-0150.pt"), weights_only=True, map_location="cpu")
        expected = torch.load(path, weights_only=True, map_location="cpu")
        session.base._mutation(lambda: session.owner.quest_policy.load_state_dict(middle["weights"]))
        optimizer = torch.optim.Adam(session.owner.quest_policy.parameters(), lr=.003)
        optimizer.load_state_dict(middle["optimizer"])
        rng = torch.Generator()
        rng.set_state(middle["rng"])
        learner = session.base.grounded.base.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        baseline = train_steps(session.owner.quest_policy, optimizer, rng, weights, profiles, 150, 300,
                               middle["baseline"], middle["trace"])
        learner.rebind(facts, library)
        session.base._refresh()
        observed = {"weights": session.owner.quest_policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": rng.get_state(), "baseline": baseline, "trace": middle["trace"]}
        if encode(observed) != encode({k: expected[k] for k in observed}):
            raise AssertionError("Training replay diverged; preserve before any repair")
        records.append({"arm": record["arm"], "seed": record["seed"], "from_step": 150, "to_step": 300, "exact": True})
    selected = torch.load(ROOT / selection["checkpoint"], weights_only=True, map_location="cpu")["weights"]
    session.base._mutation(lambda: session.owner.quest_policy.load_state_dict(selected))
    w = contexts(128, 33591, shifted=True)
    covered = np.zeros((128, 4))
    visits = np.zeros((128, 7))
    maximum_error = 0.
    for step in range(4):
        claims = np.concatenate((np.eye(4), [[0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]]))
        inputs = np.empty((128, 7, 12))
        for row in range(128):
            for method in range(7):
                inputs[row, method] = [*(w[row]*(1-covered[row])), *claims[method],
                                       [3, 1, 2, 2, .5, .1, 0][method], float(visits[row, method] > 0),
                                       step/5, float(method == 6)]
        logits = (np.tanh(inputs@selected["net.0.weight"].numpy().T+selected["net.0.bias"].numpy())
                  @selected["net.2.weight"].numpy().T+selected["net.2.bias"].numpy()).squeeze(-1)
        with torch.no_grad():
            reference = session.owner.quest_policy(torch.from_numpy(features(w, covered, visits, step))).numpy()
        maximum_error = max(maximum_error, float(np.abs(logits-reference).max()))
        if not np.array_equal(logits.argmax(-1), reference.argmax(-1)):
            raise AssertionError("Independent action replay failed")
        action = logits.argmax(-1)
        visits[np.arange(128), action] += 1
        padded = np.concatenate((profiles, np.zeros((1, 4))))
        covered = np.maximum(covered, padded[action])
    rows, truth = challenge(33581, 128)
    predictions = [deploy([0, 1, 2, 3], row, session.base)[0] for row in rows]
    method_error = max(abs(a-b) for a, b in zip(predictions, truth, strict=True))
    if method_error > 1e-8:
        raise AssertionError("Independent mechanics reference disagreed")
    write(destination, {"training_resumptions": records, "numpy_max_logit_error": maximum_error,
                        "independent_actions": 512, "fresh_mechanics_checks": 128, "maximum_mechanics_error": method_error,
                        "contracts": contracts()})
    print("Four exact training resumptions, 512 independent actions and 128 actual-owner route checks passed.")


def lifecycle():
    destination = OUT / "integration-audit.json"
    directory = ROOT / "runs/QP-owner-audit-001"
    if destination.exists() or directory.exists():
        raise FileExistsError("Keep the completed lifecycle; declare a repair directory if needed")
    directory.mkdir()
    session = QuestSession(parent=read(RUN / "parent.json"), authority=directory / "assessor", selected=True)
    store = Store(directory / "session")
    previous = None

    def persist(value):
        nonlocal previous
        previous = store.commit(value, previous)

    session.persist = persist
    old_tensors = {k: v.clone() for k, v in session.owner.state_dict().items() if not k.startswith("quest_policy.")}
    old_tasks = copy.deepcopy(session.base.grounded.tasks)
    old_subjects = copy.deepcopy(session.base.grounded.base.subjects)
    old_protected = protected_hash(session.owner)
    old_credit = session.base.credit.steps
    exact_before = session.base.grounded.base.exact_motion(["2", "3", "1"], "3", "5", "-1")
    key = session.board.open("velocity-routes", "Find multiple independently checked routes to final velocity.")
    persist(session.snapshot())
    failed = session.assess("velocity-routes")
    if failed["qualified"] or failed["correct"] != 0:
        raise AssertionError("Empty portfolio acquired a capability")
    attempts = [session.practice("velocity-routes", controller="balanced") for _ in range(5)]
    first = session.assess("velocity-routes")
    if not first["qualified"] or len(session.board.quests[key]["methods"]) != 4:
        raise AssertionError("Four independently checked routes did not qualify")
    before = session.snapshot()
    restored = QuestSession(before)
    if restored.snapshot() != before:
        raise AssertionError("Exact qualified restart failed")
    session = restored
    session.persist = persist
    session.board.open("maintenance", "Practise alternate time and impulse procedures.", ["time", "impulse"])
    updates = []
    for _ in range(4):
        result = session.practice("maintenance", controller="learned")
        updates.append(result)
        if result.get("updated"):
            break
    if not any(r.get("updated") for r in updates) or session.board.quests[key]["state"] != "STALE":
        raise AssertionError("A checked on-policy update did not stale the previous qualification")
    source = PracticeSource(33572)
    session.base.begin("portfolio-completion", "Check a missing response while retaining several methods.", source)
    session.base.decide("portfolio-completion")
    session.base.acquire("portfolio-completion", source)
    lower = session.base.verify_and_credit("portfolio-completion", source)
    session.sync_owner()
    persist(session.snapshot())
    final = session.assess("velocity-routes")
    if not final["qualified"] or final["attempt"] != 3:
        raise AssertionError("Fresh revalidation or global attempt counting failed")
    example = {"mechanism": "constant_mechanics", "origin": "SUPPLIED_CONDITIONAL_MECHANICS", "units": "SI",
               "scope": "impulse", "assumptions": {"constant_mass": True}, "m": "3", "j": "12", "v0": "0"}
    solved = session.solve("velocity-routes", example)
    if solved["answer"] != 4.:
        raise AssertionError("Usable original-task return failed")
    exact_after = session.base.grounded.base.exact_motion(["2", "3", "1"], "3", "5", "-1")
    retention = session.retention()
    if (protected_hash(session.owner) != old_protected or retention["language_max_error"] != 0
            or old_tasks != session.base.grounded.tasks or old_subjects != session.base.grounded.base.subjects
            or exact_before["result"] != exact_after["result"]):
        raise AssertionError("Inherited capability or original open goal changed")
    changed = [k for k, v in old_tensors.items() if not torch.equal(session.owner.state_dict()[k], v)]
    if any(not k.startswith("completion_policy.") for k in changed):
        raise AssertionError("Unintended inherited parameter update")
    restored = QuestSession(session.snapshot())
    if restored.snapshot() != session.snapshot():
        raise AssertionError("Post-learning exact restart failed")
    persist(session.snapshot())
    live = ROOT / "runs/sera-quests-live"
    if live.exists():
        raise FileExistsError("Preserve an existing live quest owner")
    Store(live).commit(session.snapshot(), None)
    write(OUT / "example-impulse.json", example)
    write(OUT / "example-result.json", solved)
    write(destination, {"owner": model_identity(session.owner), "actual_shared_owner": True,
                        "initial_assessment": failed, "first_qualification": first, "revalidation": final,
                        "practice_attempts": attempts, "online_maintenance": updates, "lower_level_update": lower,
                        "retention": retention, "inherited_tensors_before": len(old_tensors),
                        "inherited_tensors_changed": changed, "old_tasks_preserved": list(old_tasks),
                        "physical_definitions": sum(r["label"] != "other" for r in session.base.grounded.bank),
                        "lower_policy_updates_before": old_credit, "lower_policy_updates_after": session.base.credit.steps,
                        "exact_restarts": 2, "history_revisions": store.verify_history(),
                        "example": solved, "live_store": "runs/sera-quests-live", "factual_updates": 0,
                        "head_sha256": model_hash(session.owner.quest_policy), "state_digest": digest(session.snapshot())})
    print("Qualified four routes, resumed exactly, updated both policy levels, revalidated and returned 4 m/s.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "lifecycle", "examples"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "replay":
        replay()
    elif args.action == "lifecycle":
        lifecycle()
    else:
        examples()


def examples():
    """Three supplied descriptions of one physical scenario return the same value."""
    destination = OUT / "alternative-example.json"
    if destination.exists():
        raise FileExistsError("Preserve the worked example")
    session = QuestSession(Store(ROOT / "runs/sera-quests-live").read())
    original = session.snapshot()
    common = {"mechanism": "constant_mechanics", "origin": "SUPPLIED_CONDITIONAL_MECHANICS", "units": "SI", "v0": "0"}
    views = {
        "time": dict(common, scope="time", a="2", t="2", assumptions={"constant_acceleration": True}),
        "impulse": dict(common, scope="impulse", j="12", m="3", assumptions={"constant_mass": True}),
        "work": dict(common, scope="work_positive", w="24", m="3",
                     assumptions={"constant_mass": True, "positive_final_velocity": True}),
    }
    results = {name: session.solve("velocity-routes", case) for name, case in views.items()}
    if any(result["answer"] != 4. for result in results.values()) or session.snapshot() != original:
        raise AssertionError("Alternative routes must agree and leave learned state unchanged")
    for name, case in views.items():
        write(OUT / ("example-"+name+".json"), case)
    write(destination, {"goal": "Final speed of a 3 kg body from rest under a 6 N force for 2 seconds",
                        "supplied_consistent_views": views, "results": results,
                        "state_unchanged": True, "learned_method_weights": model_hash(session.owner.quest_policy)})
    print("Three routes through the qualified live owner returned 4 m/s with unchanged state.")


if __name__ == "__main__":
    main()
