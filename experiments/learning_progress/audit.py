"""Independent score/credit replay, exact continuation and owner retention."""

import copy
import json

import numpy as np
import torch

from experiments.concept_refinement.audit import languages
from experiments.quest_portfolio.methods import SPECS, cases, execute
from sera.session_state import model_identity
from workbench.storage import Store

from .common import OUT, PREFIXES, ROOT, RUN, contracts, read, sha, write
from .data import load_data
from .runtime import LearningSession
from .study import pack, restore


def independent(row):
    value = np.array(row["logits"], dtype=float)
    shifted = value-value.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    if row["gold"]:
        gold = np.asarray(row["gold"], dtype=bool)
        ce = -np.log((probability*gold).sum(1)).mean()
        acc = gold[np.arange(len(value)), value.argmax(1)].mean()
    else:
        labels = np.array(row["labels"])
        ce = -np.log(probability[np.arange(len(value)), labels]).mean()
        acc = (value.argmax(1) == labels).mean()
    return float(ce), float(acc)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def check_cohort(cohort):
    path = ROOT / cohort["witness"]["path"]
    if sha(path) != cohort["witness"]["sha256"]:
        raise ValueError("Changed saved predictions")
    errors, count = [], 0
    with np.load(path, allow_pickle=False) as arrays:
        for skill, row in cohort["skills"].items():
            ce, acc = independent({**row, "logits": arrays[skill]})
            errors.extend([abs(ce-row["loss"]), abs(acc-row["accuracy"])])
            count += row["count"]
    return errors, count


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Completed audit is immutable")
    final, data = read(OUT / "final.json"), load_data()
    errors, checked = [], 0
    for record in final["results"]:
        for cohort in list(record["cohorts"].values())+list(record["initial"].values()):
            new_errors, n = check_cohort(cohort)
            errors.extend(new_errors)
            checked += n
    for record in final["factorial"]:
        new_errors, n = check_cohort(record["after"])
        errors.extend(new_errors)
        checked += n
    if max(errors) > 2e-6:
        raise AssertionError("Independent scoring disagrees")
    parent = read(RUN / "parent.json")
    session = LearningSession(parent=parent)
    original = {k: v.clone() for k, v in session.owner.state_dict().items()}
    language_before, ids = languages(session.owner)
    midpoint = torch.load(RUN / "learned-3401-step-040.pt", weights_only=True)
    restore(session, midpoint)
    while any(session.remaining):
        session.step(data["teach1"], data["probe"], "learned")
    expected = torch.load(RUN / "learned-3401-step-048.pt", weights_only=True)
    if not equal(pack(session), expected):
        raise AssertionError("Exact successor continuation changed")
    name = final["selected"] if final["knowledge_admitted"] else "initial-3401"
    path = RUN / (name+"-step-096.pt" if final["knowledge_admitted"] else name+".pt")
    selected = torch.load(path, weights_only=True)
    if selected["seed"] != session.seed:
        session = LearningSession(parent=parent, seed=selected["seed"])
    restore(session, selected)
    session.refresh()
    language_after, new_ids = languages(session.owner)
    language_equal = equal(language_before, language_after) and equal(ids, new_ids)
    protected = [k for k in original if not k.startswith(PREFIXES+("learning_policy.",))]
    preserved = all(torch.equal(original[k], session.owner.state_dict()[k]) for k in protected)
    rows, answers = cases(34611, 64)
    for row, target in zip(rows, answers, strict=True):
        # No answer-dependent method selection.
        method = ("time", "impulse", "work_positive", "work_negative").index(row["scope"])
        if abs(execute(SPECS[method], row, session.base.base)-target) > 1e-10:
            raise AssertionError("Retained exact mechanics route changed")
    snap = session.snapshot()
    resumed = LearningSession(saved=snap)
    exact_snapshot = snap == resumed.snapshot()
    if not (language_equal and preserved and exact_snapshot):
        raise AssertionError("Owner, language or serialized continuation changed")
    store = Store(ROOT / "runs/sera-learning-live")
    if store.read() is not None:
        raise FileExistsError("Never overwrite an existing learner")
    store.commit(snap, None)
    # Public research archives retain checkpoint bytes; the local issuer key remains local.
    evidence = {"contracts": contracts(), "numpy_scored_predictions": checked,
                "maximum_score_error": max(errors), "exact_midpoint_continuation": True,
                "snapshot_equal": exact_snapshot, "protected_tensors": len(protected),
                "protected_equal": preserved, "language_equal": language_equal,
                "language_probes": sum(map(len, ids.values())), "mechanics_checks": len(rows),
                "owner": model_identity(session.owner), "store": "runs/sera-learning-live",
                "selected": name, "default_controller": final["default"],
                "novel_definition_gate": session.base.base.grounded.gate["autonomous"],
                "prior_quests": copy.deepcopy(parent["board"]),
                "original_goal": session.goal}
    write(OUT / "audit.json", evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "prior_quests"}, indent=2))


if __name__ == "__main__":
    main()
