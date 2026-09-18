"""Independent score replay, actual-owner retention and exact continuation checks."""

import json

import numpy as np
import torch

from experiments.self_study.algebra import check, independent
from sera.session_state import model_identity

from .common import DECISIONS, OUT, PREFIXES, ROOT, RUN, SKILLS, contracts, read, sha, write
from .data import load
from .model import identity, knowledge
from .runtime import GrowthSession, validate_events
from .study import load_state


def replay_scores(record):
    path = ROOT / record["witness"]["path"]
    if sha(path) != record["witness"]["sha256"]:
        raise ValueError("Changed independent prediction witness")
    errors, items = [], 0
    with np.load(path, allow_pickle=False) as values:
        for skill in SKILLS:
            logits = values[skill+":logits"].astype(np.float64)
            labels, gold = values[skill+":labels"], values[skill+":gold"]
            shifted = logits-logits.max(-1, keepdims=True)
            log_probs = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
            if skill == "reading":
                loss = -np.log((np.exp(log_probs)*gold).sum(-1)).mean()
                correct = int(gold[np.arange(len(gold)), logits.argmax(-1)].sum())
            else:
                loss = -log_probs[np.arange(len(labels)), labels].mean()
                correct = int((logits.argmax(-1) == labels).sum())
            expected = record["skills"][skill]
            errors.append(abs(float(loss)-expected["loss"]))
            if correct != expected["correct"]:
                raise AssertionError("Independent label accuracy disagrees")
            items += len(labels)
    if max(errors) > 2e-6:
        raise AssertionError("Independent cross-entropy replay disagrees")
    return {"items": items, "maximum_loss_difference": max(errors)}


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve completed independent audit")
    final, procedure = read(OUT / "final.json"), read(OUT / "procedure.json")
    reports = [replay_scores(final["baseline"])]
    reports += [replay_scores(row["result"]) for row in final["results"]]
    reports += [replay_scores(row["after"]) for row in procedure["trials"]]
    parent = read(RUN / "parent.json")
    session = GrowthSession(parent)
    before = {k: v.clone() for k, v in session.owner.state_dict().items() if not k.startswith((*PREFIXES, "growth_policy."))}
    selected = final["selected"] or "balanced-3801"
    chosen = load_state(RUN / f"{selected}-{DECISIONS:04d}.pt")
    session.restore(chosen)
    protected = all(torch.equal(v, session.owner.state_dict()[k]) for k, v in before.items())
    if not protected:
        raise AssertionError("Unapproved inherited tensor changed")
    mathematical = []
    rng = np.random.default_rng(38931)
    for domain in ("integral", "sum"):
        for _ in range(32):
            p = rng.integers(-8, 9, 4).tolist()
            proposal = session.owner.study_maps[domain].propose(p)
            accepted = check(domain, p, proposal)["accepted"] and independent(domain, p, proposal)
            mathematical.append({"domain": domain, "input": p, "proposal": proposal, "accepted": accepted})
    if not all(x["accepted"] for x in mathematical):
        raise AssertionError("Previously acquired polynomial route regressed")
    grounded = session.base.base.base.base.base.base.grounded
    if grounded.gate["autonomous"]:
        raise AssertionError("Novel-definition gate was bypassed")
    owner = model_identity(session.owner)
    identity_before = identity(knowledge(session.owner))
    training, probe = load("train"), load("probe")
    for arm in ("balanced", "self_directed"):
        path = RUN / f"{arm}-3801-{DECISIONS-128:04d}.pt"
        session.restore(load_state(path))
        for _ in range(128):
            session.step(training, probe, arm)
        expected = load_state(RUN / f"{arm}-3801-{DECISIONS:04d}.pt")
        validate_events(session.events, session.highwater)
        actual = session.state()
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
            return a == b
        if not equal(actual, expected):
            raise AssertionError("Sustained practice did not resume exactly: "+arm)
    session.restore(chosen)
    if identity(knowledge(session.owner)) != identity_before or model_identity(session.owner) != owner:
        raise AssertionError("Audit failed to return to selected original goal")
    write(OUT / "audit.json", {"contracts": contracts(), "owner": owner, "selected": selected,
        "independent_prediction_items": sum(r["items"] for r in reports),
        "maximum_loss_difference": max(r["maximum_loss_difference"] for r in reports),
        "exact_resume": {"balanced_decisions": 128, "self_directed_decisions": 128, "matched": True},
        "protected_tensors": len(before), "protected_equal": protected, "mathematics": mathematical,
        "novel_definition_gate": grounded.gate["autonomous"],
        "same_actual_owner": session.owner is session.base.owner is session.base.base.owner,
        "knowledge_identity": identity_before, "evaluation_mutations": 0})
    print(json.dumps({"audit_complete": True, "owner": owner, "independent_items": sum(r["items"] for r in reports)}))


if __name__ == "__main__":
    main()
