"""Independent numerical replay, retained capabilities and persistent online credit."""

import json

import numpy as np
import torch

from experiments.book_learning.runtime import GroundedSession
from experiments.concept_refinement.audit import languages
from workbench.storage import Store

from .common import OUT, ROOT, RUN, digest, read, sha, write
from .model import Investigator
from .runtime import CompletionSession
from .study import subset, summary
from .task import AUDIT_VARIANCE, STOP, PracticeSource, consequences, features


def load_bank(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def numpy_policy(policy, x):
    state = {k: v.detach().numpy() for k, v in policy.state_dict().items()}
    x = (x-state["center"])/state["scale"]
    return (np.tanh(x @ state["net.0.weight"].T+state["net.0.bias"])
            @ state["net.2.weight"].T+state["net.2.bias"])[..., 0]


def resume_training():
    bank = load_bank(RUN / "train.npz")
    x = torch.tensor(features(bank))
    results = []
    for seed in (3211, 3212):
        folder = RUN / str(seed)
        saved = torch.load(folder / "step-0400.pt", weights_only=True)
        expected = torch.load(folder / "step-0800.pt", weights_only=True)
        policy = Investigator()
        policy.load_state_dict(saved["weights"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
        optimizer.load_state_dict(saved["optimizer"])
        rng = torch.Generator()
        rng.set_state(saved["rng"])
        baseline = saved["baseline"]
        traces = read(folder / "trace.json")
        for step in range(401, 801):
            ids = torch.randint(len(x), (64,), generator=rng)
            logits = policy(x[ids])
            probabilities = logits.softmax(-1)
            actions = torch.multinomial(probabilities, 1, generator=rng).squeeze(-1)
            rewards = torch.tensor(consequences(subset(bank, ids.numpy()), actions.numpy())["reward"])
            assert traces[step-1] == {"step": step, "indices": ids.tolist(), "actions": actions.tolist(),
                                      "rewards": rewards.tolist(), "baseline": baseline}
            loss = -((rewards-baseline)*logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(-1)).mean()
            entropy = -(probabilities*logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss-.01*entropy).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            baseline = .98*baseline+.02*float(rewards.mean())
        assert all(torch.equal(v, policy.state_dict()[k]) for k, v in expected["weights"].items())
        assert torch.equal(rng.get_state(), expected["rng"]) and baseline == expected["baseline"]
        from .credit import encode
        assert encode(optimizer.state_dict()) == encode(expected["optimizer"])
        results.append({"seed": seed, "steps_replayed": 400, "exact_weights_optimizer_rng_baseline_trace": True})
    return results


def independent_finals(policy):
    final = read(OUT / "final.json")
    errors = []
    count = 0
    for family in ("matched", "shifted", "omitted"):
        bank = load_bank(RUN / f"final-{family}.npz")
        x = features(bank)
        with torch.no_grad():
            actual = policy(torch.tensor(x)).numpy()
        independent = numpy_policy(policy, x)
        errors.append(float(np.max(np.abs(actual-independent))))
        assert np.allclose(actual, independent, atol=2e-12, rtol=2e-12)
        selected = load_bank(RUN / f"{family}-learned.npz")
        assert np.array_equal(independent.argmax(-1), selected["actions"])
        for control in final["results"][family]:
            result = load_bank(RUN / f"{family}-{control}.npz")
            assert summary(result) == final["results"][family][control]
            # Precision-form conditioning is independent of the production gain update.
            for i, action in enumerate(result["actions"]):
                mean, covariance = bank["mean"][i], bank["covariance"][i]
                if action != STOP and not bank["duplicate"][i, action]:
                    a, variance = bank["design"][i, action], bank["variance"][i, action]
                    precision = np.linalg.inv(covariance)
                    covariance = np.linalg.inv(precision+np.outer(a, a)/variance)
                    mean = covariance @ (precision @ mean+a*bank["outcomes"][i, action]/variance)
                q = bank["target"][i]
                after, v = float(q @ mean), float(q @ covariance @ q)+AUDIT_VARIANCE
                before, bv = result["before"][i], result["before_variance"][i]
                y = bank["audit_y"][i]
                reward = -.5*(np.log(v/bv)+(y-after)**2/v-(y-before)**2/bv)-result["cost"][i]
                assert np.isclose(after, result["after"][i], atol=2e-10, rtol=2e-10)
                assert np.isclose(v, result["after_variance"][i], atol=2e-10, rtol=2e-10)
                assert np.isclose(reward, result["reward"][i], atol=2e-9, rtol=2e-9)
                count += 1
    return {"independent_precision_form_and_logscore": count, "numpy_policy_maximum_error": max(errors),
            "actions_and_published_summaries_match": True}


def without_identity(result):
    return {k: v for k, v in result.items() if k not in {"owner", "parent_owner"}}


def main():
    torch.set_num_threads(1)
    if (OUT / "integration-audit.json").exists():
        raise FileExistsError("Preserve the completed audit")
    replays = resume_training()
    parent_saved = Store(ROOT / "runs/sera-grounded-live").read()
    parent = GroundedSession(parent_saved)
    old = {k: v.clone() for k, v in parent.owner.state_dict().items()}
    before_languages, ids = languages(parent.owner)
    before_math = parent.base.exact_motion([2, 3, 1], "3", "5", "-1")
    subject = next(iter(parent.base.subjects))
    before_empirical = parent.base.predict(subject, [.8]*4)
    reading = ("What is force?", "A force is a push or pull. A measurement has a unit.", "engineering retention probe")
    before_reader = parent.read_passage(*reading)
    taught = [r for r in parent.bank if r["term"] and r["label"] != "other"]
    before_bindings = [without_identity(parent.bind(r["term"], r["text"], r["source"])) for r in taught]
    session = CompletionSession(parent_saved=parent_saved)
    finals = independent_finals(session.owner.completion_policy)
    receipts = []
    directory = ROOT / "runs/VC-owner-audit-001"
    directory.mkdir(exist_ok=False)
    store, previous = Store(directory / "session"), None
    for i, seed in enumerate((32420, 32421, 32422)):
        source, identifier = PracticeSource(seed), f"audit-practice-{i}"
        question = "Improve a conditional acceleration prediction by choosing independent evidence"
        first = session.begin(identifier, question, source)
        assert first["observation_updates_from_imagination"] == 0
        session.decide(identifier)
        if i == 0:
            previous = store.commit(session.snapshot(), previous)
            session = CompletionSession(store.read())
        session.acquire(identifier, source)
        if i == 0:
            uninterrupted = session
            previous = store.commit(session.snapshot(), previous)
            session = CompletionSession(store.read())
            expected = uninterrupted.verify_and_credit(identifier, source)
            actual = session.verify_and_credit(identifier, source)
            assert actual == expected and session.snapshot() == uninterrupted.snapshot()
        else:
            actual = session.verify_and_credit(identifier, source)
        receipts.append(actual)
        # Model weights, evidence, source attribution and original goal survive a full checkpoint.
        previous = store.commit(session.snapshot(), previous)
    restored = CompletionSession(store.read())
    assert restored.snapshot() == session.snapshot()
    assert session.owner is session.grounded.owner is session.grounded.base.study.owner
    assert all(torch.equal(v, session.owner.state_dict()[k]) for k, v in old.items())
    after_languages, after_ids = languages(session.owner)
    assert ids == after_ids
    assert all(torch.equal(a, b) for key in ids for a, b in zip(before_languages[key], after_languages[key], strict=True))
    assert before_math == session.grounded.base.exact_motion([2, 3, 1], "3", "5", "-1")
    assert without_identity(before_empirical) == without_identity(session.grounded.base.predict(subject, [.8]*4))
    assert without_identity(before_reader) == without_identity(session.grounded.read_passage(*reading))
    after_bindings = [without_identity(session.grounded.bind(r["term"], r["text"], r["source"])) for r in taught]
    assert before_bindings == after_bindings and sum(r["admitted"] for r in after_bindings) == 26
    assert session.grounded.gate == parent.gate
    assert session.grounded.base.subjects == parent.base.subjects
    for key in ("goals", "events", "tasks", "concepts"):
        assert getattr(session.grounded, key) == getattr(parent, key)
    assert "momentum-gap-001" in session.grounded.tasks
    result = {"status": "PASS", "actual_shared_owner": True, "protected_tensors_exact": len(old),
              "language_probes_exact": {k: len(v) for k, v in ids.items()},
              "taught_physical_bindings_exact": len(taught), "retained_math_reader_empirical": True,
              "novel_definition_gate": parent.gate, "parent_store_sha256": digest(parent_saved),
              "open_momentum_task_preserved": True, "imagined_factual_observations": 0,
              "training_replay": replays, "independent_final_replay": finals,
              "online_rewards": [r["receipt"]["reward"] for r in receipts], "online_policy_steps": session.credit.steps,
              "exact_interrupted_online_resume": True, "saved_owner": session.snapshot()["owner"],
              "history_revisions_verified": store.verify_history(), "selection_sha256": sha(OUT / "selection.json")}
    write(OUT / "online-example.json", {"receipts": receipts, "candidate": session.candidate("audit-practice-2"),
                                         "recommendation": session.recommend("audit-practice-2")})
    write(OUT / "integration-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
