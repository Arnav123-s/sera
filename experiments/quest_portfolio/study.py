"""Frozen finite reward comparison on the continuing owner's method weights."""

import argparse
import copy

import numpy as np
import torch

from workbench.storage import Store

from .common import OUT, ROOT, RUN, contracts, model_hash, read, sha, write
from .methods import SPECS, STOP, cases, contexts, deploy, features, screen
from .runtime import QuestSession, protected_hash

COST = np.array([s["cost"] for s in SPECS]+[0.])


def rollout(policy, weights, profiles, rng, *, arm="verified", controller="sample", train=False):
    n = len(weights)
    profiles = np.concatenate((profiles, np.zeros((1, 4))))
    covered, visits = np.zeros((n, 4)), np.zeros((n, 7))
    choices, rewards, logp, entropies = [], [], [], []
    total_cost, verified_return = np.zeros(n), np.zeros(n)
    rows = np.arange(n)
    for step in range(5):
        value = torch.from_numpy(features(weights, covered, visits, step))
        with torch.set_grad_enabled(train):
            logits = policy(value)
            probabilities = logits.softmax(-1)
            if step == 4:
                action = visits[:, :6].argmin(-1)
            elif controller == "sample":
                action = torch.multinomial(probabilities.detach(), 1, generator=rng).squeeze(-1).numpy()
            elif controller == "greedy":
                action = logits.detach().argmax(-1).numpy()
            elif controller == "balanced":
                action = np.full(n, step)
            elif controller == "useful_random":
                action = torch.randint(4, (n,), generator=rng).numpy()
            elif controller == "coverage":
                gains = np.einsum("nd,ad->na", weights*(1-covered), profiles)
                gain_bonus = .03*((profiles.sum(-1)>0)[None, :])*(visits == 0)
                action = (gains+gain_bonus-.01*COST).argmax(-1)
            else:
                raise ValueError("Unknown frozen controller")
            if step != 4:
                logp.append(logits.log_softmax(-1)[rows, action])
                entropies.append(-(probabilities*logits.log_softmax(-1)).sum(-1))
        gained = (weights*(1-covered)*profiles[action]).sum(-1)
        bonus = .03*((profiles[action].sum(-1)>0) & (visits[rows, action] == 0))
        reward = gained+bonus-.01*COST[action]
        verified_return += reward
        rewards.append(reward if arm == "verified" else (weights*profiles[action]).sum(-1)-.01*COST[action])
        visits[rows, action] += 1
        covered = np.maximum(covered, profiles[action])
        total_cost += COST[action]
        choices.append(action.copy())
    reward_tensor = torch.from_numpy(np.asarray(rewards))
    future = reward_tensor.flip(0).cumsum(0).flip(0)
    return {"choices": np.stack(choices, axis=-1), "coverage": (weights*covered).sum(-1),
            "methods": (visits[:, :4] > 0).sum(-1), "cost": total_cost, "verified_utility": verified_return,
            "own_reward": reward_tensor.sum(0).numpy(), "future": future,
            "logp": torch.stack(logp), "entropy": torch.stack(entropies)}


def summary(result):
    return {k: float(result[k].mean()) for k in ("coverage", "methods", "cost", "verified_utility", "own_reward")} | {
        "repeated_attempt_fraction": float(np.mean([len(row)-len(set(row)) for row in result["choices"]])/5)}


def train_steps(policy, optimizer, rng, weights, profiles, start, stop, baseline, trace):
    for step in range(start, stop):
        indices = torch.randint(len(weights), (64,), generator=rng).numpy()
        result = rollout(policy, weights[indices], profiles, rng, arm=trace["arm"], train=True)
        advantage = result["future"][:4]-baseline
        loss = -(advantage.detach()*result["logp"]).mean()-.01*result["entropy"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        baseline = .95*baseline+.05*float(result["future"][:4].mean())
        trace["rows"].append({"step": step+1, **summary(result), "loss": float(loss.detach())})
    return baseline


def save(path, policy, optimizer, rng, baseline, step, trace):
    torch.save({"weights": copy.deepcopy(policy.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": rng.get_state(), "baseline": baseline, "step": step, "trace": copy.deepcopy(trace),
                "contracts": contracts()}, path)


def train():
    if (OUT / "selection.json").exists() or RUN.exists():
        raise FileExistsError("Preserve completed training; resume its exact checkpoints or declare a repair")
    RUN.mkdir(parents=True)
    train_weights, development = contexts(2048, 33001), contexts(256, 33002)
    profiles, errors = screen()
    write(RUN / "practice-screen.json", {"profiles": profiles.tolist(), "counterexample_indices": errors})
    parent = Store(ROOT / "runs/sera-completion-live").read()
    if parent is None:
        raise ValueError("Reconcile and continue the actual saved completion owner")
    write(RUN / "parent.json", parent)
    session = QuestSession(parent=parent, authority=RUN / "unused-assessor")
    retained = protected_hash(session.owner)
    parent_weights = {k: t.clone() for k, t in session.owner.state_dict().items() if not k.startswith("quest_policy.")}
    records = []
    for arm in ("verified", "repeated_success"):
        for seed in (3301, 3302):
            folder = RUN / f"{arm}-{seed}"
            folder.mkdir()
            from .model import MethodPolicy
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                initial = MethodPolicy().state_dict()
            learner = session.base.grounded.base.learner
            facts, library = learner.legacy.snapshot(), learner.library.record()
            policy = session.owner.quest_policy
            policy.load_state_dict(initial)
            optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
            rng = torch.Generator().manual_seed(seed+10000)
            trace, baseline = {"arm": arm, "seed": seed, "rows": []}, 0.
            save(folder / "step-0000.pt", policy, optimizer, rng, baseline, 0, trace)
            for start, stop in ((0, 150), (150, 300)):
                baseline = train_steps(policy, optimizer, rng, train_weights, profiles, start, stop, baseline, trace)
                save(folder / f"step-{stop:04d}.pt", policy, optimizer, rng, baseline, stop, trace)
            learner.rebind(facts, library)
            session.base._refresh()
            if retained != protected_hash(session.owner) or any(not torch.equal(t, session.owner.state_dict()[k]) for k, t in parent_weights.items()):
                raise AssertionError("Method reward changed inherited weights")
            evaluation = rollout(policy, development, profiles, torch.Generator().manual_seed(33301), controller="greedy")
            row = {"arm": arm, "seed": seed, "checkpoint": (folder / "step-0300.pt").relative_to(ROOT).as_posix(),
                   "development": summary(evaluation), "model": model_hash(policy)}
            records.append(row)
            write(folder / "training.json", trace)
            print(row, flush=True)
    selected = max((r for r in records if r["arm"] == "verified"), key=lambda r: r["development"]["verified_utility"])
    selection = selected | {"sha256": sha(ROOT / selected["checkpoint"]), "contracts": contracts(),
                            "rule": "development verified utility; final data unopened", "all_development": records,
                            "parent_owner": parent["owner"], "old_tensors_unchanged": len(parent_weights),
                            "training_presentations_per_arm_seed": 300*64, "attempt_slots_per_arm_seed": 300*64*5}
    write(OUT / "selection.json", selection)


def evaluate():
    if (OUT / "final.json").exists():
        raise FileExistsError("Final evaluations are sealed; preserve them")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Frozen implementation changed after training")
    session = QuestSession(parent=read(RUN / "parent.json"), authority=RUN / "unused-assessor")
    profiles = np.asarray(read(RUN / "practice-screen.json")["profiles"])
    rows = []
    candidates = [(r["arm"]+"-"+str(r["seed"]), r["checkpoint"], "greedy") for r in selection["all_development"]]
    initial = "runs/QP-study-001/verified-3301/step-0000.pt"
    candidates += [(name, initial, controller) for name, controller in (("initial", "greedy"), ("balanced", "balanced"),
                                                                      ("useful_random", "useful_random"), ("coverage", "coverage"))]
    for shifted, seed in ((False, 33991), (True, 33992)):
        weights = contexts(256, seed, shifted=shifted)
        tasks, truth = cases(seed+100, 256)
        omitted, _ = cases(seed+200, 64, omitted=True)
        for name, path, controller in candidates:
            state = torch.load(ROOT / path, weights_only=True, map_location="cpu")
            session.base._mutation(lambda: session.owner.quest_policy.load_state_dict(state["weights"]))
            result = rollout(session.owner.quest_policy, weights, profiles, torch.Generator().manual_seed(seed), controller=controller)
            correct, wrong, squared, abstained, routes = 0, 0, [], 0, []
            for chosen, task, answer in zip(result["choices"], tasks, truth, strict=True):
                # Admission used independent practice checks, never the final answer.
                admitted = sorted({int(i) for i in chosen if i < STOP and profiles[i].sum() > 0})
                prediction, route = deploy(admitted, task, session.base)
                routes.append(route)
                if prediction is None:
                    abstained += 1
                else:
                    error = float((prediction-answer)**2)
                    squared.append(error)
                    correct += abs(prediction-answer) <= 1e-8
                    wrong += abs(prediction-answer) > 1e-8
            omitted_returns = sum(deploy([0, 1, 2, 3], c, session.base)[0] is not None for c in omitted)
            row = {"model": name, "cohort": "shifted" if shifted else "matched", **summary(result),
                   "correct": int(correct), "wrong": int(wrong), "abstained": abstained, "n": len(tasks),
                   "returned_mse": float(np.mean(squared)) if squared else None, "omitted_returns": omitted_returns,
                   "choices_sha256": None, "prediction_routes": routes}
            from .common import digest
            row["choices_sha256"] = digest(result["choices"].tolist())
            write(RUN / f"final-{name}-{seed}.json", {"choices": result["choices"].tolist(), "summary": row})
            rows.append(row)
    learned_name = "verified-"+str(selection["seed"])
    learned, balanced = [r for r in rows if r["model"] == learned_name], [r for r in rows if r["model"] == "balanced"]
    promote = all(a["correct"] > b["correct"] and a["wrong"] <= b["wrong"] and a["cost"] < b["cost"]
                  for a, b in zip(learned, balanced, strict=True))
    write(OUT / "final.json", {"selection_sha256": sha(OUT / "selection.json"), "contracts": contracts(),
                              "rows": rows, "default_control": "learned" if promote else "balanced",
                              "selected": learned_name, "sealed": True, "final_reward_updates": 0,
                              "qualification_scope": "supplied conditional mechanics alternatives"})
    print([(r["model"], r["cohort"], r["coverage"], r["correct"], r["cost"]) for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "final"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    train() if args.action == "train" else evaluate()


if __name__ == "__main__":
    main()
