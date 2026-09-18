"""Frozen conditional reward study, strong controls and sealed final evaluation."""

import argparse
import copy

import numpy as np
import torch

from .common import OUT, ROOT, RUN, contracts, digest, model_hash, read, sha, write
from .runtime import CompletionSession
from .task import STOP, analytic, consequences, features, make_bank


def subset(bank, indices):
    return {k: (v[indices] if v.ndim and len(v) == len(bank["mean"]) else v) for k, v in bank.items()}


def summary(result):
    return {"cases": len(result["actions"]), "reward": float(result["reward"].mean()),
            "mse": float(result["squared_error"].mean()), "coverage": float(result["covered"].mean()),
            "mean_cost": float(result["cost"].mean()), "stop_rate": float((result["actions"] == STOP).mean())}


def evaluate(policy, bank):
    with torch.no_grad():
        action = policy(torch.tensor(features(bank))).argmax(-1).numpy()
    return consequences(bank, action)


def prepare():
    path = OUT / "prior.json"
    if path.exists():
        raise FileExistsError("Preserve prior source identity")
    source = ROOT / "research/intake/v14-update/SERA_v14/results/data/prior.npz"
    with np.load(source, allow_pickle=False) as data:
        prior = {"mean": data["mean"].tolist(), "covariance": data["covariance"].tolist(),
                 "source_sha256": sha(source), "source": "SERA_v14/results/data/prior.npz",
                 "origin": "48 numerical mechanisms, 1152 synthetic observations; packet prior reused unchanged"}
    write(path, prior | {"identity": digest(prior)})


def train():
    if (OUT / "selection.json").exists() or RUN.exists():
        raise FileExistsError("Preserve training; resume explicitly from saved state")
    RUN.mkdir(parents=True)
    train_bank, development = make_bank(4096, 32001), make_bank(512, 32002)
    np.savez_compressed(RUN / "train.npz", **train_bank)
    np.savez_compressed(RUN / "development.npz", **development)
    options = []
    for seed in (3211, 3212):
        session = CompletionSession(load_selected=False, seed=seed)
        policy = session.owner.completion_policy
        old = {k: v.clone() for k, v in session.owner.state_dict().items() if not k.startswith("completion_")}
        f = torch.tensor(features(train_bank))
        policy.center.copy_(f.flatten(0, 1).mean(0))
        policy.scale.copy_(f.flatten(0, 1).std(0, unbiased=False).clamp_min(.05))
        optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
        rng = torch.Generator().manual_seed(seed)
        folder = RUN / str(seed)
        folder.mkdir()
        initial = {"weights": copy.deepcopy(policy.state_dict()), "contracts": contracts()}
        torch.save(initial, folder / "initial.pt")
        baseline = 0.
        traces = []
        for step in range(1, 801):
            indices = torch.randint(len(f), (64,), generator=rng)
            logits = policy(f[indices])
            probabilities = logits.softmax(-1)
            actions = torch.multinomial(probabilities, 1, generator=rng).squeeze(-1)
            # Only the sampled action's independent reward is supplied to the learner.
            result = consequences(subset(train_bank, indices.numpy()), actions.numpy())
            rewards = torch.tensor(result["reward"])
            advantage = rewards-baseline
            loss = -(advantage*logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(-1)).mean()
            entropy = -(probabilities*logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss-.01*entropy).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            traces.append({"step": step, "indices": indices.tolist(), "actions": actions.tolist(),
                           "rewards": rewards.tolist(), "baseline": baseline})
            baseline = .98*baseline+.02*float(rewards.mean())
            if step in (400, 800):
                torch.save({"weights": copy.deepcopy(policy.state_dict()), "optimizer": optimizer.state_dict(),
                            "rng": rng.get_state(), "step": step, "baseline": baseline, "contracts": contracts(),
                            "seed": seed, "training": sha(RUN / "train.npz"), "policy": model_hash(policy)},
                           folder / f"step-{step:04d}.pt")
        assert all(torch.equal(session.owner.state_dict()[k], v) for k, v in old.items())
        candidate = {"seed": seed, "checkpoint": (folder / "step-0800.pt").relative_to(ROOT).as_posix(),
                     "sha256": sha(folder / "step-0800.pt"), "development": summary(evaluate(policy, development)),
                     "predecessor_tensors_retained": len(old)}
        write(folder / "trace.json", traces)
        options.append(candidate)
    chosen = max(options, key=lambda x: (x["development"]["reward"], -x["seed"]))
    write(OUT / "development.json", {"candidates": options, "cases": 512,
                                    "conditional_reward_presentations": 2*800*64,
                                    "unique_training_worlds": 4096, "human_foundation_updates": 0})
    write(OUT / "selection.json", chosen | {"contracts": contracts(), "criterion": "development independent mean reward"})
    print(chosen)


def final():
    if (OUT / "final.json").exists():
        raise FileExistsError("Final evaluation remains sealed")
    selection = read(OUT / "selection.json")
    session = CompletionSession()
    trained = session.owner.completion_policy
    initial = copy.deepcopy(trained)
    initial.load_state_dict(torch.load(RUN / str(selection["seed"]) / "initial.pt", weights_only=True)["weights"])
    rows, paired = {}, {}
    for family, seed in (("matched", 32991), ("shifted", 32992), ("omitted", 32993)):
        bank = make_bank(256, seed, family)
        np.savez_compressed(RUN / f"final-{family}.npz", **bank)
        records = {"learned": evaluate(trained, bank), "initial": evaluate(initial, bank)}
        for name, actions in (("random", np.random.default_rng(32994).integers(0, 9, len(bank["mean"]))),
                              ("stop", np.full(len(bank["mean"]), STOP)),
                              ("goal_information", analytic(bank)), ("parameter_information", analytic(bank, "parameter"))):
            records[name] = consequences(bank, actions)
        for name, result in records.items():
            np.savez_compressed(RUN / f"{family}-{name}.npz", **result)
        rows[family] = {name: summary(result) for name, result in records.items()}
        paired[family] = {name: {"mean_reward_gain": float(np.mean(records["learned"]["reward"]-result["reward"])),
                                 "mean_mse_reduction": float(np.mean(result["squared_error"]-records["learned"]["squared_error"]))}
                          for name, result in records.items() if name != "learned"}
    passed = all(paired["matched"][name]["mean_reward_gain"] > 0 and paired["matched"][name]["mean_mse_reduction"] > 0
                 for name in ("initial", "random"))
    result = {"selection_sha256": sha(OUT / "selection.json"), "results": rows, "paired": paired,
              "independent_worlds": 768, "crossed_evaluations": 768*6,
              "conditional_learned_use_qualified": passed,
              "default_control": "learned" if passed and paired["matched"]["goal_information"]["mean_reward_gain"] > 0 else "goal_information",
              "factual_training_updates": 0}
    write(OUT / "final.json", result)
    print(result)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "final"))
    action = parser.parse_args().action
    {"prepare": prepare, "train": train, "final": final}[action]()


if __name__ == "__main__":
    main()
