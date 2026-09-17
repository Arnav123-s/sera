"""Finite training and once-only final evaluation, charged by the bounded runner."""

import argparse
import json
import time

import numpy as np
import torch

from . import data, physical
from .common import EXPERIMENT as OUT
from .common import ROOT, RUNS, contracts, read, restore_parent, sha, write
from .model import KINDS, RefiningR1, apply, delta


def prepare():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("Protocol already frozen; do not replace it")
    RUNS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "dev"):
        path = RUNS / f"{split}.json"
        if path.exists():
            raise FileExistsError(path)
        write(path, data.bank(split))
        paths[split] = sha(path)
    write(
        OUT / "freeze.json",
        {
            "contracts": contracts(),
            "data": paths,
            "seeds": [2801, 2802],
            "steps": 240,
            "final_opened": False,
        },
    )
    print(json.dumps({"prepared": paths, "final_opened": False}))


def train(kind, seed, pilot=False, resume=None):
    freeze = read(OUT / "freeze.json")
    if freeze["contracts"] != contracts():
        raise ValueError("Frozen model/data/protocol changed")
    folder = RUNS / (f"pilot-{kind}-{seed}" if pilot else f"{kind}-{seed}")
    if resume is None:
        folder.mkdir(exist_ok=False)
    session = restore_parent()
    owner = session.owner
    protected = {n: v.clone() for n, v in owner.state_dict().items()}
    RefiningR1.attach(owner, kind, seed)
    if kind == "r1_adapted" and len(owner.concept_parameter_names) < 5:
        raise AssertionError("Adaptation must reach the actual delta-memory projections")
    optimizer = torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=0.001)
    rng = np.random.default_rng(seed)
    x, y = (torch.from_numpy(a) for a in data.training_arrays(read(RUNS / "train.json")))
    dev = [
        r for r in read(RUNS / "dev.json") if r["condition"] == "clean" and r["family"] != "omitted"
    ]
    dx, dy = (torch.from_numpy(a) for a in data.training_arrays(dev))
    start, logs, best, best_delta, first_step = time.perf_counter(), [], 1e30, None, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            checkpoint["contracts"] != contracts()
            or checkpoint["kind"] != kind
            or checkpoint["seed"] != seed
        ):
            raise ValueError("Changed resume identity")
        apply(owner, checkpoint["delta"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        first_step, logs, best, best_delta = (
            checkpoint["step"],
            checkpoint["logs"],
            checkpoint["best"],
            checkpoint["best_delta"],
        )
    owner.eval()
    with torch.no_grad():
        before = float(((owner.empirical(dx)[:, 12:] - dy[:, 12:]) ** 2).mean())
    count = 8 if pilot else 240
    for step in range(first_step + 1, count + 1):
        owner.train()
        indices = rng.integers(0, len(x), 8)
        offset = int(rng.integers(0, x.shape[1] - 40 + 1))
        bx, by = x[indices, offset : offset + 40], y[indices, offset : offset + 40]
        optimizer.zero_grad(set_to_none=True)
        loss = ((owner.empirical(bx)[:, 12:] - by[:, 12:]) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; preserve checkpoint")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in owner.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 40 == 0 or step == count:
            owner.eval()
            with torch.no_grad():
                value = float(((owner.empirical(dx)[:, 12:] - dy[:, 12:]) ** 2).mean())
            if value < best:
                best, best_delta = value, delta(owner)
            logs.append(
                dict(
                    step=step,
                    train_loss=float(loss.detach()),
                    dev_mse=value,
                    elapsed_seconds=time.perf_counter() - start,
                )
            )
            checkpoint = dict(
                schema="sera.concept.train.1",
                contracts=contracts(),
                kind=kind,
                seed=seed,
                step=step,
                delta=delta(owner),
                optimizer=optimizer.state_dict(),
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                logs=logs,
                best=best,
                best_delta=best_delta,
            )
            torch.save(checkpoint, folder / f"step-{step:04d}.pt")
            write(
                folder / "progress.json",
                {k: checkpoint[k] for k in ("kind", "seed", "step", "logs", "best")},
            )
            print(json.dumps(logs[-1]), flush=True)
            if time.perf_counter() - start > 440 and step < count:
                print("Exact checkpoint saved; resume the same finite run", flush=True)
                return
    if not all(torch.equal(value, owner.state_dict()[name]) for name, value in protected.items()):
        raise AssertionError("A protected predecessor tensor changed")
    torch.save(
        dict(
            schema="sera.concept.selected.1",
            contracts=contracts(),
            kind=kind,
            seed=seed,
            delta=best_delta,
            best_development_mse=best,
        ),
        folder / "selected.pt",
    )
    write(
        folder / "fit.json",
        dict(
            kind=kind,
            seed=seed,
            updates=count,
            before_dev_mse=before,
            best_dev_mse=best,
            logs=logs,
            protected_tensors=len(protected),
            trainable_parameters=sum(p.numel() for p in owner.parameters() if p.requires_grad),
            conditional_memory_parameters=owner.concept_parameter_names,
            wall_seconds=time.perf_counter() - start,
            checkpoint_sha256=sha(folder / "selected.pt"),
        ),
    )


@torch.no_grad()
def neural_forecasts(owner, rows):
    values = np.array([r["values"] for r in rows])
    masks = np.array([r["masks"] for r in rows])
    controls = np.array([r["controls"] for r in rows])
    future = np.array([r["future_controls"] for r in rows])
    prefix = np.stack(
        [data.features(v[:-1], u, m[:-1]) for v, u, m in zip(values, controls, masks)]
    )
    outputs = []
    present = values[:, -1].copy()
    for t in range(future.shape[1]):
        feature = np.column_stack(
            (present[:, 0] / 10, present[:, 1], future[:, t], np.ones((len(rows), 3)))
        )
        prefix = np.concatenate((prefix, feature[:, None].astype("float32")), axis=1)
        a = owner.empirical(torch.from_numpy(prefix))[:, -1].numpy()
        v = present[:, 1] + data.DT * a
        present = np.column_stack((present[:, 0] + data.DT * v, v))
        outputs.append(present.copy())
    return np.stack(outputs, axis=1)


def metrics(rows, predictions):
    truth = np.array([r["truth"] for r in rows])
    error = (predictions - truth) ** 2
    result = {}
    for family in data.FAMILIES:
        for condition in data.CONDITIONS:
            selected = [
                i
                for i, r in enumerate(rows)
                if r["family"] == family and r["condition"] == condition
            ]
            score = {}
            for horizon in (1, 4, 12):
                e = error[selected, horizon - 1]
                score[str(horizon)] = dict(
                    position_rmse=float(np.sqrt(e[:, 0].mean())),
                    velocity_rmse=float(np.sqrt(e[:, 1].mean())),
                    mse=float(e.mean()),
                )
            pairs = []
            by_world = {}
            for i in selected:
                by_world.setdefault(rows[i]["world"], {})[rows[i]["history"]] = i
            for pair in by_world.values():
                a, b = pair[0], pair[1]
                real, imagined = (
                    truth[a, -1, 1] - truth[b, -1, 1],
                    predictions[a, -1, 1] - predictions[b, -1, 1],
                )
                pairs.append(
                    dict(
                        truth=float(real),
                        prediction=float(imagined),
                        eligible=bool(abs(real) > 0.002),
                        sign_correct=bool(real * imagined > 0),
                    )
                )
            eligible = [p for p in pairs if p["eligible"]]
            score["pairs"] = pairs
            score["pair_sign_accuracy"] = (
                sum(p["sign_correct"] for p in eligible) / len(eligible) if eligible else None
            )
            result[f"{family}/{condition}"] = score
    return result


def evaluate(split, arm):
    if read(OUT / "freeze.json")["contracts"] != contracts():
        raise ValueError("Frozen contracts changed")
    destination = RUNS / f"evaluation-{split}-{arm}.json"
    if destination.exists():
        raise FileExistsError("Completed evaluation is sealed")
    if split == "final":
        selection = read(OUT / "selection.json")
        if selection["contracts"] != contracts():
            raise ValueError("Final selection contract changed")
        if not (RUNS / "final.json").exists():
            write(RUNS / "final.json", data.bank("final"))
            write(
                OUT / "final-open.json",
                {
                    "selection_sha": sha(OUT / "selection.json"),
                    "data_sha": sha(RUNS / "final.json"),
                    "contracts": contracts(),
                },
            )
    rows = read(RUNS / f"{split}.json")
    start, models = time.perf_counter(), []
    if arm in ("physical", "instant", "coarse", "memory"):
        predictions = []
        for row in rows:
            model = physical.fit(
                row["values"],
                row["masks"],
                row["controls"],
                force=None if arm == "physical" else arm,
            )
            predicted = physical.predict(model, row["values"], row["future_controls"])
            independent = physical.predict(
                model, row["values"], row["future_controls"], independent=True
            )
            np.testing.assert_allclose(predicted, independent, atol=2e-12, rtol=0)
            predictions.append(predicted)
            models.append(model)
        predictions = np.array(predictions)
    else:
        folder = RUNS / arm
        checkpoint = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
        if checkpoint["contracts"] != contracts():
            raise ValueError("Candidate contract changed")
        owner = restore_parent().owner
        RefiningR1.attach(owner, checkpoint["kind"], checkpoint["seed"])
        apply(owner, checkpoint["delta"])
        owner.eval()
        predictions = np.concatenate(
            [neural_forecasts(owner, rows[i : i + 24]) for i in range(0, len(rows), 24)]
        )
    summary = metrics(rows, predictions)
    write(
        destination,
        dict(
            arm=arm,
            split=split,
            contracts=contracts(),
            data_sha=sha(RUNS / f"{split}.json"),
            row_ids=[r["id"] for r in rows],
            predictions=predictions.tolist(),
            models=models,
            metrics=summary,
            wall_seconds=time.perf_counter() - start,
        ),
    )
    write(
        OUT / f"{split}-{arm}.json",
        {
            "arm": arm,
            "split": split,
            "metrics": summary,
            "raw_path": str(destination.relative_to(ROOT)),
            "raw_sha": sha(destination),
            "wall_seconds": time.perf_counter() - start,
        },
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "split": split,
                "seconds": time.perf_counter() - start,
                "delayed_clean": summary["delayed/clean"]["12"],
                "simple_clean": summary["simple/clean"]["12"],
            }
        )
    )


def gate(candidate, baseline):
    c, b = candidate["metrics"], baseline["metrics"]
    checks = {
        "delayed_improvement": c["delayed/clean"]["12"]["mse"]
        <= 0.8 * b["delayed/clean"]["12"]["mse"],
        "history_sign": (c["delayed/clean"]["pair_sign_accuracy"] or 0) >= 0.75,
        "simple_retention": c["simple/clean"]["12"]["mse"]
        <= b["simple/clean"]["12"]["mse"] + 0.0025,
        "noisy": c["delayed/noisy"]["12"]["mse"] < 0.25,
        "missing": c["delayed/missing"]["12"]["mse"] < 0.25,
    }
    return {"checks": checks, "admitted": all(checks.values())}


def select():
    path = OUT / "selection.json"
    if path.exists() or (RUNS / "final.json").exists():
        raise FileExistsError("Selection already frozen or final already opened")
    physical_result = read(OUT / "dev-physical.json")
    baseline = read(OUT / "dev-instant.json")
    candidates = {"physical": gate(physical_result, baseline)}
    winner = "physical" if candidates["physical"]["admitted"] else None
    score = physical_result["metrics"]["delayed/clean"]["12"]["mse"]
    for kind in KINDS:
        for seed in (2801, 2802):
            arm = f"{kind}-{seed}"
            result = read(OUT / f"dev-{arm}.json")
            candidates[arm] = gate(result, baseline)
            value = result["metrics"]["delayed/clean"]["12"]["mse"]
            if candidates[arm]["admitted"] and value < score * 0.95:
                winner, score = arm, value
    write(
        path,
        {
            "contracts": contracts(),
            "selected": winner,
            "candidates": candidates,
            "development_receipts": {p.name: sha(p) for p in sorted(OUT.glob("dev-*.json"))},
        },
    )
    print(json.dumps(read(path)))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "train", "evaluate", "select", "training", "development", "final"),
    )
    parser.add_argument("--kind", choices=KINDS, default="r1_adapted")
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    parser.add_argument("--arm", default="physical")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    if args.action == "train":
        train(args.kind, args.seed, args.pilot, args.resume)
    if args.action == "evaluate":
        evaluate(args.split, args.arm)
    if args.action == "select":
        select()
    if args.action == "training":
        for kind in KINDS:
            for seed in (2801, 2802):
                if (RUNS / f"{kind}-{seed}/fit.json").exists():
                    continue
                print(f"START {kind} {seed}", flush=True)
                train(kind, seed)
    if args.action in ("development", "final"):
        split = "dev" if args.action == "development" else "final"
        for arm in ["physical", "instant", "coarse", "memory"] + [
            f"{k}-{s}" for k in KINDS for s in (2801, 2802)
        ]:
            if (RUNS / f"evaluation-{split}-{arm}.json").exists():
                continue
            evaluate(split, arm)


if __name__ == "__main__":
    main()
