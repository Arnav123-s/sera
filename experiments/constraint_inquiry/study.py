"""Finite acquired interfaces, with exact optimizer/RNG state and preserved parents."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.language_inquiry.runtime import Runtime
from experiments.language_inquiry.study import ROOT, read, sha, write
from sera.session_state import model_identity
from workbench.storage import Store

from .data import corpus, encode
from .model import ConstraintR1, apply, delta, source

OUT = ROOT / "research-continuation/25_constraint_inquiry"


def pin():
    path = OUT / "parent.json"
    if not path.exists():
        write(path, Store(ROOT / "runs/sera-inquiry").read())
    return read(path)


def parent():
    record = pin()
    return Runtime(record["checkpoint"], record)


def restore(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["source"] != source() or saved["parent"] != sha(OUT / "parent.json"):
        raise ValueError("Changed training source or predecessor")
    base = parent()
    learner = base.session.learner
    facts, library = learner.legacy.snapshot(), learner.library.record()
    ConstraintR1.attach(base.owner, saved["seed"], saved["kind"])
    apply(base.owner, saved["delta"])
    learner.rebind(facts, library)
    return base, saved


def fit(base_owner, seed, kind, steps, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    owner = copy.deepcopy(base_owner)
    old = {n: v.clone() for n, v in owner.state_dict().items()}
    ConstraintR1.attach(owner, seed, kind)
    for name, value in owner.named_parameters():
        value.requires_grad_(name.startswith("cloud_") and not name.startswith("cloud_proposal."))
    owner.eval()
    params = [p for p in owner.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=.003)
    rows = corpus("train")
    x = torch.tensor(encode([r["text"] for r in rows]))
    y = torch.tensor([r["target"] for r in rows])
    history = []
    shared_gradient = None
    # Check the derivative through the existing fusion while keeping it protected.
    probe = next(owner.fusion.parameters())
    probe.requires_grad_(True)
    loss = sum(F.cross_entropy(p, y[:, k]) for k, p in enumerate(owner.binding(x)))
    derivative, = torch.autograd.grad(loss, probe)
    shared_gradient = float(derivative.norm())
    probe.requires_grad_(False)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(p, y[:, k]) for k, p in enumerate(owner.binding(x)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        history.append(float(loss.detach()))
        if step % 50 == 0:
            with torch.no_grad():
                outputs = owner.binding(x)
                exact = float((torch.stack([p.argmax(-1) for p in outputs], 1) == y).all(-1).float().mean())
                score = float(sum(F.cross_entropy(p, y[:, k]) for k, p in enumerate(outputs)))
            torch.save({"source": source(), "parent": sha(OUT / "parent.json"),
                        "seed": seed, "kind": kind, "step": step, "delta": delta(owner),
                        "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                        "history": history, "teaching_exact": exact, "teaching_loss": score},
                       output / f"step-{step:04d}.pt")
    unchanged = all(torch.equal(v, owner.state_dict()[n]) for n, v in old.items())
    if not unchanged:
        raise AssertionError("A protected predecessor tensor changed")
    checkpoint = output / f"step-{steps:04d}.pt"
    result = {"seed": seed, "kind": kind, "steps": steps, "teaching_exact": exact,
              "teaching_loss": score, "teaching_strings": len(rows), "presentations": steps * len(rows),
              "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "sha256": sha(checkpoint),
              "old_tensors_unchanged": len(old), "shared_path_gradient_norm": shared_gradient,
              "new_parameters": sum(v.numel() for n, v in owner.named_parameters() if n.startswith("cloud_")),
              "seconds": time.perf_counter() - started}
    write(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def train_proposal(owner, output):
    started = time.perf_counter()
    torch.manual_seed(25100)
    generator = torch.Generator().manual_seed(25100)
    n = 4096
    velocity = torch.rand(n, generator=generator) * .5 - .25
    mean = owner.interaction_mean.float()
    abc = mean.expand(n, -1).clone()
    # The parent supplies the conditional teacher; no observed-fact admission occurs.
    controls = torch.rand((n, 2), generator=generator) * 1.6 - .8
    a, gain, bias = abc.unbind(-1)
    offset = (a + a * a) * velocity + (2 + a) * bias + gain * ((1 + a) * controls[:, 0] + controls[:, 1])
    features = torch.cat((velocity[:, None], abc, offset[:, None]), 1)
    vector = torch.stack((gain * (1 + a), gain), 1)
    residual = offset - (a + a * a) * velocity - (2 + a) * bias
    target = (vector * (residual / vector.square().sum(-1))[:, None]).clamp(-1, 1)
    opt = torch.optim.Adam(owner.cloud_proposal.parameters(), lr=.003)
    history = []
    for step in range(1, 401):
        idx = torch.randint(n, (128,), generator=generator)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(owner.cloud_proposal(features[idx]), target[idx])
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if step % 100 == 0:
            torch.save({"state": owner.cloud_proposal.state_dict(), "optimizer": opt.state_dict(),
                        "step": step, "rng": generator.get_state(), "history": history,
                        "features": features, "targets": target, "provenance": "parent_conditional_simulation"},
                       output / f"proposal-{step:04d}.pt")
    return {"steps": 400, "examples": n, "presentations": 51200, "last_loss": history[-1],
            "seconds": time.perf_counter() - started,
            "source": "Supplied inverse control teacher applied to the acquired parent dynamics"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    pin()
    write(args.output / "freeze.json", {"source": source(), "protocol": sha(OUT / "protocol.md"),
                                        "parent": sha(OUT / "parent.json")})
    base = parent()
    results = []
    seeds = [2499] if args.pilot else [2501, 2502]
    for seed in seeds:
        for kind, steps in ([("ordered", 200)] if args.pilot else [("ordered", 200), ("bag", 200), ("ordered", 400)]):
            results.append(fit(base.owner, seed, kind, steps, args.output / f"{kind}-{steps}-{seed}"))
    if not args.pilot:
        selected = min((r for r in results if r["kind"] == "ordered" and r["steps"] == 200),
                       key=lambda r: r["teaching_loss"])
        if selected["teaching_exact"] != 1.:
            raise ValueError("No exact primary teaching fit; preserve failed acquisition")
        runtime, saved = restore(ROOT / selected["checkpoint"])
        proposal = train_proposal(runtime.owner, args.output)
        saved["delta"] = delta(runtime.owner)
        saved["proposal"] = proposal
        torch.save(saved, args.output / "selected.pt")
        write(args.output / "selected.json", {"checkpoint": (args.output / "selected.pt").relative_to(ROOT).as_posix(),
                                              "sha256": sha(args.output / "selected.pt"), "language": selected,
                                              "proposal": proposal, "owner": model_identity(runtime.owner)})
    write(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
