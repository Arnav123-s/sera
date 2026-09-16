"""Finite supervised acquisition, exact checkpoints and protected-parent audit."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.task_transfer.runtime import Session as TaskSession
from sera.session_state import model_identity

from .compatibility import TRAINING_SOURCES
from .data import TEACHING, encode
from .model import InquiryR1, apply, delta, source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/24_language_inquiry"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def pin():
    path = OUT / "parent.json"
    if path.exists():
        return read(path)
    directory = ROOT / "runs/sera-task-transfer"
    ref = read(directory/"current.json")
    revision = directory/"revisions"/ref["revision"]
    if sha(revision) != ref["sha256"] or sha(directory/"parent.json") != ref["parent_sha256"]:
        raise ValueError("Task parent changed during reconciliation")
    bundle = {"workbench": read(directory/"parent.json"), "tasks": read(revision), "reference": ref}
    write(path, bundle)
    return bundle


def base(bundle=None):
    bundle = pin() if bundle is None else bundle
    return TaskSession(copy.deepcopy(bundle["workbench"]), copy.deepcopy(bundle["tasks"]))


def restore_fit(path, bundle=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["source"] != source() and saved["source"] not in TRAINING_SOURCES) or saved["parent_bundle"] != sha(OUT/"parent.json"):
        raise ValueError("Fit source or parent differs")
    session = base(bundle)
    factual, library = session.learner.legacy.snapshot(), session.learner.library.record()
    InquiryR1.attach(session.owner, saved["seed"], saved["kind"])
    session.owner.inquiry_source = saved["source"]
    apply(session.owner, saved["delta"])
    session.learner.rebind(factual, library)
    return session, saved


def fit(seed, kind, output, steps=100):
    output = Path(output).resolve()
    if (output/"summary.json").exists():
        return read(output/"summary.json")
    if (output/f"step-{steps:04d}.pt").exists():
        final = output/f"step-{steps:04d}.pt"
        recovered, saved = restore_fit(final)
        summary = {"seed": seed, "kind": kind, "steps": steps, "training_spans": len(TEACHING),
                   "training_presentations": steps*len(TEACHING), "training_exact": saved["training_exact"],
                   "training_loss": saved["training_loss"], "checkpoint": final.relative_to(ROOT).as_posix(),
                   "sha256": sha(final), "additional_parameters": sum(t.numel() for t in delta(recovered.owner).values()),
                   "same_owner": True, "seconds": None,
                   "recovery": "Completed checkpoint recovered after relative-path reporting defect; no optimizer step repeated. Full failed invocation cost remains in the supervisor ledger."}
        write(output/"summary.json", summary)
        return summary
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    session = base()
    owner = session.owner
    before = {n: t.clone() for n, t in owner.state_dict().items()}
    parent_id = model_identity(owner)
    InquiryR1.attach(owner, seed, kind)
    for n, p in owner.named_parameters():
        p.requires_grad_(n.startswith("inquiry_"))
    parameters = [p for p in owner.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.003)
    ids, target = torch.tensor(encode([r[0] for r in TEACHING])), torch.tensor([r[1] for r in TEACHING])
    history = []
    for step in range(steps+1):
        if step:
            owner.train()
            logits = owner.span_logits(ids)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            history.append({"step": step, "loss": float(loss.detach())})
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                owner.eval()
                logits = owner.span_logits(ids)
                exact = float((logits.argmax(-1) == target).float().mean())
                loss = float(F.cross_entropy(logits, target))
            saved = {"source": source(), "parent_bundle": sha(OUT/"parent.json"),
                     "parent_owner": parent_id, "seed": seed, "kind": kind, "step": step,
                     "delta": delta(owner), "optimizer": optimizer.state_dict(),
                     "torch_rng": torch.get_rng_state(), "history": history,
                     "training_exact": exact, "training_loss": loss}
            torch.save(saved, output/f"step-{step:04d}.pt")
    changed = [n for n, t in before.items() if not torch.equal(t, owner.state_dict()[n])]
    if changed:
        raise AssertionError("Protected parent changed: "+str(changed))
    if not (session.learner.solver.neural.owner is session.learner.solver.components["typed"].owner is owner):
        raise AssertionError("Shared owner aliases diverged")
    final = output/f"step-{steps:04d}.pt"
    summary = {"seed": seed, "kind": kind, "steps": steps, "training_spans": len(TEACHING),
               "training_presentations": steps*len(TEACHING), "training_exact": exact, "training_loss": loss,
               "checkpoint": str(final.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(final),
               "additional_parameters": sum(t.numel() for t in delta(owner).values()),
               "old_tensors_unchanged": len(before), "same_owner": True,
               "seconds": time.perf_counter()-started}
    write(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    pin()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    frozen = {"source": source(), "protocol_sha256": sha(OUT/"protocol.md"),
              "parent_bundle_sha256": sha(OUT/"parent.json"), "seeds": [2399] if a.pilot else [2401, 2402, 2403]}
    if a.resume:
        if read(a.output/"frozen-inputs.json") != frozen:
            raise ValueError("Continuation inputs differ")
    else:
        write(a.output/"frozen-inputs.json", frozen)
    records = []
    for seed in ([2399] if a.pilot else [2401, 2402, 2403]):
        for kind in ("shared", "bag"):
            records.append(fit(seed, kind, a.output/f"{kind}-{seed}"))
    write(a.output/"summary.json", records)
    eligible = [r for r in records if r["kind"] == "shared" and r["training_exact"] == 1.]
    if eligible:
        write(a.output/"selected.json", min(eligible, key=lambda r: (r["training_loss"], r["seed"])))


if __name__ == "__main__":
    main()
