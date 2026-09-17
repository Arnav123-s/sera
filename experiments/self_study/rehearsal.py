"""Exact optimizer continuation with a prospectively larger rehearsal resource."""
import argparse
import random
from pathlib import Path

import torch

from experiments.stream_curriculum.data import PREPARED, sha
from experiments.stream_curriculum.model import apply, delta
from experiments.stream_curriculum.study import update
from sera.session_state import model_identity
from .multilingual import DATA, LOCALES, contracts, english_probe, meters, read, rows, write
from .runtime import OUT, StudySession
from .sources import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = read(args.parent)
    if parent["source"] != contracts():
        raise ValueError("Changed parent training contract")
    session = StudySession(read(OUT / "parent-request.json"), read(args.study)["snapshot"],
                           request_update=parent["checkpoint"])
    owner = session.owner
    protected = {n: v.clone() for n, v in owner.state_dict().items() if not n.startswith("stream_")}
    facts, library = session.learner.legacy.snapshot(), session.learner.library.record()
    saved = torch.load(ROOT / parent["checkpoint"]["path"], map_location="cpu", weights_only=True)
    optimizer = torch.optim.Adam([p for n, p in owner.named_parameters() if n.startswith("stream_")], lr=.001)
    optimizer.load_state_dict(saved["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = .0005
    rng = random.Random(2721)
    torch.set_rng_state(saved["torch_rng"])
    extension = {"source": sha(Path(__file__)), "protocol": sha(OUT / "retention-repair-protocol.md"),
                 "parent": parent["checkpoint"], "updates": 800, "seed": 2721}
    teaching = {l: rows(DATA / f"{l}-train.jsonl") for l in LOCALES}
    english = rows(PREPARED / "train.jsonl")
    development = {l: rows(DATA / f"{l}-development.jsonl") for l in LOCALES}
    development["en-US"] = english_probe()
    vocabulary = owner.stream_config["vocabulary"]
    before = meters(owner, vocabulary, development)
    start, seen = 0, set()
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume["extension"] != extension or resume["contracts"] != contracts():
            raise ValueError("Changed continuation contract")
        apply(owner, resume["delta"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["torch_rng"])
        rng.setstate(resume["sample_rng"])
        start, seen = resume["extension_step"], set(resume["seen_ids"])
    write(output / "freeze.json", {"extension": extension, "contracts": contracts(), "before": before})
    for step in range(start, 800):
        batch = rng.choices(english, k=16)
        for _ in range(16):
            locale = rng.choice(LOCALES)
            batch.append(rng.choice(teaching[locale]))
        seen.update(str(row["id"]) for row in batch)
        loss = update(owner, optimizer, batch, vocabulary)
        if (step + 1) % 100 == 0:
            checkpoint = {"schema": "sera.multilingual.1", "contracts": contracts(), "extension": extension,
                          "step": 1200 + step + 1, "extension_step": step + 1, "delta": delta(owner),
                          "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                          "sample_rng": rng.getstate(), "seen_ids": sorted(seen), "loss": loss}
            path = output / f"step-{1200 + step + 1:06d}.pt"
            torch.save(checkpoint, path)
            write(output / "resume.json", {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                                           "next_step": 1200 + step + 2})
    assert all(torch.equal(v, owner.state_dict()[n]) for n, v in protected.items())
    session.learner.rebind(facts, library)
    scores = meters(owner, vocabulary, development)
    final = output / "step-002000.pt"
    if not final.exists() and start == 800:
        final = args.resume.resolve()
    admitted = (scores["en-US"]["intent_accuracy"] >= parent["baseline"]["en-US"]["intent_accuracy"] - .1
                and min(scores[l]["intent_accuracy"] for l in LOCALES) >= .35)
    result = {"checkpoint": {"path": final.relative_to(ROOT).as_posix(), "sha256": sha(final)},
              "source": contracts(), "extension": extension, "baseline": parent["baseline"],
              "before": before, "development": scores, "admitted": admitted, "replay": True, "seed": 2721,
              "protected_tensors": len(protected), "old_non_stream_tensors_bit_exact": True,
              "original_sum": session.propose("sum", [0, 2, 0, 3, 0, 1]),
              "motion": session.motion([2, 3, 1], "3", "5", "-1"), "owner": model_identity(owner),
              "additional_presentations": 800 * 32, "additional_unique_ids": len(seen),
              "additional_updates": 800, "total_lineage_updates": 2000}
    write(output / "result.json", result)
    print({"admitted": admitted, "intent": {l: v["intent_accuracy"] for l, v in scores.items()}})


if __name__ == "__main__":
    main()
