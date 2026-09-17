"""Successive real-language teaching on the owner that retained source-learned math."""
import argparse
import json
import random
from pathlib import Path

import torch

from experiments.stream_curriculum.data import PREPARED, sha
from experiments.stream_curriculum.model import apply, delta
from experiments.stream_curriculum.study import measure, update
from sera.session_state import model_identity
from .runtime import OUT, StudySession, fingerprint
from .sources import ROOT

DATA = ROOT / "runs/SS-language-data"
LOCALES = ("es-ES", "fr-FR", "de-DE")
MIGRATABLE_TRAINERS = {"e302a3bdf9d31e43c811533cf0ff84a9c595e00a41a5910d8fbed6423d69430d"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def contracts():
    return {"runtime": fingerprint(), "trainer": sha(Path(__file__)), "data": sha(DATA / "manifest.json"),
            "protocol": sha(OUT / "protocol.md"), "parent": sha(OUT / "parent-request.json")}


def english_probe():
    data = rows(PREPARED / "dev.jsonl")
    return random.Random(2710).sample(data, 160)


def meters(owner, vocabulary, development):
    result = {locale: measure(owner, data, vocabulary) for locale, data in development.items()}
    return {name: {k: v for k, v in value.items() if k != "records"} for name, value in result.items()}


def train(output, study, seed, replay, resume=None):
    output.mkdir(parents=True, exist_ok=False)
    parent = read(OUT / "parent-request.json")
    saved = read(study)["snapshot"]
    session = StudySession(parent, saved)
    owner = session.owner
    vocabulary = owner.stream_config["vocabulary"]
    facts, library = session.learner.legacy.snapshot(), session.learner.library.record()
    protected = {n: v.clone() for n, v in owner.state_dict().items() if not n.startswith("stream_")}
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam([p for n, p in owner.named_parameters() if n.startswith("stream_")], lr=.001)
    teaching = {locale: rows(DATA / f"{locale}-train.jsonl") for locale in LOCALES}
    development = {locale: rows(DATA / f"{locale}-development.jsonl") for locale in LOCALES}
    development["en-US"] = english_probe()
    memory_rng, current_rng = random.Random(seed + 1), random.Random(seed + 2)
    memory = memory_rng.sample(rows(PREPARED / "train.jsonl"), 256)
    baseline = meters(owner, vocabulary, development)
    history, seen = [], set()
    restored = None
    if resume is not None:
        restored = torch.load(resume, map_location="cpu", weights_only=True)
        restored_contracts = dict(restored["contracts"])
        if restored_contracts["trainer"] in MIGRATABLE_TRAINERS:
            restored_contracts["trainer"] = contracts()["trainer"]
        if (restored_contracts != contracts() or restored["seed"] != seed or restored["replay"] != replay
                or restored["math_snapshot_sha256"] != sha(study)):
            raise ValueError("Resume requires the exact source, parent, seed and arm")
        apply(owner, restored["delta"])
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["torch_rng"])
        current_rng.setstate(restored["current_rng"])
        memory_rng.setstate(restored["memory_rng"])
        memory, seen = restored["memory"], set(restored["seen_ids"])
        history = restored["history"]
        baseline = restored["baseline"]
    write(output / "freeze.json", {"contracts": contracts(), "seed": seed, "replay": replay,
                                   "math_snapshot": {"path": str(study), "sha256": sha(study)},
                                   "baseline": baseline, "stages": list(LOCALES), "updates_per_stage": 400})
    count = 0 if restored is None else restored["step"]
    start = 0 if restored is None else restored["stage"]
    for stage in range(start, len(LOCALES)):
        locale = LOCALES[stage]
        data = teaching[locale]
        if restored is not None and stage == start:
            order, cursor, stage_step = restored["order"], restored["cursor"], restored["stage_step"]
        else:
            order, cursor, stage_step = list(range(len(data))), 0, 0
            current_rng.shuffle(order)
        for step in range(stage_step, 400):
            indices = []
            for _ in range(16):
                if cursor == len(order):
                    current_rng.shuffle(order)
                    cursor = 0
                indices.append(order[cursor])
                cursor += 1
            fresh = [data[i] for i in indices]
            extra = memory_rng.choices(memory, k=16)
            batch = fresh + (extra if replay else fresh)
            seen.update(row["id"] for row in fresh)
            loss = update(owner, optimizer, batch, vocabulary)
            count += 1
            if (step + 1) % 100 == 0:
                checkpoint = {"schema": "sera.multilingual.1", "contracts": contracts(),
                              "seed": seed, "replay": replay, "step": count, "stage": stage,
                              "stage_step": step + 1, "delta": delta(owner), "optimizer": optimizer.state_dict(),
                              "torch_rng": torch.get_rng_state(), "memory_rng": memory_rng.getstate(),
                              "current_rng": current_rng.getstate(), "order": order, "cursor": cursor,
                              "memory": memory, "seen_ids": sorted(seen), "loss": loss,
                              "math_snapshot_sha256": sha(study), "history": history, "baseline": baseline}
                name = output / f"step-{count:06d}.pt"
                torch.save(checkpoint, name)
                write(output / "resume.json", {"path": str(name.relative_to(ROOT)), "sha256": sha(name),
                                              "contracts": contracts(), "next_step": count + 1})
        scores = meters(owner, vocabulary, development)
        history.append({"stage": locale, "step": count, "development": scores, "unique_new_language_ids": len(seen)})
        write(output / "history.json", history)
        print(json.dumps({"run": output.name, "stage": locale,
                          "intent": {k: v["intent_accuracy"] for k, v in scores.items()}}), flush=True)
        memory.extend(memory_rng.sample(data, 256))
    assert all(torch.equal(v, owner.state_dict()[name]) for name, v in protected.items())
    session.learner.rebind(facts, library)
    final = output / f"step-{count:06d}.pt"
    if not final.exists() and restored is not None and count == restored["step"]:
        final = resume
    scores = history[-1]["development"]
    admitted = (replay and scores["en-US"]["intent_accuracy"] >= baseline["en-US"]["intent_accuracy"] - .1
                and min(scores[locale]["intent_accuracy"] for locale in LOCALES) >= .35)
    write(output / "result.json", {"checkpoint": {"path": final.relative_to(ROOT).as_posix(), "sha256": sha(final)},
                                   "source": contracts(), "baseline": baseline, "development": scores,
                                   "history": history, "protected_tensors": len(protected),
                                   "original_sum": session.propose("sum", [0, 2, 0, 3, 0, 1]),
                                   "motion": session.motion([2, 3, 1], "3", "5", "-1"),
                                   "old_non_stream_tensors_bit_exact": True, "owner": model_identity(owner),
                                   "admitted": bool(admitted), "seed": seed, "replay": replay,
                                   "new_unique_ids": len(seen), "updates": count,
                                   "presentations": count * 32, "new_stream_presentations": count * 16})


def final(output, studies, study):
    output.mkdir(parents=True, exist_ok=False)
    records = [(p, read(p / "result.json")) for p in studies]
    for _, record in records:
        if record["source"] != contracts():
            raise ValueError("Training contract changed before final assessment")
    eligible = [(p, r) for p, r in records if r["admitted"]]
    selected = sorted(eligible, key=lambda item: (-sum(item[1]["development"][l]["intent_accuracy"] for l in LOCALES), str(item[0])))
    selection = {"selected": str(selected[0][0]) if selected else None,
                 "rule": "predeclared development gate, then mean new-language intent", "final_accessed": False}
    write(output / "selection.json", selection)
    # Only now read the separately frozen language-final rows.
    final_rows = {l: rows(DATA / f"{l}-final.jsonl") for l in LOCALES}
    summary = {}
    original = read(study)["snapshot"]
    for folder, record in records:
        s = StudySession(read(OUT / "parent-request.json"), original, request_update=record["checkpoint"])
        result = {l: measure(s.owner, data, s.owner.stream_config["vocabulary"]) for l, data in final_rows.items()}
        write(output / f"{folder.name}.json", result)
        summary[folder.name] = {l: {k: v for k, v in r.items() if k != "records"} for l, r in result.items()}
        if selected and folder == selected[0][0]:
            write(output / "selected-session.json", s.snapshot())
    selection["final_accessed"] = True
    write(output / "selection.json", selection)
    write(output / "summary.json", summary)
    print(json.dumps({"selected": selection["selected"], "results": summary}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("train", "final"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2711)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--runs", type=Path, nargs="+")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.output, args.study = args.output.resolve(), args.study.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if args.runs is not None:
        args.runs = [path.resolve() for path in args.runs]
    torch.set_num_threads(1)
    if args.mode == "train":
        train(args.output, args.study, args.seed, args.replay, args.resume)
    else:
        final(args.output, args.runs, args.study)


if __name__ == "__main__":
    main()
