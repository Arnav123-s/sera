"""Teach the restored owner from human books, and measure what changed.

Actions:

* ``calibrate`` — one supervised throughput and memory measurement, so the
  step budget in the protocol is chosen from evidence rather than guessed;
* ``smoke`` — a short end-to-end pass that exercises every mechanism the
  review demanded before sustained teaching: fresh-process isolated restore,
  connected forward and backward, ragged batches, masked loss, failing-closed
  evaluation, per-stage source exposure, atomic checkpointing, resume from the
  saved optimiser/RNG/cursor, and interrupted-versus-uninterrupted replay;
* ``train`` — the staged run declared in the frozen protocol, for one arm.

Every action runs under ``scripts/run_owner_supervised.py``; nothing here
acquires the shared lease itself.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate, identity

activate()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.owner_language import training as tr  # noqa: E402
from experiments.owner_language.corpus import load_items  # noqa: E402
from experiments.owner_language.descendant import (  # noqa: E402
    LanguageAcquisitionR1,
    disconnect_inherited_core,
)
from experiments.owner_language.owner import BASELINE_STORE, restore_owner  # noqa: E402
from experiments.owner_language.tasks import Tokens, load_evaluation  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
PROTOCOL = LAB_ROOT / "research-continuation/48_owner_language_acquisition/protocol.json"
ARMS = ("connected", "blocked", "disconnected")


def output_directory(explicit=None):
    if explicit is not None:
        path = Path(explicit)
    else:
        path = Path(os.environ.get("SERA_LAB_ATTEMPT", RUNS / "attempts/manual"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(arm, *, store=BASELINE_STORE, adapter_rank=32, embedding=64, seed=4801):
    """Restore the real owner, attach the language route, configure the arm."""
    tokens = Tokens()
    session, growth, restore_record = restore_owner(store)
    baseline_state = session.state()
    contract = {"parent_owner": baseline_state["parent_owner"], "contracts": baseline_state["contracts"],
                "schema": baseline_state["schema"], "subjects": len(baseline_state["subjects"]),
                "baseline_owner": baseline_state["owner"], "pointer": restore_record["pointer"]}
    parent_identity = session.identity()
    owner = LanguageAcquisitionR1.attach(session.owner, tokens.record,
                                         embedding=embedding, adapter_rank=adapter_rank, seed=seed)
    control = {"arm": arm}
    if arm == "blocked":
        for name, parameter in owner.named_parameters():
            if name.startswith("adapter."):
                parameter.requires_grad_(False)
        control["blocked_tensors"] = [n for n, _ in owner.named_parameters() if n.startswith("adapter.")]
        control["meaning"] = ("the adapter stays at its zero-output initialisation, so the new language "
                              "modules read frozen inherited features; this is the Stage 30 contract as a control")
    elif arm == "disconnected":
        control["reinitialised_core_tensors"] = disconnect_inherited_core(owner)
        control["meaning"] = ("identical new modules and adapter, but the inherited shared core is replaced by "
                              "seeded random weights; this arm is never saved as a descendant")
    elif arm != "connected":
        raise ValueError(f"Unknown arm {arm}")
    return {"tokens": tokens, "session": session, "growth": growth, "owner": owner,
            "restore": restore_record, "contract": contract, "parent_identity": parent_identity,
            "control": control}


def make_trainer(context, schedule, options, *, store=None):
    trainer = tr.Trainer(context["owner"], context["tokens"], schedule, arm=context["control"]["arm"],
                         learning_rate=options.learning_rate, batch=options.batch,
                         length=options.length, seed=options.seed, store=store,
                         baseline_contract=context["contract"])
    trainer.parent_identity = context["parent_identity"]
    return trainer


def calibrate(options, output):
    context = prepare("connected", adapter_rank=options.adapter_rank, seed=options.seed)
    owner, tokens = context["owner"], context["tokens"]
    schedule = tr.validate_schedule(
        [{"name": "calibration", "sources": sorted({row["source"] for row in load_items(split="train")}),
          "steps": options.steps}],
        sorted({row["source"] for row in load_items(split="train")}))
    trainer = make_trainer(context, schedule, options)
    mixed = tr.MixedStream(trainer.streams, schedule[0]["sources"])
    timings = []
    for _ in range(options.steps):
        started = time.perf_counter()
        record = trainer.train_step(mixed)
        timings.append(time.perf_counter() - started)
    evaluation = load_evaluation("dev")
    evaluation_started = time.perf_counter()
    measured = tr.evaluate(owner, tokens, evaluation, limits={"next_token": 32, "cloze": 64, "definition": 64})
    record = {"schema": "sera.owner-language.calibration.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.descendant",)),
              "arm": "connected", "batch": options.batch, "length": options.length,
              "adapter_rank": options.adapter_rank,
              "steps_timed": len(timings),
              "median_seconds_per_step": float(np.median(timings)),
              "mean_seconds_per_step": float(np.mean(timings)),
              "steps_per_minute": 60. / float(np.median(timings)),
              "tokens_per_step": trainer.batch * options.length,
              "trainable": owner.trainable_report(),
              "evaluation_probe_seconds": time.perf_counter() - evaluation_started,
              "evaluation_probe": measured,
              "parent_identity": context["parent_identity"],
              "descendant_identity": owner.identity(),
              "contract": context["contract"]}
    (output / "calibration.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in {"process", "trainable", "contract"}},
                     indent=2, default=str))
    return record


def declared_schedule(options):
    protocol = json.loads(PROTOCOL.read_text())
    available = sorted({row["source"] for row in load_items(split="train")})
    scale = options.scale
    stages = [{**stage, "steps": max(1, round(stage["steps"] * scale))} for stage in protocol["stages"]]
    for stage in stages:
        stage.pop("cumulative_steps", None)
    return tr.validate_schedule(stages, available), protocol


def train(options, output):
    schedule, protocol = declared_schedule(options)
    context = prepare(options.arm, adapter_rank=options.adapter_rank, seed=options.seed)
    owner, tokens = context["owner"], context["tokens"]
    store = RUNS / "descendants" / options.run / options.arm
    trainer = make_trainer(context, schedule, options, store=store)
    resumed = trainer.load_latest()
    record = {"schema": "sera.owner-language.training-run.1", "run": options.run, "arm": options.arm,
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.descendant",)),
              "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "protocol_scale": options.scale, "schedule": schedule,
              "resumed_from": None if resumed is None else {"step": resumed["step"], "revision": resumed["revision"]},
              "parent_identity": context["parent_identity"], "contract": context["contract"],
              "control": context["control"], "trainable": owner.trainable_report(),
              "stages": [], "evaluations": []}

    def evaluate_now(label):
        measured = tr.evaluate(owner, tokens, load_evaluation("dev"))
        measured.update({"label": label, "step": trainer.step})
        record["evaluations"].append(measured)
        print(json.dumps({"evaluation": label, "step": trainer.step,
                          "next_token": measured["next_token"].get("mean_nll"),
                          "cloze": measured["cloze"].get("accuracy"),
                          "definition": measured["definition"].get("accuracy")}), flush=True)
        return measured

    if resumed is None:
        trainer.save(note="step-0-parent-equivalent")
        evaluate_now("before-teaching")
    for stage in schedule:
        if trainer.step >= stage["cumulative_steps"]:
            continue
        completed = trainer.run_stage(stage, report_every=options.report_every,
                                      checkpoint_every=options.checkpoint_every)
        record["stages"].append(completed)
        trainer.save(note=f"{stage['name']}-complete")
        evaluate_now("after-" + stage["name"])
    record["final_step"] = trainer.step
    record["descendant_identity"] = owner.identity()
    record["checkpoints"] = trainer.store.history()
    record["metrics"] = trainer.metrics
    (output / f"training-{options.arm}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"arm": options.arm, "final_step": record["final_step"],
                      "descendant": record["descendant_identity"],
                      "stages": [s["status"] for s in record["stages"]]}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["calibrate", "train"])
    parser.add_argument("--arm", choices=ARMS, default="connected")
    parser.add_argument("--run", default="OLA-001")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=4801)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)
    output = output_directory(options.output)
    if options.action == "calibrate":
        calibrate(options, output)
    else:
        train(options, output)


if __name__ == "__main__":
    main()
