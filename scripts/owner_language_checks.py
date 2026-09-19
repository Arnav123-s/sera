"""The gate that must pass before any sustained teaching is credited.

Each item corresponds to something the review required and the earlier draft did
not do. Every check records what it actually executed; a check that raises is a
failure and is preserved, not skipped.

The interrupted-versus-uninterrupted comparison really does cross a process
boundary: this script re-invokes itself, so the resumed half restores the owner
from disk in a clean interpreter and continues from the saved optimiser, RNG and
source cursors.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.owner_language.isolation import activate, identity

activate()

import torch  # noqa: E402

from experiments.owner_language import training as tr  # noqa: E402
from experiments.owner_language.corpus import load_items  # noqa: E402
from experiments.owner_language.tasks import load_evaluation  # noqa: E402
from scripts.owner_language_train import prepare  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
RUNS = LAB_ROOT / "runs/owner-learning-001"
MINI = [{"name": "mini-definitions", "sources": ["webster"], "steps": 6},
        {"name": "mini-grammar", "sources": ["grammar", "webster"], "steps": 6},
        {"name": "mini-dialogue", "sources": ["plato", "grammar", "webster"], "steps": 6},
        {"name": "mini-mathematics", "sources": ["calculus", "plato", "grammar", "webster"], "steps": 6}]


class Options:
    learning_rate = 1e-3
    batch = 8
    length = 64
    seed = 4801


def check(record, name, function):
    started = time.perf_counter()
    try:
        value = function()
        record["checks"].append({"name": name, "status": "PASS", "seconds": time.perf_counter() - started,
                                 "detail": value})
    except Exception as error:
        record["checks"].append({"name": name, "status": "FAIL", "seconds": time.perf_counter() - started,
                                 "error": type(error).__name__ + ": " + str(error)})
    return record["checks"][-1]


# --------------------------------------------------------------- the checks --

def check_ragged_batches_and_masked_loss(owner, tokens):
    """An uneven final chunk must be ordinary, and padding must never be a target."""
    rows = [tokens.encode("the first passage is deliberately long enough to matter here"),
            tokens.encode("a much shorter one"),
            tokens.encode("and a third of middling length for good measure")]
    lengths = [len(row) for row in rows]
    ids = tr.pad_batch(rows)
    loss, counted = owner.lex_next_token_loss(ids)
    expected = sum(length - 1 for length in lengths)
    if counted != expected:
        raise ValueError(f"masked loss counted {counted} positions, expected {expected}")
    if not torch.isfinite(loss):
        raise ValueError("ragged batch produced a non-finite loss")
    padded = tr.pad_batch(rows + [tokens.encode("tiny")])
    longer, counted_longer = owner.lex_next_token_loss(padded)
    if not torch.isfinite(longer):
        raise ValueError("an uneven final chunk produced a non-finite loss")
    singles = []
    for row in rows:
        value, _ = owner.lex_next_token_loss(torch.tensor([row], dtype=torch.long))
        singles.append(float(value.detach()))
    combined = sum(float(value) * (len(row) - 1) for value, row in zip(singles, rows)) / expected
    difference = abs(combined - float(loss))
    if difference > 1e-4:
        raise ValueError(f"padding changed the loss by {difference}; the mask is not doing its job")
    return {"lengths": lengths, "counted_positions": counted, "padded_batch_positions": counted_longer,
            "loss": float(loss), "same_as_unpadded_within": difference}


def check_evaluation_fails_closed(owner, tokens):
    """No eligible sample must produce UNAVAILABLE, never loss 0 and perplexity 1."""
    empty = tr.measure_next_token(owner, [])
    if empty["status"] != "UNAVAILABLE" or empty["qualifies"]:
        raise ValueError("an empty evaluation did not fail closed")
    if "mean_nll" in empty or "perplexity" in empty:
        raise ValueError("an unavailable evaluation reported a score")
    empty_choice = tr.measure_choice(owner, tokens, [], "cloze")
    if empty_choice["status"] != "UNAVAILABLE" or empty_choice["qualifies"]:
        raise ValueError("an empty choice evaluation did not fail closed")

    class Broken:
        def lex_next_token_loss(self, ids):
            raise RuntimeError("deliberate failure inside the measurement")

        def lex_logits(self, ids):
            raise RuntimeError("deliberate failure inside the measurement")

    rows = load_evaluation("dev")["families"]["next_token"][:4]
    broken = tr.measure_next_token(Broken(), rows)
    if broken["status"] != "UNAVAILABLE" or broken["qualifies"]:
        raise ValueError("a failing evaluation did not fail closed")
    choice_rows = load_evaluation("dev")["families"]["cloze"][:4]
    broken_choice = tr.measure_choice(Broken(), tokens, choice_rows, "cloze")
    if broken_choice["status"] != "UNAVAILABLE":
        raise ValueError("a failing choice evaluation did not fail closed")
    return {"empty_next_token": empty, "empty_cloze": empty_choice,
            "failing_next_token": broken, "failing_cloze": broken_choice}


def check_schedule_validation():
    """The ladder is derived and validated, not indexed twice."""
    available = sorted({row["source"] for row in load_items(split="train")})
    validated = tr.validate_schedule(copy.deepcopy(MINI), available)
    cumulative = [stage["cumulative_steps"] for stage in validated]
    if cumulative != [6, 12, 18, 24]:
        raise ValueError(f"derived cumulative schedule {cumulative} is wrong")
    failures = {}
    for name, stages in {
            "duplicate_stage": [MINI[0], MINI[0]],
            "unknown_source": [{"name": "x", "sources": ["nonexistent"], "steps": 1}],
            "wrong_cumulative": [{**MINI[0], "cumulative_steps": 999}],
            "source_never_taught": [MINI[0]]}.items():
        try:
            tr.validate_schedule(copy.deepcopy(stages), available)
            failures[name] = "ACCEPTED_A_BAD_SCHEDULE"
        except ValueError as error:
            failures[name] = str(error)
    if any(value == "ACCEPTED_A_BAD_SCHEDULE" for value in failures.values()):
        raise ValueError(f"schedule validation is too permissive: {failures}")
    return {"cumulative": cumulative, "rejected": failures}


def check_no_progress_detection(owner, tokens):
    """A source that yields nothing must end the stage, not spin."""
    class Exhausted:
        source = "exhausted"

        def take(self, count):
            return []

        def cursor(self):
            return {"source": self.source, "epoch": 0, "position": 0, "consumed": 0, "tokens": 0, "groups": []}

    mixed = tr.MixedStream({"exhausted": Exhausted()}, ["exhausted"])
    try:
        mixed.batch(4)
    except ValueError as error:
        return {"raised": str(error)}
    raise ValueError("an exhausted stream produced a batch")


def check_every_stage_runs(owner, tokens, output):
    """Run the miniature ladder and read exposure from the cursors."""
    available = sorted({row["source"] for row in load_items(split="train")})
    schedule = tr.validate_schedule(copy.deepcopy(MINI), available)
    trainer = tr.Trainer(owner, tokens, schedule, arm="connected", learning_rate=Options.learning_rate,
                         batch=Options.batch, length=Options.length, seed=Options.seed,
                         store=output / "mini-store")
    trainer.parent_identity = "unused-in-this-check"
    records = [trainer.run_stage(stage, report_every=1000) for stage in schedule]
    unexposed = [record["stage"] for record in records if record["status"] != "COMPLETE"]
    if unexposed:
        raise ValueError(f"stages did not expose every declared source: {unexposed}")
    taught = {source for record in records for source in record["exposure"]}
    if taught != set(available):
        raise ValueError(f"the ladder exposed {sorted(taught)} but the corpus trains {available}")
    return {"stages": [{k: record[k] for k in ("stage", "status", "steps", "exposure")} for record in records]}


def check_checkpoint_carries_the_contract(trainer, contract, mixed):
    # Adam allocates its state lazily, so take one real step before saving;
    # a checkpoint written at step 0 legitimately has an empty optimiser state.
    trainer.train_step(mixed)
    written = trainer.save(note="gate-contract-check")
    if written["baseline_contract"] != contract:
        raise ValueError("checkpoint did not carry the baseline contract")
    for field in ("parent_owner", "contracts", "schema", "subjects", "baseline_owner", "pointer"):
        if field not in written["baseline_contract"]:
            raise ValueError(f"baseline contract is missing {field}")
    if written["descendant_owner"] == written["baseline_contract"]["baseline_owner"]:
        raise ValueError("the descendant did not receive a distinct identity")
    if not written["cursors"] or not written["weights_sha256"]:
        raise ValueError("checkpoint omitted cursors or weight integrity")
    found = trainer.store.read()
    if found is None:
        raise ValueError("the checkpoint pointer did not resolve")
    metadata, payload = found
    if set(payload) != {"language_state", "optimizer", "rng"}:
        raise ValueError(f"checkpoint payload keys are {sorted(payload)}")
    if not payload["optimizer"]["state"]:
        raise ValueError("optimiser state was not saved")
    return {"revision": metadata["revision"], "descendant": metadata["descendant_owner"],
            "parent": metadata["baseline_contract"]["baseline_owner"],
            "cursor_sources": sorted(metadata["cursors"]),
            "payload_keys": sorted(payload), "weights_sha256": metadata["weights_sha256"],
            "rng_streams": sorted(payload["rng"])}


def check_retained_tasks_through_the_descendant(session, growth, output):
    """Real retained queries, executed through the reloaded, modified owner."""
    from scripts.intervention_use import perform
    tasks = json.loads((LAB_ROOT / "research-continuation/47_intervention_understanding/retention-tasks.json").read_text())
    cases, failures = [], []
    for request in tasks:
        try:
            cases.append({"id": request.get("id"), "kind": request.get("kind"), "status": "ANSWERED",
                          "result": perform(session, growth, dict(request))})
        except Exception as error:
            failures.append(request.get("id"))
            cases.append({"id": request.get("id"), "kind": request.get("kind"), "status": "FAILED",
                          "error": type(error).__name__ + ": " + str(error)})
    (output / "gate-retention-cases.json").write_text(json.dumps(cases, indent=1, default=str) + "\n",
                                                      encoding="utf-8")
    if failures:
        raise ValueError(f"retained tasks failed through the modified owner: {failures}")
    return {"declared": len(tasks), "answered": len(tasks) - len(failures), "failed": failures}


# ------------------------------------------------------------ replay phases --

def replay_probe(owner, tokens):
    rows = load_evaluation("dev")["families"]["next_token"][:16]
    measured = tr.measure_next_token(owner, rows)
    return {"mean_nll": measured.get("mean_nll"), "predicted_tokens": measured.get("predicted_tokens"),
            "status": measured["status"]}


def run_replay_phase(phase, steps, store, output):
    """Teach `steps` mini-ladder steps, resuming from `store` when one exists."""
    context = prepare("connected")
    owner, tokens = context["owner"], context["tokens"]
    available = sorted({row["source"] for row in load_items(split="train")})
    schedule = tr.validate_schedule([{"name": "replay", "sources": available, "steps": steps}], available)
    trainer = tr.Trainer(owner, tokens, schedule, arm="connected", learning_rate=Options.learning_rate,
                         batch=Options.batch, length=Options.length, seed=Options.seed, store=store,
                         baseline_contract=context["contract"])
    trainer.parent_identity = context["parent_identity"]
    resumed = trainer.load_latest()
    trainer.run_stage(schedule[0], report_every=1000)
    trainer.save(note=f"replay-{phase}-{steps}")
    result = {"phase": phase, "steps": steps, "step": trainer.step,
              "resumed_from_step": None if resumed is None else resumed["step"],
              "descendant": owner.identity(), "probe": replay_probe(owner, tokens),
              "cursors": {source: stream.cursor() for source, stream in trainer.streams.items()},
              "process": identity()["pid"]}
    (output / f"replay-{phase}.json").write_text(json.dumps(result, indent=2, default=str) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("phase", "step", "resumed_from_step", "descendant")}), flush=True)
    return result


def check_interrupted_replay(output):
    """Cross real process boundaries: halves then whole must agree exactly."""
    script = str(LAB_ROOT / "scripts/owner_language_checks.py")
    steps = 16
    half = steps // 2
    plans = [("uninterrupted", steps, output / "replay-store-whole"),
             ("interrupted-first", half, output / "replay-store-split"),
             ("interrupted-second", steps, output / "replay-store-split")]
    results = {}
    for phase, count, store in plans:
        command = [sys.executable, "-X", "utf8", "-u", script, "--phase", phase,
                   "--steps", str(count), "--store", str(store), "--output", str(output)]
        finished = subprocess.run(command, cwd=str(LAB_ROOT), capture_output=True, text=True)
        if finished.returncode != 0:
            raise ValueError(f"replay phase {phase} failed: {finished.stderr[-2000:]}")
        results[phase] = json.loads((output / f"replay-{phase}.json").read_text())
    whole, resumed = results["uninterrupted"], results["interrupted-second"]
    if resumed["resumed_from_step"] != half:
        raise ValueError(f"the resumed run did not continue from step {half}")
    if whole["descendant"] != resumed["descendant"]:
        raise ValueError("interrupted and uninterrupted development runs produced different descendants")
    if whole["probe"] != resumed["probe"]:
        raise ValueError(f"development probes differ: {whole['probe']} vs {resumed['probe']}")
    if len({results[phase]["process"] for phase in results}) != 3:
        raise ValueError("the replay phases did not run in separate processes")
    return {"steps": steps, "interruption_at": half,
            "identical_descendant": whole["descendant"], "probe": whole["probe"],
            "process_ids": {phase: results[phase]["process"] for phase in results},
            "cursors_match": whole["cursors"] == resumed["cursors"]}


# ------------------------------------------------------------------- driver --

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="all")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--store", default=None)
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(options.output or os.environ.get("SERA_LAB_ATTEMPT", RUNS / "attempts/checks"))
    output.mkdir(parents=True, exist_ok=True)

    if options.phase != "all":
        run_replay_phase(options.phase, options.steps, Path(options.store), output)
        return

    record = {"schema": "sera.owner-language.pre-training-gate.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "process": identity(("experiments.owner_language.descendant",
                                   "experiments.owner_language.training")),
              "supervised_token_present": bool(os.environ.get("SERA_LAB_SUPERVISED")),
              "checks": []}

    # The replay phases each restore their own owner. Run them before this
    # parent restores one of its own: two resident owners would sit against the
    # 2 GiB process-tree cap, and attempt 001 was killed by it.
    check(record, "interrupted-and-uninterrupted-development-replay-agree",
          lambda: check_interrupted_replay(output))

    started = time.perf_counter()
    context = prepare("connected")
    owner, tokens, session, growth = context["owner"], context["tokens"], context["session"], context["growth"]
    record["restore_seconds"] = time.perf_counter() - started
    record["restore"] = context["restore"]
    record["parent_identity"] = context["parent_identity"]
    record["descendant_identity_at_attach"] = owner.identity()
    record["trainable"] = owner.trainable_report()

    check(record, "explicit-baseline-and-descendant-restore", lambda: {
        "store": context["restore"]["store"], "pointer": context["restore"]["pointer"],
        "parent": context["parent_identity"], "descendant": owner.identity(),
        "distinct": context["parent_identity"] != owner.identity(),
        "contract": context["contract"]})
    check(record, "ragged-batches-and-masked-loss", lambda: check_ragged_batches_and_masked_loss(owner, tokens))
    check(record, "evaluation-fails-closed", lambda: check_evaluation_fails_closed(owner, tokens))
    check(record, "schedule-derived-and-validated", check_schedule_validation)
    check(record, "zero-progress-stream-detected", lambda: check_no_progress_detection(owner, tokens))
    check(record, "every-declared-stage-runs-with-recorded-exposure",
          lambda: check_every_stage_runs(owner, tokens, output))

    available = sorted({row["source"] for row in load_items(split="train")})
    schedule = tr.validate_schedule([{"name": "contract", "sources": available, "steps": 1}], available)
    trainer = tr.Trainer(owner, tokens, schedule, arm="connected", learning_rate=Options.learning_rate,
                         batch=Options.batch, length=Options.length, seed=Options.seed,
                         store=output / "contract-store", baseline_contract=context["contract"])
    trainer.parent_identity = context["parent_identity"]
    mixed = tr.MixedStream(trainer.streams, available)
    check(record, "checkpoint-carries-lineage-optimiser-rng-and-cursors",
          lambda: check_checkpoint_carries_the_contract(trainer, context["contract"], mixed))
    check(record, "retained-tasks-answer-through-the-modified-owner",
          lambda: check_retained_tasks_through_the_descendant(session, growth, output))

    record["passed"] = all(item["status"] == "PASS" for item in record["checks"])
    record["failed"] = [item["name"] for item in record["checks"] if item["status"] != "PASS"]
    (output / "pre-training-gate.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"passed": record["passed"], "failed": record["failed"],
                      "checks": [(item["name"], item["status"]) for item in record["checks"]]}, indent=2))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
