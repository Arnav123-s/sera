"""Independent replay, exact retention, shared ownership and usable acquisition loop."""

import copy
import json

import numpy as np
import torch

from experiments.self_study.multilingual import DATA, english_probe
from experiments.self_study.multilingual import rows as language_rows
from experiments.stream_curriculum.model import batch
from sera.session_state import model_identity
from workbench.storage import Store

from . import physical
from .common import OUT, ROOT, RUNS, read, restore_parent, sha, write
from .data import DT, identity, training_arrays
from .model import KINDS, RefiningR1, apply, delta
from .runtime import RefinementSession, fingerprint


def observations(row, subject="body-1", start=0.0):
    return [
        {
            "subject": subject,
            "time": start + i * DT,
            "position": value[0] if mask[0] else None,
            "velocity": value[1] if mask[1] else None,
            "command": (row["controls"] + [float(row["future_controls"][0])])[i],
            "available": mask,
            "units": ["m", "m/s", "1"],
            "source": "sera:synthetic:" + row["id"],
            "evidence": "SYNTHETIC",
        }
        for i, (value, mask) in enumerate(zip(row["values"], row["masks"]))
    ]


def languages(owner):
    probes = {
        "en-US": english_probe()[:32],
        **{
            locale: language_rows(DATA / f"{locale}-development.jsonl")[:32]
            for locale in ("es-ES", "fr-FR", "de-DE")
        },
    }
    logits = {}
    owner.eval()
    with torch.no_grad():
        for locale, examples in probes.items():
            x, _, _ = batch(examples, owner.stream_config["vocabulary"])
            logits[locale] = tuple(t.clone() for t in owner.request_logits(x))
    return logits, {k: [r["id"] for r in v] for k, v in probes.items()}


def main():
    torch.set_num_threads(1)
    destination = OUT / "integration-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed audit; use a new declared repair revision")
    parent = restore_parent()
    original = {k: v.clone() for k, v in parent.owner.state_dict().items()}
    before_languages, probe_ids = languages(parent.owner)
    original_math = parent.motion([2, 3, 1], "3", "5", "-1")
    original_goal = copy.deepcopy(parent.research_goals)
    resumptions = []
    x, y = (torch.from_numpy(a) for a in training_arrays(read(RUNS / "train.json")))
    for kind in KINDS:
        for seed in (2801, 2802):
            folder = RUNS / f"{kind}-{seed}"
            initial = torch.load(folder / "step-0200.pt", map_location="cpu", weights_only=True)
            expected = torch.load(folder / "step-0240.pt", map_location="cpu", weights_only=True)
            owner = copy.deepcopy(parent.owner)
            RefiningR1.attach(owner, kind, seed)
            parameters = [p for p in owner.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            apply(owner, initial["delta"])
            optimizer.load_state_dict(initial["optimizer"])
            rng = np.random.default_rng()
            rng.bit_generator.state = initial["rng"]
            torch.set_rng_state(initial["torch_rng"])
            owner.train()
            for _ in range(40):
                indices = rng.integers(0, len(x), 8)
                offset = int(rng.integers(0, x.shape[1] - 40 + 1))
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    (
                        owner.empirical(x[indices, offset : offset + 40])[:, 12:]
                        - y[indices, offset + 12 : offset + 40]
                    )
                    ** 2
                ).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            actual = delta(owner)
            assert all(
                torch.equal(value, expected["delta"][name]) for name, value in actual.items()
            )
            resumptions.append(
                {"kind": kind, "seed": seed, "from_step": 200, "to_step": 240, "exact": True}
            )
    runtime = RefinementSession()
    rows = read(RUNS / "dev.json")
    case = next(r for r in rows if r["family"] == "delayed" and r["condition"] == "clean")
    incoming = observations(case)
    write(OUT / "example-observations.json", incoming)
    before_owner = model_identity(runtime.owner)
    result = runtime.autonomous_refine(
        "predict-body-1", "body-1", case["future_controls"], incoming
    )
    assert result["attempts"][0]["before"]["status"] == "MISSING_KNOWLEDGE"
    assert result["status"] == "EMPIRICAL_PREDICTION"
    assert model_identity(runtime.owner) != before_owner
    after_owner = model_identity(runtime.owner)
    first_snapshot = runtime.snapshot()
    factual_hash = identity(first_snapshot)
    branch = runtime.predict("body-1", [-0.2] * 12, branch="counterfactual")
    assert identity(runtime.snapshot()) == factual_hash
    assert (
        branch["evidence_kind"] == "CONDITIONAL_IMAGINATION"
        and not branch["occurrence_established"]
    )
    replay = RefinementSession(first_snapshot)
    assert replay.predict("body-1", case["future_controls"]) == runtime.predict(
        "body-1", case["future_controls"]
    )
    retained = all(torch.equal(v, runtime.owner.state_dict()[k]) for k, v in original.items())
    assert retained
    assert runtime.exact_motion([2, 3, 1], "3", "5", "-1")["result"] == original_math
    assert runtime.study.research_goals == original_goal
    after_languages, _ = languages(runtime.owner)
    language_equal = {
        k: all(torch.equal(a, b) for a, b in zip(before_languages[k], after_languages[k]))
        for k in before_languages
    }
    assert all(language_equal.values())
    # A different subject/history changes its own learned operator without mutating body-1.
    alternate = next(
        r
        for r in rows
        if r["family"] == "delayed" and r["condition"] == "clean" and r["history"] == 1
    )
    saved_prediction = runtime.predict("body-1", case["future_controls"])["position_velocity"]
    runtime.autonomous_refine(
        "predict-body-2", "body-2", alternate["future_controls"], observations(alternate, "body-2")
    )
    assert (
        runtime.predict("body-1", case["future_controls"])["position_velocity"] == saved_prediction
    )
    assert len(runtime.owner.empirical_weights) == 2
    # New observed data invalidates cached empirical state; a new fitting version retains its predecessor.
    last = incoming[-1]
    extra = copy.deepcopy(last)
    extra.update(
        time=last["time"] + DT,
        position=case["truth"][0][0],
        velocity=case["truth"][0][1],
        command=case["future_controls"][1],
    )
    runtime.observe("body-1", [extra])
    assert runtime.predict("body-1", [0.8])["status"] == "NEEDS_REFINEMENT"
    updated = runtime.refine("body-1")
    assert updated["status"] == "EMPIRICAL_MODEL_LEARNED"
    assert len(runtime.subjects["body-1"]["models"]) == 2
    assert (
        runtime.subjects["body-1"]["models"][0] == first_snapshot["subjects"]["body-1"]["models"][0]
    )
    assert all(torch.equal(v, runtime.owner.state_dict()[k]) for k, v in original.items())
    # Hash protected predecessor files after every integration update.
    protected = read(OUT / "reconciliation.json")["protected"]
    assert all(sha(ROOT / name) == wanted for name, wanted in protected.items())
    raw = read(RUNS / "evaluation-final-physical.json")
    final_rows = read(RUNS / "final.json")
    largest = 0.0
    for row, model, expected in zip(final_rows, raw["models"], raw["predictions"]):
        actual = physical.predict(model, row["values"], row["future_controls"], independent=True)
        largest = max(largest, float(np.max(np.abs(actual - expected))))
    assert largest < 2e-12
    # Publish a separate append-only local runtime lineage; Stage 27 remains untouched.
    store = Store(ROOT / "runs/sera-refinement-live")
    if store.read() is not None:
        raise FileExistsError("Do not overwrite an existing live refinement session")
    revision0 = store.commit(first_snapshot, None)
    store.commit(runtime.snapshot(), revision0)
    assert store.verify_history() == 2
    assert RefinementSession(store.read()).snapshot() == runtime.snapshot()
    write(
        destination,
        {
            "status": "PASS",
            "source": fingerprint(),
            "owner_before_learning": before_owner,
            "owner_after_first_learning": after_owner,
            "final_owner": model_identity(runtime.owner),
            "actual_owner_shared": True,
            "protected_parent_tensors": len(original),
            "protected_files": protected,
            "four_language_logits_identical": language_equal,
            "language_probe_ids": probe_ids,
            "exact_motion": original_math,
            "original_research_goals_retained": list(original_goal),
            "independent_final_rollouts": len(final_rows),
            "max_independent_difference": largest,
            "branch_isolation": True,
            "subject_isolation": True,
            "observation_invalidates_model": True,
            "previous_refinement_retained": True,
            "weight_parameters": sum(p.numel() for p in runtime.owner.empirical_weights.values()),
            "snapshot_exact_replay": True,
            "optimizer_resumptions": resumptions,
            "live_store": "runs/sera-refinement-live",
            "live_revisions": 2,
            "example_task": result,
            "updated_model": updated,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "parent_tensors": len(original),
                "language_examples": 128,
                "independent_final_rollouts": len(final_rows),
                "max_difference": largest,
                "live_store": "runs/sera-refinement-live",
            }
        )
    )


if __name__ == "__main__":
    main()
