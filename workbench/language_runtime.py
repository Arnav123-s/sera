"""Optional qualified language parameters belong to the same continuing R1 owner."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from experiments.grounded_language.acquisition import (
    guarded_interpret,
    lesson,
    qualified_shapes,
    shape,
    validate_shapes,
)
from experiments.grounded_language.data import examples
from experiments.grounded_language.model import LanguageR1, fingerprint
from sera.session_state import model_identity

from .math_contract import check_translation
from .owner import LiveR1, base_snapshot

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT/"research-continuation/21_grounded_language/qualification.json"
MIGRATABLE_RUNTIMES = {"30bdf634b4f92e6af25fc5114a692dbebdc0b3d75fa090795a0d977a33d965bf"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_source():
    paths = [Path(__file__), Path(__file__).with_name("model.py"),
             Path(__file__).with_name("math_contract.py"),
             ROOT/"experiments/grounded_language/acquisition.py"]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def language_state(owner):
    return {name: value.detach().clone() for name, value in owner.state_dict().items() if name.startswith("language_")}


def apply_language(owner, state):
    expected = language_state(owner)
    if set(state) != set(expected) or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all() for name, value in state.items()):
        raise ValueError("Language checkpoint violates its tensor contract")
    full = owner.state_dict()
    full.update(state)
    owner.load_state_dict(full, strict=True)


def restore(owner, record, *, migrate=False):
    current = runtime_source()
    if record["runtime_source"] != current and not (migrate and record["runtime_source"] in MIGRATABLE_RUNTIMES):
        raise ValueError("The task-time interpreter changed; preserve and migrate this session explicitly")
    path = (ROOT/record["checkpoint"]["path"]).resolve()
    if not path.is_relative_to(ROOT/"runs") or sha(path) != record["checkpoint"]["sha256"]:
        raise ValueError("Language checkpoint identity differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != "sera.workbench.language.1" or saved["language_source"] != fingerprint():
        raise ValueError("Preserve this language state and migrate changed interpreters explicitly")
    LanguageR1.attach(owner, record["initialization_seed"])
    apply_language(owner, saved["state"])
    owner.language_words = list(record["admitted_words"])
    record["runtime_source"] = current
    return owner


def interaction_snapshot(session):
    if not isinstance(session.owner, LanguageR1):
        return base_snapshot(session)
    # The neural core was never modified. Strip only the new interface from the
    # frozen A08 serialization projection; the full owner identity is saved too.
    projected = copy.copy(session)
    projected.owner = copy.deepcopy(session.owner)
    for name in list(projected.owner._modules) + list(projected.owner._parameters):
        if name.startswith("language_"):
            delattr(projected.owner, name)
    projected.owner.__class__ = LiveR1
    return base_snapshot(projected)


def install(owner):
    qualification = read(QUALIFICATION)
    if qualification["status"] != "QUALIFIED_RESTRICTED_INTERFACE" or qualification["language_source"] != fingerprint():
        raise ValueError("No matching qualified language interface is available")
    receipt = qualification["checkpoint"]
    path = (ROOT/receipt["path"]).resolve()
    if not path.is_relative_to(ROOT/"runs") or sha(path) != receipt["sha256"]:
        raise ValueError("Qualified initial checkpoint differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["config"]["scope"] != "interface" or saved["config"]["scratch"] or any(not name.startswith("language_") for name in saved["delta"]):
        raise ValueError("Only a qualified unchanged-core successor may enter this workspace")
    LanguageR1.attach(owner, saved["config"]["seed"])
    apply_language(owner, saved["delta"])
    return {"initialization_seed": saved["config"]["seed"], "qualification_sha256": sha(QUALIFICATION),
            "runtime_source": runtime_source(),
            "qualification": qualification["candidate"], "admitted_words": owner.language_words,
            "shapes": qualified_shapes(), "threshold": qualification["threshold"], "attempts": [],
            "lessons": 0, "paid_teaching_examples": 0, "optimizer_steps": 0}


def persist(owner, record, folder):
    folder.mkdir(parents=True, exist_ok=True)
    value = {"schema": "sera.workbench.language.1", "language_source": fingerprint(), "state": language_state(owner)}
    # Per-transaction artifact; completed files are immutable and referenced by hash.
    path = folder/"language.pt"
    if path.exists():
        raise FileExistsError("Language transaction already contains a checkpoint")
    torch.save(value, path)
    record["checkpoint"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
    record["admitted_words"] = list(owner.language_words)


def execute_acquired(learner, interpretation):
    """Compile the verified interpretation to the owner's preserved affine skill."""
    mode, a, b, c = interpretation["labels"]
    compiled_b = b if mode == 0 else (a*b) % 11
    execution = learner.solve(a, compiled_b, c)
    if execution["solutions"] != interpretation["solutions"]:
        raise ValueError("Preserved guarded program disagrees with independent language arithmetic")
    return {**interpretation, "solutions": execution["solutions"],
            "execution": {key: execution[key] for key in ("equation", "method", "work", "proof_scope", "status")}}


def perform(learner, text, folder):
    """Acquire only the supplied lesson family needed by a qualified request."""
    structure = shape(text)
    base_shapes = set(qualified_shapes())
    novel_shapes = set(qualified_shapes(novel=True))-base_shapes
    word_shapes = set(qualified_shapes(word=True))-base_shapes
    if structure not in base_shapes | novel_shapes | word_shapes:
        return {"status": "WITHHELD", "reason": "This instruction is outside the current lesson library",
                "scope": "Two operation orders, the shown wording families, coefficients zero to ten, modulo eleven."}
    folder.mkdir(parents=True, exist_ok=False)
    # A hard numerical limit leaves both the exact parent session and optimizer state.
    (folder/"parent-session.json").write_text(json.dumps(learner.snapshot(), indent=2)+"\n", encoding="utf-8")
    factual, library = learner.legacy.snapshot(), learner.library.record()
    if learner.language is None:
        learner.language = install(learner.session.owner)
    owner, record = learner.session.owner, learner.language
    if not isinstance(owner, LanguageR1):
        raise ValueError("Language state lost its actual persistent owner")
    before = check_translation(text, guarded_interpret(owner, text, threshold=record["threshold"], shapes=record["shapes"]))
    acquired = None
    if before["status"] != "ACCEPTED":
        if structure in word_shapes:
            kind = "word" if "scale" not in owner.language_words else "wording"
            wording, lexical = "known", True
        else:
            kind, wording, lexical = "wording", "novel" if structure in novel_shapes else "known", False
        seed = 180001+record["lessons"]*101
        teaching = examples(seed, 48 if kind == "word" else 192, "train", wording=wording, lesson=lexical)
        development = examples(seed+1, 128, "development", wording=wording, lesson=lexical)
        (folder/"teaching.json").write_text(json.dumps({"teaching": teaching, "development": development}, indent=2)+"\n", encoding="utf-8")
        checkpoints = []
        def save(state):
            path = folder/f"lesson-{state['step']:04d}.pt"
            reduced = {**state, "state": {n: v for n, v in state["state"].items() if n.startswith("language_")},
                       "best_state": {n: v for n, v in state["best_state"].items() if n.startswith("language_")},
                       "language_source": fingerprint(), "kind": kind, "seed": seed+2, "total_steps": 240,
                       "schema": "sera.workbench.language-lesson-resume.1"}
            torch.save(reduced, path)
            checkpoints.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "step": state["step"]})
            (folder/"pending.json").write_text(json.dumps({"checkpoint": checkpoints[-1], "text": text,
                "parent_session_sha256": sha(folder/"parent-session.json"),
                "initial_owner": before, "original_committed_workspace": "Restore the unchanged preceding transaction, then apply the saved interface tensors and optimizer state."}, indent=2)+"\n", encoding="utf-8")
        acquired = lesson(owner, teaching, development, kind=kind, steps=240, seed=seed+2, checkpoint=save)
        validation_rows = examples(seed+3, 384, "calibration")+examples(seed+4, 128, "calibration", wording="novel")
        if "scale" in owner.language_words:
            validation_rows += examples(seed+5, 128, "calibration", lesson=True)
        validation = validate_shapes(owner, validation_rows)
        withdrawn = sorted(set(record["shapes"])-set(validation["shapes"]))
        record["shapes"] = validation["shapes"]
        record["lessons"] += 1
        record["paid_teaching_examples"] += len(teaching)
        record["optimizer_steps"] += acquired["new_steps"]
        acquired.update(shape_validation=validation, withdrawn_shapes=withdrawn, checkpoints=checkpoints,
                        material="Supplied finite English/arithmetic generator; no external tutorial comprehension.")
        (folder/"lesson-result.json").write_text(json.dumps(acquired, indent=2)+"\n", encoding="utf-8")
    answer = check_translation(text, guarded_interpret(owner, text, threshold=record["threshold"], shapes=record["shapes"]))
    reproof = learner.rebind(factual, library)
    if answer["status"] == "ACCEPTED":
        answer = execute_acquired(learner, answer)
    result = {**answer, "text": text, "before_status": before["status"], "learned_on_request": acquired is not None,
              "lesson": None if acquired is None else {key: acquired[key] for key in ("kind", "teaching_examples", "steps", "development", "withdrawn_shapes")},
              "active_wording_shapes": len(record["shapes"]), "reproof_work": reproof,
              "owner": model_identity(owner), "same_owner": learner.solver.neural.owner is learner.solver.components["typed"].owner is owner}
    record["attempts"].append(result)
    persist(owner, record, folder)
    return result
