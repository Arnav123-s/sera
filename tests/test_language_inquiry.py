import copy
import runpy
import shutil

import numpy as np
import pytest
import torch

from experiments.language_inquiry.data import surface
from experiments.language_inquiry.graph import Investigation, interpret
from experiments.language_inquiry.runtime import Runtime
from experiments.language_inquiry.study import ROOT, read
from sera.session_state import model_identity
from sera.storage import digest


@pytest.fixture(scope="module")
def parent():
    if not (ROOT/"runs/LI-fits-pilot-001/selected.json").exists():
        pytest.skip("Requires the preserved local inquiry pilot; no experiment is restarted by CI")
    row = read(ROOT/"runs/LI-fits-pilot-001/selected.json")
    return Runtime({"path": row["checkpoint"], "sha256": row["sha256"]})


@pytest.fixture
def live(parent):
    return Runtime(parent.checkpoint, parent.snapshot())


def ask(live, name="one", *, anchor="current"):
    return live.graph.add(live.owner, name,
        "predict the position of orbit after push right then wait then push left", anchor=anchor)


def test_actual_common_owner_and_parent_weights(live, tmp_path):
    owner = live.owner
    assert owner is live.session.learner.solver.neural.owner is live.session.learner.solver.components["typed"].owner
    from experiments.language_inquiry.study import base
    previous = base()
    assert all(torch.equal(value, owner.state_dict()[name]) for name, value in previous.owner.state_dict().items())
    # A previous source ID must not bless an unreviewed future implementation.
    for name in ("compatibility.py", "model.py", "data.py", "runtime.py", "graph.py"):
        shutil.copyfile(ROOT/"experiments/language_inquiry"/name, tmp_path/name)
    approved = runpy.run_path(str(tmp_path/"compatibility.py"))
    assert approved["TRAINING_SOURCES"] and approved["RUNTIME_SOURCES"] and approved["GRAPH_SOURCES"]
    for name, key in (("model.py", "TRAINING_SOURCES"), ("runtime.py", "RUNTIME_SOURCES"), ("graph.py", "GRAPH_SOURCES")):
        with (tmp_path/name).open("a") as handle:
            handle.write("\n# unreviewed interpreter change\n")
        assert not runpy.run_path(str(tmp_path/"compatibility.py"))[key]


def test_language_order_changes_executable_frame(live):
    one = interpret(live.owner, "predict the position of orbit after not right but left then push right")
    two = interpret(live.owner, "predict the position of orbit after not left but right then push left")
    assert one["frame"]["actions"] == [-1, 1]
    assert two["frame"]["actions"] == [1, -1]


def test_unknown_subject_and_missing_units_do_not_guess():
    for text in ("predict the position of moon after wait", "predict the speed in mph of orbit after wait",
                 "predict the position of orbit after teleport", "predict the position of orbit after coast"):
        with pytest.raises(ValueError):
            surface(text)


def test_explicit_entity_renaming_is_supplied(live):
    parsed = interpret(live.owner, "after wait report the location of comet", "comet")
    assert parsed["frame"] == {"entity": "comet", "field": "position", "actions": [0]}


def test_ambiguity_preserved_until_clarification(live):
    j = live.graph.add(live.owner, "ambiguous", "predict the velocity of orbit after do not wait")
    assert j["status"] == "clarification"
    live.graph.run(live.owner, 3)
    assert live.graph.jobs["ambiguous"]["cursor"] == 0
    with pytest.raises(ValueError):
        live.graph.clarify("ambiguous", [0])
    live.graph.clarify("ambiguous", [-1])
    live.graph.run(live.owner, 1)
    assert live.graph.jobs["ambiguous"]["result"]["actions"] == [-1]


def test_imagination_never_admits_facts(live):
    before = model_identity(live.owner)
    observations = copy.deepcopy(live.session.learner.session.events)
    ask(live)
    live.graph.run(live.owner, 3)
    assert model_identity(live.owner) == before
    assert live.session.learner.session.events == observations
    assert live.graph.jobs["one"]["result"]["status"] == "CONDITIONAL_UNCALIBRATED"


def test_pause_restore_and_exact_shared_prefix(live):
    ask(live, "a")
    ask(live, "b")
    live.graph.run(live.owner, 1)
    live.graph.pause("a")
    restored = Investigation.restore(live.owner, live.graph.snapshot())
    restored.resume(live.owner, "a")
    restored.run(live.owner, 5)
    live.graph.resume(live.owner, "a")
    live.graph.run(live.owner, 5)
    assert restored.jobs == live.graph.jobs
    assert restored.work["cache_hits"] == 3
    assert restored.work["particle_transitions"] == 21


@pytest.mark.parametrize("location", ["answer", "intermediate", "cache"])
def test_forged_outer_checksum_does_not_hide_derived_tampering(live, location):
    ask(live)
    live.graph.run(live.owner, 3)
    record = live.graph.snapshot()
    if location == "answer":
        record["payload"]["jobs"]["one"]["result"]["value"][0] += 1
    elif location == "intermediate":
        record["payload"]["jobs"]["one"]["states"][0][0] += 1
    else:
        next(iter(record["payload"]["cache"].values()))["states"][0][0] += 1
    record["sha256"] = digest(record["payload"])
    with pytest.raises(ValueError):
        Investigation.restore(live.owner, record)


def test_unrelated_registered_readout_does_not_invalidate(live):
    ask(live)
    key = live.graph.current
    with torch.no_grad():
        next(iter(live.owner.task_readouts.values())).weight.add_(.001)
    live.graph.sync(live.owner)
    assert live.graph.current == key and live.graph.jobs["one"]["status"] == "queued"


def test_shared_recurrent_change_invalidates_language(live):
    ask(live)
    with torch.no_grad():
        next(live.owner.fusion.parameters()).add_(.001)
    live.graph.sync(live.owner)
    assert live.graph.jobs["one"]["status"] == "stale"


def test_tampered_frame_cannot_be_justified_by_recomputed_outer_hash(live):
    ask(live)
    record = live.graph.snapshot()
    record["payload"]["jobs"]["one"]["interpretation"]["frame"]["actions"][0] = -1
    record["sha256"] = digest(record["payload"])
    with pytest.raises(ValueError):
        Investigation.restore(live.owner, record)


def test_factual_revision_stales_current_but_preserves_historical(live):
    ask(live, "current")
    ask(live, "past", anchor="historical")
    live.graph.run(live.owner, 6)
    old = copy.deepcopy(live.graph.jobs["past"]["result"])
    live.observe(1)
    assert live.graph.jobs["current"]["status"] == "stale"
    assert live.graph.jobs["past"]["status"] == "complete"
    assert live.graph.jobs["past"]["result"] == old
    live.graph.rebase(live.owner, "current")
    live.graph.run(live.owner, 3)
    assert len(live.graph.history) == 1
    record = live.snapshot()
    assert Runtime(record["checkpoint"], record).graph.jobs == live.graph.jobs


def test_failed_fit_rolls_back_accepted_state(live, monkeypatch):
    before = live.snapshot()
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate fit failure")
    monkeypatch.setattr(np.linalg, "solve", fail)
    with pytest.raises(np.linalg.LinAlgError):
        live.observe(0)
    assert live.snapshot() == before


def test_returned_records_are_not_mutable_aliases(live):
    returned = ask(live)
    returned["interpretation"]["frame"]["actions"][0] = -1
    assert live.graph.jobs["one"]["interpretation"]["frame"]["actions"][0] == 1


def test_historical_execution_is_rejected(live):
    ask(live, anchor="historical")
    live.graph.run(live.owner, 3)
    with pytest.raises(ValueError):
        live.execute("one")


def test_single_correction_retains_and_rebinds(live, tmp_path):
    ask(live)
    receipt = live.teach("coast", 0, checkpoint_folder=tmp_path)
    assert receipt["admitted"] and receipt["old_correct"] == 13
    assert live.graph.jobs["one"]["status"] == "stale"
    assert interpret(live.owner, "predict the position of orbit after coast")["frame"]["actions"] == [0]
    record = live.snapshot()
    restored = Runtime(record["checkpoint"], record)
    assert model_identity(restored.owner) == model_identity(live.owner)
