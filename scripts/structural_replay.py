"""Independent C02 replay, current-owner retention and concrete usable examples."""

import copy
import gc
import json

import numpy as np
import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.structural_check import (
    independent_fit,
    prediction,
    qualification,
    same_qualification,
)
from experiments.structural_field import OUT, FieldSession
from experiments.structural_inquiry import investigate
from scripts.counterfactual_live import restore as restore_parent
from scripts.structural_study import RUN, STORE, restore, substance
from scripts.structural_use import perform
from workbench.storage import Store


def interrupted_acquisition():
    """Resume the real owner after a committed probe, with no target hint."""
    from experiments.structural_check import Source
    source = Source("directional", 46099, "lifecycle-new-field")
    parent, growth = restore_parent()
    session = FieldSession(parent)
    session.start(source.subject, "Acquire a missing direction-dependent field", source.identity,
                  original_state=[.3, .7, -.1, .2, .1, 0.])
    saved = None
    def stop_after_commit(value):
        nonlocal saved
        saved = value
        if value["pending"] and value["pending"]["decision"] is not None:
            write(RUN / "lifecycle-interrupted.json", value)
            raise InterruptedError("Deliberate interruption after query commitment")
    inputs = (source.bank("initial", 12, narrow=True), source.bank("candidates", 128),
              source.bank("selection", 16), source.bank("adequacy", 32))
    try:
        investigate(session, source.subject, *inputs, source.observe, "disagreement", stop_after_commit)
    except InterruptedError:
        pass
    else:
        raise ValueError("No owned interruption boundary was exercised")
    if saved is None or len(saved["subjects"][source.subject]["observations"]) != 12:
        raise ValueError("Interruption did not preserve exact evidence count")
    # Continue the original process-local object as the deterministic reference.
    investigate(session, source.subject, *inputs, source.observe, "disagreement", lambda _: None)
    expected = session.state()
    write(RUN / "lifecycle-uninterrupted-successor.json", expected)
    del session, growth, parent
    gc.collect()
    parent, growth = restore_parent()
    resumed = FieldSession(parent, saved)
    observations = []
    def measured(*args):
        row = source.observe(*args)
        observations.append(row["id"])
        return row
    investigate(resumed, source.subject, *inputs, measured, "disagreement", lambda _: None)
    if resumed.state() != expected or len(observations) != 60 or len(set(observations)) != 60:
        raise ValueError("Interrupted acquisition did not resume exactly")
    write(RUN / "lifecycle-resumed-successor.json", resumed.state())
    return {"exact": True, "repeated_initial_observations": 0, "remaining_observed_pairs": len(observations),
            "qualification": resumed.subjects[source.subject]["qualification"]["accepted"]}


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve the completed independent audit")
    final = read(OUT / "final.json")
    if not final["integration_gate"]:
        raise ValueError("The prospective integration gate did not pass")
    session, growth = restore()
    baseline = read(OUT / "LOCAL_RECONCILIATION.json")
    tensors = session.owner.state_dict()
    changed = [name for name, r in baseline["tensors"].items() if digest(tensors[name].detach().tolist()) != r["sha256"]]
    if changed:
        raise ValueError("Inherited learner tensors changed: " + str(changed))
    grounded = growth.base.base.base.base.base.base.grounded
    if grounded.owner is not session.owner or grounded.gate != baseline["source_gate"]:
        raise ValueError("Shared language owner or source gate changed")
    taught = read(ROOT / "research-continuation/45_counterfactual_inquiry/inventory.json")["grounded_bindings"]
    bindings = {r["entry"]["id"]: grounded.bind(r["entry"]["term"], r["entry"]["text"], r["entry"]["source"]) for r in taught}
    if substance(bindings) != substance(read(RUN / "parent-bindings.json")):
        raise ValueError("A retained taught meaning changed")
    write(RUN / "retained-bindings.json", bindings)
    if sorted(session.base.base.records) != baseline["route_ids"]:
        raise ValueError("The qualified exact route portfolio changed")
    tasks = read(OUT / "retention-tasks.json")
    before = session.state()
    actual = [perform(session, growth, request) for request in tasks]
    if substance(actual) != substance(read(RUN / "parent-results.json")):
        raise ValueError("Earlier practical capability changed")
    if session.state() != before:
        raise ValueError("A hypothetical or retained request changed field observations or weights")
    write(RUN / "retained-results.json", actual)
    replays = []
    for path in sorted((RUN / "final/completed").glob("*.json")):
        record = read(path)
        subject = record["subject"]
        model_record = session.subjects[subject]
        weights = {kind: session.model(subject, kind).weight.detach().tolist() for kind in ("radial", "directional")}
        receipt = qualification(subject, model_record, weights, record["qualification"]["selection"], record["qualification"]["audit"])
        if not same_qualification(model_record["qualification"], receipt) or model_record["qualification"] != record["qualification"]:
            raise ValueError("Qualification did not replay independently")
        duplicate = independent_fit(model_record["observations"], record["selected"])
        new_prediction = prediction(record["final_x"], duplicate, record["selected"])
        difference = float(np.max(np.abs(new_prediction-record["final_prediction"])))
        if difference > 1e-8:
            raise ValueError("Independent duplicate changed a final prediction")
        trace = record["investigation"]["trace"]
        for item in trace:
            d = item["decision"]
            if d["id"] != digest({k: v for k, v in d.items() if k != "id"}) or item["observation"]["x"] != d["x"]:
                raise ValueError("Evidence did not follow its committed decision")
        replays.append({"subject": subject, "qualified": receipt["accepted"], "difference": difference,
                        "committed_probes": len(trace), "parameter_count": sum(session.model(subject, k).weight.numel() for k in weights)})
    # Resume the already completed acquisition using a callback that must never run.
    selected_subject = next(k for k, r in session.subjects.items() if r["qualification"]["accepted"] and r["selected"] == "directional")
    record = session.subjects[selected_subject]
    def forbidden(*_):
        raise AssertionError("Completed inquiry attempted to repeat evidence or credit")
    repeated = investigate(session, selected_subject, None, None, None, None, forbidden, "disagreement", forbidden)
    if repeated != record["qualification"] or session.state() != before:
        raise ValueError("A completed inquiry repeated work")
    view = session.view(selected_subject)
    row = copy.deepcopy(record["observations"][0])
    try:
        session.observe(selected_subject, row)
    except ValueError:
        pass
    else:
        raise ValueError("Repeated evidence was accepted")
    if view.version != session.view(selected_subject).version or session.state() != before:
        raise ValueError("Rejected evidence changed the model")
    state, identity = session.state(), session.identity()
    history = Store(STORE).verify_history()
    added = {k: v for k, v in tensors.items() if k not in baseline["tensors"]}
    summary = {"passed": True, "owner": identity, "parent_owner": session.parent_owner,
               "retained_tensors": len(baseline["tensors"]), "changed_tensors": changed,
               "retained_routes": len(baseline["route_ids"]), "retained_practical_tasks": len(tasks),
               "retained_taught_meanings": len(bindings),
               "shared_owner": True, "source_gate": grounded.gate, "exact_replay_episodes": len(replays),
               "new_tensor_records": len(added), "new_tensor_bytes": sum(v.numel()*v.element_size() for v in added.values()),
               "total_tensor_bytes": sum(v.numel()*v.element_size() for v in tensors.values()),
               "inherited_tensor_bytes": sum(r["bytes"] for r in baseline["tensors"].values()),
               "recurrent_state_bytes": session.owner.core_state_bytes(),
               "new_learned_coefficients": sum(m.weight.numel() for m in session.owner.physical_fields.values()),
               "history_revisions": history, "read_only_imagination": True, "duplicate_evidence_rejected": True,
               "repeat_completion_is_read_only": True, "final_sha256": sha(OUT / "final.json")}
    write(RUN / "independent-replay.json", replays)
    del tensors, added, grounded, session, growth, view
    gc.collect()
    restored, growth = restore()
    if restored.state() != state or restored.identity() != identity:
        raise ValueError("The actual successor failed exact restoration")
    summary["exact_owner_restore"] = True
    # Explicitly bind a branch to an earlier revision and reject it after new evidence.
    from experiments.structural_check import Source
    subject_data = selected_subject.split("-")
    source = Source(subject_data[0], int(subject_data[1]), selected_subject)
    stale = restored.view(selected_subject)
    restored.observe(selected_subject, source.observe([.3, .4, .1, -.2, 0., 0.], "audit-new-observation", "acquisition"))
    try:
        stale.imagine([.3, .4, .1, -.2, 0., 0.])
    except ValueError:
        summary["stale_view_rejected"] = True
    else:
        raise ValueError("New evidence did not invalidate the old branch")
    # The destructive probe above was process-local; no live checkpoint is written.
    del restored, growth, stale
    gc.collect()
    restored, growth = restore()
    if restored.state() != state:
        raise ValueError("A lifecycle probe altered the saved owner")
    examples = [{"id": "new-direction", "kind": "field_what_if", "subject": selected_subject,
                 "state": [.3, .7, -.1, .2, .1, 0.]},
                {"id": "imagine-motion", "kind": "field_trajectory", "subject": selected_subject,
                 "state": [.3, .4, .1, -.1], "control": [.2, -.1]},
                {"id": "choose-a-control", "kind": "field_plan", "subject": selected_subject,
                 "state": [.3, .4, .1, -.1], "target": [.25, -.2]},
                {"id": "why-this-model", "kind": "field_explain", "subject": selected_subject}, *tasks]
    write(OUT / "example-tasks.json", examples)
    results = [perform(restored, growth, task) for task in examples]
    write(RUN / "example-results.json", results)
    if restored.state() != state:
        raise ValueError("A practical example altered retained state")
    summary["practical_examples"] = len(results)
    del restored, growth
    gc.collect()
    summary["interrupted_acquisition"] = interrupted_acquisition()
    write(OUT / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
