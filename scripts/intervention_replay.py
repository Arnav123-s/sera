"""Independent final replay, retention and actual-owner interrupted continuation."""

import gc

import numpy as np
import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.intervention_assess import predict, qualification, same_qualification
from experiments.intervention_controls import r2_probabilities
from experiments.intervention_model import KINDS, OUT, InterventionSession
from experiments.intervention_source import Source, bank
from scripts.intervention_checks import RUN
from scripts.intervention_study import GOAL, ORIGINAL, STORE, investigate, restore
from scripts.intervention_use import perform
from scripts.structural_study import restore as restore_parent
from scripts.structural_study import substance
from sera.r2 import ControlledInstrument
from workbench.storage import Store


def r2_replay(control, queries):
    model = ControlledInstrument(dimension=4, rank=2)
    state = {}
    for name, value in control["weights"].items():
        dtype = getattr(torch, value["dtype"].removeprefix("torch."))
        tensor = torch.tensor(value["real"], dtype=dtype)
        if value["imag"] is not None:
            tensor = tensor + 1j*torch.tensor(value["imag"], dtype=dtype)
        state[name] = tensor
    model.load_state_dict(state)
    with torch.no_grad():
        probabilities = r2_probabilities(model, queries).double().numpy()
    difference = float(np.max(np.abs(probabilities-control["prediction"])))
    if difference > 2e-6:
        raise ValueError("Saved trained R2 did not replay")
    return difference


def lifecycle():
    parent, growth = restore_parent()
    session = InterventionSession(parent)
    source = Source("pulse_loss", 47099, "lifecycle")
    session.start("lifecycle", GOAL, source.identity, ORIGINAL)
    for p in bank("initial"):
        session.observe("lifecycle", source.observe(session.commit_probe("lifecycle", p)))
    # Exact interruption after the eighth observation, before its scheduled fit.
    awaiting_fit = session.state()
    write(RUN / "lifecycle-awaiting-fit.json", awaiting_fit)
    session.fit("lifecycle")
    session.commit_probe("lifecycle", ORIGINAL)
    pending = session.state()
    write(RUN / "lifecycle-pending.json", pending)
    investigate(session, "lifecycle", source, "information", RUN / "lifecycle-reference-progress.json")
    expected = session.state()
    write(RUN / "lifecycle-completed.json", expected)
    del session, parent, growth
    gc.collect()
    parent, growth = restore_parent()
    resumed = InterventionSession(parent, pending)
    investigate(resumed, "lifecycle", source, "information", RUN / "lifecycle-resumed-progress.json")
    if resumed.state() != expected:
        raise ValueError("Committed probe did not resume identically")
    # A completed loop must never call an evidence source again.
    class UnavailableSource:
        def __getattr__(self, name):
            raise AssertionError("Completed inquiry attempted new acquisition: "+name)
    investigate(resumed, "lifecycle", UnavailableSource(), "information", RUN / "must-not-be-written.json")
    if resumed.state() != expected:
        raise ValueError("Completed inquiry changed state or credit")
    del resumed, parent, growth
    gc.collect()
    parent, growth = restore_parent()
    resumed = InterventionSession(parent, awaiting_fit)
    # Perform the pending original request after the deferred fit, then continue.
    resumed.fit("lifecycle")
    resumed.commit_probe("lifecycle", ORIGINAL)
    investigate(resumed, "lifecycle", source, "information", RUN / "lifecycle-fit-resumed-progress.json")
    if resumed.state() != expected:
        raise ValueError("Post-observation boundary did not resume identically")
    return {"actual_owner": True, "committed_probe_exact": True, "awaiting_fit_exact": True,
            "repeat_completion_read_only": True, "completed_observations_repeated": 0,
            "qualified": resumed.subjects["lifecycle"]["qualification"]["accepted"]}


def main():
    torch.set_num_threads(1)
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve the completed independent audit")
    final = read(OUT / "final.json")
    if not final["integration_gate"]:
        raise ValueError("Prospective gate rejected promotion; retain the candidate owner")
    session, growth = restore()
    baseline = read(OUT / "LOCAL_RECONCILIATION.json")
    tensors = session.owner.state_dict()
    changed = [k for k, v in baseline["tensors"].items() if digest(tensors[k].detach().tolist()) != v["sha256"]]
    if changed:
        raise ValueError("Inherited tensors changed")
    parent_state = session.base.state()
    parent_state["owner"] = baseline["owner"]
    if digest(parent_state) != baseline["parent_state"]:
        raise ValueError("Inherited field history changed")
    earlier = Store(ROOT / "runs/sera-counterfactual-qualified-live").read()["state"]
    challenged = session.base.base
    if challenged.credits != earlier["credits"] or challenged.training_receipts != earlier["training_receipts"]:
        raise ValueError("Corrected novelty/procedure credit changed")
    if sorted(challenged.base.records) != baseline["route_ids"]:
        raise ValueError("Retained route portfolio changed")
    grounded = growth.base.base.base.base.base.base.grounded
    if grounded.owner is not session.owner or grounded.gate != baseline["source_gate"]:
        raise ValueError("Shared language owner or applicability gate changed")
    taught = read(ROOT / "research-continuation/45_counterfactual_inquiry/inventory.json")["grounded_bindings"]
    bindings = {r["entry"]["id"]: grounded.bind(r["entry"]["term"], r["entry"]["text"], r["entry"]["source"]) for r in taught}
    if substance(bindings) != substance(read(RUN / "parent-bindings.json")):
        raise ValueError("Retained meaning changed")
    tasks = read(OUT / "retention-tasks.json")
    before = session.state()
    actual = [perform(session, growth, t) for t in tasks]
    if substance(actual) != substance(read(RUN / "parent-results.json")) or session.state() != before:
        raise ValueError("Earlier practical behavior changed")
    write(RUN / "retained-results.json", actual)
    write(RUN / "retained-bindings.json", bindings)
    replay = []
    for path in sorted((RUN / "final/completed").glob("*.json")):
        saved = read(path)
        row, state = saved["result"], saved["state"]
        subject = row["subject"]
        record = session.subjects[subject]
        if session.subject_state(subject) != state:
            raise ValueError("Live state and final evidence differ")
        source = Source(row["family"], row["seed"], subject)
        if source.identity != row["source"]:
            raise ValueError("Source identity changed")
        for observation, decision in zip(record["observations"], record["decisions"], strict=True):
            if source.observe(decision) != observation:
                raise ValueError("Independent source or actuator outcome did not replay")
        q = record["qualification"]
        if source.assessment("selection") != q["selection"] or source.assessment("adequacy") != q["audit"]:
            raise ValueError("Independent assessment outcomes changed")
        weights = {k: session.model(subject, k).weight.detach().tolist() for k in KINDS}
        assessed = qualification(subject, record, weights, q["selection"], q["audit"])
        if not same_qualification(q, assessed):
            raise ValueError("Admission changed on independent replay")
        yp = predict(row["final_programs"], weights[q["selected"]])
        difference = float(np.max(np.abs(yp-row["final_prediction"])))
        if difference > 1e-10 or not np.array_equal(source.truth(row["final_programs"]), row["final_truth"]):
            raise ValueError("Final numerical replay changed")
        r2_difference = r2_replay(row["controls"]["current_r2_fitted"], row["final_programs"]) if "controls" in row else None
        replay.append({"subject": subject, "accepted": q["accepted"], "probability_difference": difference,
                       "r2_difference": r2_difference, "observations": len(record["observations"]),
                       "actuation": q["actuation"], "independent_fit_difference": q["independent_fit"]["max_prediction_difference"]})
    write(RUN / "independent-replay.json", replay)
    added = {k: v for k, v in tensors.items() if k not in baseline["tensors"]}
    summary = {"passed": True, "owner": session.identity(), "parent_owner": baseline["owner"],
               "retained_tensors": len(baseline["tensors"]), "changed_tensors": changed,
               "new_tensor_records": len(added), "new_tensor_bytes": sum(v.numel()*v.element_size() for v in added.values()),
               "new_learned_coefficients": sum(m.weight.numel() for m in session.owner.intervention_models.values()),
               "total_tensor_bytes": sum(v.numel()*v.element_size() for v in tensors.values()),
               "retained_routes": len(baseline["route_ids"]), "retained_practical_tasks": len(tasks),
               "retained_taught_meanings": len(bindings), "recurrent_state_bytes": session.owner.core_state_bytes(),
               "shared_owner": True, "source_gate": grounded.gate, "corrected_credit_unchanged": True,
               "credit_identity": digest(challenged.credits), "training_receipts_identity": digest(challenged.training_receipts),
               "exact_replay_episodes": len(replay), "history_revisions": Store(STORE).verify_history(),
               "final_sha256": sha(OUT / "final.json"), "hypothetical_requests_read_only": True}
    examples = [
        {"id": "return-to-original-question", "kind": "intervention_what_if", "subject": "mixed-47101-information"},
        {"id": "compare-actual-pulse-histories", "kind": "intervention_plan", "subject": "mixed-47101-information",
         "programs": [{"ticks": 12, "pulses": []}, {"ticks": 12, "pulses": [6]}, {"ticks": 12, "pulses": [3, 6, 9]}]},
        {"id": "explain-which-mechanism-survived", "kind": "intervention_explain", "subject": "pulse_loss-47101-information"},
        {"id": "check-whether-the-pulse-worked", "kind": "intervention_explain", "subject": "unreliable_pulses-47101-information"},
        {"id": "retain-passive-ambiguity", "kind": "intervention_what_if", "subject": "mixed-47101-passive"},
        {"id": "retain-unresolved-explanation", "kind": "intervention_explain", "subject": "omitted-47101-information"}, *tasks]
    write(OUT / "example-tasks.json", examples)
    write(RUN / "example-results.json", [perform(session, growth, t) for t in examples])
    if session.state() != before:
        raise ValueError("Examples changed the saved learner")
    identity = session.identity()
    del session, growth, tensors, added, challenged, grounded
    gc.collect()
    restored, growth = restore()
    if restored.identity() != identity or restored.state() != before:
        raise ValueError("Actual successor failed exact restoration")
    summary["exact_owner_restore"] = True
    del restored, growth, before
    gc.collect()
    summary["lifecycle"] = lifecycle()
    summary["practical_examples"] = len(examples)
    write(OUT / "audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
