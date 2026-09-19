"""Finite prospective W01/W02 integration through the latest qualified owner."""

import argparse
import gc
import time

import numpy as np
import torch

from experiments.gap_inquiry import ROOT, read, sha, write
from experiments.intervention_assess import choose_program, predict
from experiments.intervention_controls import controls
from experiments.intervention_model import KINDS, OUT, InterventionSession
from experiments.intervention_source import FAMILIES, Source, bank
from scripts.intervention_checks import PARENT, RUN
from scripts.structural_study import restore as restore_parent
from scripts.structural_study import save_checkpoint
from workbench.storage import Store

STORE = ROOT / "runs/sera-intervention-live"
GOAL = "Predict plus-X probability after the original verified pulse history, and investigate which retained explanation accounts for the passive decay and intervention."
ORIGINAL = {"ticks": 10, "pulses": [5]}


def verify():
    frozen = read(OUT / "freeze.json")
    if sha(PARENT) != frozen["parent_current"]:
        raise ValueError("Parent advanced; preserve and reconcile")
    for path, expected in frozen["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Changed frozen intervention source: "+path)


def freeze():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("The final contract is already frozen")
    if sha(PARENT) != read(OUT / "LOCAL_RECONCILIATION.json")["parent_current"]:
        raise ValueError("Parent advanced")
    files = [*ROOT.glob("experiments/intervention_*.py"),
             ROOT / "scripts/intervention_study.py", ROOT / "scripts/intervention_use.py",
             OUT / "PROTOCOL.md", OUT / "LOCAL_RECONCILIATION.json", OUT / "packet-verification.json"]
    write(OUT / "freeze.json", {"schema": "sera.intervention-freeze.1", "parent_current": sha(PARENT),
                                "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                                "final_seeds": list(range(47101, 47109)), "families": list(FAMILIES),
                                "policies": ["information", "balanced", "passive"]})


def restore(store=STORE):
    verify()
    parent, growth = restore_parent()
    saved = Store(store).read()
    if saved is None or saved["freeze"] != sha(OUT / "freeze.json") or saved["parent_current"] != sha(PARENT):
        raise ValueError("Missing or changed intervention successor")
    return InterventionSession(parent, saved["state"]), growth


def investigate(session, subject, source, policy, progress):
    if subject not in session.subjects:
        session.start(subject, GOAL, source.identity, ORIGINAL)
    record = session.subjects[subject]

    def checkpoint():
        save_checkpoint(progress, session.subject_state(subject))

    if record["revision"] in (8, 11, 14, 17, 20) and (not record["fits"] or record["fits"][-1]["revision"] != record["revision"]):
        session.fit(subject)
        checkpoint()
    for index in range(record["revision"], 20):
        if record["pending"] is None:
            chosen = bank("initial")[index] if index < 8 else choose_program(session, subject, bank("candidates"), policy, index-8)
            session.commit_probe(subject, chosen)
            checkpoint()  # Decision exists before the independent source is called.
        session.observe(subject, source.observe(record["pending"]))
        checkpoint()
        if record["revision"] in (8, 11, 14, 17, 20):
            session.fit(subject)
            checkpoint()
    if record["fits"][-1]["revision"] != 20:
        session.fit(subject)
        checkpoint()
    if record["qualification"] is None:
        session.qualify(subject, source.assessment("selection"), source.assessment("adequacy"))
        checkpoint()


def evaluate(session, subject, source, policy):
    record = session.subjects[subject]
    q = record["qualification"]
    query, wide = bank("final"), bank("wide")
    truth, truth_wide = source.truth(query), source.truth(wide)
    answer = session.answer(subject)
    predicted = np.array([session.answer(subject, p)["plus_probability"] for p in query])
    predicted_wide = np.array([session.answer(subject, p)["plus_probability"] for p in wide])
    weights = {k: session.model(subject, k).weight.detach().tolist() for k in KINDS}
    difference = float(np.max(np.abs(predicted-predict(query, weights[q["selected"]]))))
    if difference > 1e-10:
        raise ValueError("Independent probability replay disagreed with owner")
    original_truth = float(source.truth([ORIGINAL])[0])
    initial = record["initial"]
    row = {"subject": subject, "family": source.family, "seed": source.seed, "source": source.identity,
           "policy": policy, "goal_id": record["goal_id"], "accepted": q["accepted"], "selected": q["selected"],
           "weights": weights, "qualification": q, "answer": answer, "initial_answer": initial,
           "final_programs": query, "final_truth": truth.tolist(), "final_prediction": predicted.tolist(),
           "mse": float(np.mean((predicted-truth)**2)), "wide_mse": float(np.mean((predicted_wide-truth_wide)**2)),
           "wide_programs": wide, "wide_truth": truth_wide.tolist(), "wide_prediction": predicted_wide.tolist(),
           "original_truth": original_truth, "original_initial_squared_error": (initial["plus_probability"]-original_truth)**2,
           "original_final_squared_error": (answer["plus_probability"]-original_truth)**2,
           "independent_prediction_difference": difference, "closure_evaluations": sum(v["closure_evaluations"] for f in record["fits"] for v in f["cost"].values()),
           "exposure": {"acquisition": 20*1024, "selection": 12*1024, "adequacy": 20*2048},
           "teacher_parameters_assessor_only": [source.a, source.b, source.c, source.d, source.failure]}
    if policy == "information":
        initial_weights = initial["alternatives"]["representatives"][1][:2]
        row["controls"] = controls(session.owner, record["observations"], q["selection"], query, initial_weights, source.seed)
        for result in row["controls"].values():
            result["mse"] = float(np.mean((np.asarray(result["prediction"])-truth)**2))
    return row


def summarize(rows):
    summaries = []
    for family in FAMILIES:
        for policy in ("information", "balanced", "passive"):
            selected = [r for r in rows if r["family"] == family and r["policy"] == policy]
            if not selected:
                continue
            item = {"family": family, "policy": policy, "worlds": len(selected),
                    "accepted": sum(r["accepted"] for r in selected), "pulse_loss_selected": sum(r["selected"] == "pulse_loss" for r in selected),
                    "mse": float(np.mean([r["mse"] for r in selected])), "wide_mse": float(np.mean([r["wide_mse"] for r in selected])),
                    "original_initial_squared_error": float(np.mean([r["original_initial_squared_error"] for r in selected])),
                    "original_final_squared_error": float(np.mean([r["original_final_squared_error"] for r in selected]))}
            if policy == "information":
                item["controls"] = {k: float(np.mean([r["controls"][k]["mse"] for r in selected])) for k in selected[0]["controls"]}
            summaries.append(item)
    eligible = [r for r in rows if r["policy"] == "information" and r["family"] != "omitted"]
    negative = [r for r in rows if r["policy"] == "information" and r["family"] == "omitted"]
    passive = [r for r in rows if r["policy"] == "passive"]
    accepted = [r for r in eligible if r["accepted"]]
    gate = (len(accepted) >= .8*len(eligible) and all(not r["accepted"] for r in passive)
            and sum(r["accepted"] for r in negative) <= len(negative)*.25
            and all(r["mse"] < .0025 for r in accepted)
            and np.mean([r["original_final_squared_error"] for r in eligible]) < np.mean([r["original_initial_squared_error"] for r in eligible])*.5)
    return {"integration_gate": bool(gate), "episodes": len(rows), "rows": summaries,
            "exposed_shots": sum(sum(r["exposure"].values()) for r in rows),
            "closure_evaluations": sum(r["closure_evaluations"] for r in rows),
            "final_predictions": sum(len(r["final_prediction"]) for r in rows),
            "independent_duplicate_max_error": max(r["independent_prediction_difference"] for r in rows)}


def run(phase, run_name):
    summary_path = OUT / (run_name+".json")
    if summary_path.exists():
        raise FileExistsError("The finite phase is complete; preserve it")
    if phase == "final":
        verify()
        if run_name != "final":
            raise ValueError("One sealed final cohort per protocol")
    folder = RUN / run_name
    parent, growth = restore_parent()
    del growth
    session = InterventionSession(parent)
    rows = []
    for path in sorted((folder / "completed").glob("*.json")):
        value = read(path)
        session.restore_subject(value["result"]["subject"], value["state"])
        rows.append(value["result"])
    done = {r["subject"] for r in rows}
    seeds = range(47101, 47109) if phase == "final" else (47001, 47002)
    for family in FAMILIES:
        for seed in seeds:
            for policy in ("information", "balanced", "passive"):
                subject = f"{family}-{seed}-{policy}"
                if subject in done:
                    continue
                started = time.perf_counter()
                progress = folder / "progress" / (subject+".json")
                if progress.exists():
                    session.restore_subject(subject, read(progress))
                source = Source(family, seed, subject)
                investigate(session, subject, source, policy, progress)
                row = evaluate(session, subject, source, policy)
                row["elapsed_seconds"] = time.perf_counter()-started
                save_checkpoint(folder / "completed" / (subject+".json"), {"result": row, "state": session.subject_state(subject)})
                rows.append(row)
                print({k: row[k] for k in ("subject", "accepted", "selected", "mse", "elapsed_seconds")}, flush=True)
                gc.collect()
    summary = summarize(rows)
    if phase == "final":
        previous = Store(STORE).read()
        if previous is not None:
            raise ValueError("Do not replace an existing live successor")
        Store(STORE).commit({"state": session.state(), "freeze": sha(OUT / "freeze.json"), "parent_current": sha(PARENT)}, None)
        summary["owner"] = session.identity()
    write(summary_path, summary)
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["development", "freeze", "final"])
    p.add_argument("--run-name")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.action == "freeze":
        freeze()
    else:
        run(a.action, a.run_name or a.action)


if __name__ == "__main__":
    main()
