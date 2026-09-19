"""C01/C02 finite integration, with prospective evidence and resumable owner state."""

import argparse
import gc
import json
import subprocess
import time

import numpy as np
import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.structural_check import Source, independent_fit, prediction
from experiments.structural_field import OUT, FieldSession, r1_features
from experiments.structural_inquiry import investigate
from scripts.counterfactual_live import perform as prior_perform
from scripts.counterfactual_live import restore as restore_parent
from workbench.storage import Store

RUN = ROOT / "runs/SF-study-001"
STORE = ROOT / "runs/sera-structural-live"
PARENT = ROOT / "runs/sera-counterfactual-qualified-live/current.json"


def save_checkpoint(path, value):
    """Compact, atomic JSON preserves every value without pretty-print overhead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.1)


def substance(value):
    if isinstance(value, list):
        return [substance(v) for v in value]
    if isinstance(value, dict):
        metadata = {"owner"}
        if value.get("status") == "CHECKED_CONDITIONAL_PORTFOLIO":
            metadata |= {"proof_receipt", "proposal_file", "source_sha256"}
        return {k: substance(v) for k, v in value.items() if k not in metadata}
    return value


def reconcile():
    if (OUT / "LOCAL_RECONCILIATION.json").exists():
        raise FileExistsError("Preserve the reconciled baseline")
    repositories = {}
    for name, directory in (("sera", ROOT), ("kavi", ROOT / "research-continuation/00_sources/kavi-pinned")):
        if not directory.exists():
            repositories[name] = {"available": False}
            continue
        def git(*args):
            return subprocess.run(["git", "-c", "safe.directory=" + directory.as_posix(), *args],
                                  cwd=directory, check=True, text=True, capture_output=True).stdout.strip()
        repositories[name] = {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                              "dirty_files": git("status", "--short").splitlines()}
    resource = read(ROOT / "runs/v3-batch-001/budget.json")
    session, growth = restore_parent()
    examples = read(ROOT / "research-continuation/45_counterfactual_inquiry/example-tasks.json")
    examples += read(ROOT / "research-continuation/42_composed_discovery/example-tasks.json")
    examples += [{"id": "language-" + language, "kind": "request", "text": text} for language, text in (
        ("en", "set an alarm for tomorrow morning"), ("es", "pon una alarma para manana"),
        ("fr", "regle une alarme pour demain"), ("de", "stelle einen wecker fur morgen"))]
    outputs = [prior_perform(session, growth, request) for request in examples]
    write(OUT / "retention-tasks.json", examples)
    write(RUN / "parent-results.json", outputs)
    tensors = {k: {"sha256": digest(v.detach().tolist()), "shape": list(v.shape), "bytes": v.numel()*v.element_size()}
               for k, v in session.owner.state_dict().items()}
    grounded = growth.base.base.base.base.base.base.grounded
    taught = read(ROOT / "research-continuation/45_counterfactual_inquiry/inventory.json")["grounded_bindings"]
    bindings = {r["entry"]["id"]: grounded.bind(r["entry"]["term"], r["entry"]["text"], r["entry"]["source"]) for r in taught}
    if len(bindings) != 26 or not all(v["admitted"] for v in bindings.values()):
        raise ValueError("The inherited taught meanings need reconciliation")
    write(RUN / "parent-bindings.json", bindings)
    record = {"owner": session.identity(), "parent_current": sha(PARENT), "tensors": tensors,
              "repositories": repositories, "resource_ledger_sha256": sha(ROOT / "runs/v3-batch-001/budget.json"),
              "resource_authorization": resource["unlimited_local_time"],
              "owned_job": read(ROOT / "runs/v3-batch-001/active.lock"),
              "route_ids": sorted(session.base.records), "source_gate": grounded.gate,
              "shared_language_owner": grounded.owner is session.owner, "practical_tasks": len(examples),
              "practical_results": sha(RUN / "parent-results.json"), "state_bytes": session.owner.core_state_bytes(),
              "taught_meanings": len(bindings), "binding_results": sha(RUN / "parent-bindings.json"),
              "parent_audit": sha(ROOT / "research-continuation/45_counterfactual_inquiry/audit.json")}
    write(OUT / "LOCAL_RECONCILIATION.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("tensors", "route_ids")}, indent=2))


def verify():
    record = read(OUT / "freeze.json")
    if sha(PARENT) != record["parent_current"]:
        raise ValueError("The qualified parent advanced; preserve and reconcile")
    for p, expected in record["files"].items():
        if sha(ROOT / p) != expected:
            raise ValueError("Changed frozen field source: " + p)


def freeze():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("This prospective protocol is already frozen")
    baseline = read(OUT / "LOCAL_RECONCILIATION.json")
    if sha(PARENT) != baseline["parent_current"]:
        raise ValueError("Reconcile the latest owner before freezing")
    verification_path = ROOT / "runs/SF-v16-verification/portable/state.json"
    verification = read(verification_path)
    if len(verification["completed"]) != 6:
        raise ValueError("Finish the relevant v16 evidence checks")
    paths = [*ROOT.glob("experiments/structural_*.py"), ROOT / "scripts/structural_study.py",
             ROOT / "scripts/structural_use.py", OUT / "PROTOCOL.md", OUT / "LOCAL_RECONCILIATION.json"]
    write(OUT / "freeze.json", {"files": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                                "parent_current": sha(PARENT), "packet_verification": sha(verification_path),
                                "final_seeds": list(range(46101, 46113)), "families": ["radial", "directional", "omitted"]})


def restore(store=STORE):
    verify()
    parent, growth = restore_parent()
    saved = Store(store).read()
    if saved and (saved["freeze"] != sha(OUT / "freeze.json") or saved["parent_current"] != sha(PARENT)):
        raise ValueError("Changed field store contract")
    session = FieldSession(parent, saved["state"] if saved else None)
    return session, growth


def controls(owner, source, record, query):
    acquisition = record["observations"]
    assessment = record["qualification"]["selection"] + record["qualification"]["audit"]
    all_rows = acquisition + assessment
    definitions = {"frozen_radial": (acquisition[:12], "radial"), "radial_extra_data": (all_rows, "radial"),
                   "always_directional": (acquisition, "directional")}
    rows = {}
    for name, (examples, kind) in definitions.items():
        w = independent_fit(examples, kind)
        yp = prediction(query, w, kind)
        rows[name] = {"weights": w.tolist(), "kind": kind, "fitted_pairs": len(examples),
                      "mse": float(np.mean((yp-source.truth(query))**2)), "prediction": yp.tolist()}
    x = np.asarray([r["x"] for r in all_rows])
    f, fq = r1_features(owner, acquisition, x), r1_features(owner, acquisition, query)
    y = np.asarray([r["y"] for r in all_rows])
    w = np.linalg.solve(f.T @ f + np.eye(f.shape[1])*.01, f.T @ y)
    yp = fq @ w
    rows["current_r1_readout"] = {"fitted_pairs": len(all_rows), "feature_dimensions": f.shape[1],
                                  "new_readout_weights": w.tolist(), "new_readout_bytes": w.nbytes,
                                  "context_pairs": len(acquisition), "mse": float(np.mean((yp-source.truth(query))**2)),
                                  "prediction": yp.tolist()}
    return rows


def evaluate(session, source, subject, policy, with_controls):
    record = session.subjects[subject]
    view = session.view(subject)
    weights = session.model(subject, record["selected"]).weight.detach().tolist()
    x, wide = source.bank("final", 256), source.bank("wide", 64, wide=True)
    y, yw = prediction(x, weights, record["selected"]), prediction(wide, weights, record["selected"])
    # All values are issued through the owned view; the independent duplicate is a verifier.
    actual = np.array([view.imagine(row)["acceleration"] for row in x])
    duplicate_error = float(np.max(np.abs(actual-y)))
    if duplicate_error > 1e-10:
        raise ValueError("Owner-connected force differs from the independent duplicate")
    trajectories = []
    initial = source.bank("trajectory", 4)[:, :4] * .4
    for z in initial:
        imagined = view.trajectory(z, [0., 0.])
        truth = source.trajectory(z, [0., 0.])
        trajectories.append({"initial": z.tolist(), "prediction": imagined, "truth": truth.tolist(),
                             "coordinate_mse": float(np.mean((np.asarray(imagined["path"])-truth)**2)),
                             "position_rmse_m": float(np.sqrt(np.mean((np.asarray(imagined["path"])[:, :2]-truth[:, :2])**2))),
                             "velocity_rmse_m_s": float(np.sqrt(np.mean((np.asarray(imagined["path"])[:, 2:]-truth[:, 2:])**2)))})
    plan = view.plan(initial[0], [.25, -.2])
    for branch in plan["branches"]:
        truth = source.trajectory(initial[0], branch["control"])
        branch["independent_goal_error"] = float(np.linalg.norm(truth[-1, :2] - [.25, -.2]))
    result = {"family": source.family, "seed": source.seed, "subject": subject, "policy": policy,
              "source": source.identity, "goal": record["goal"], "accepted": record["qualification"]["accepted"],
              "selected": record["selected"], "weights": weights, "observed_pairs": 72,
              "mse": float(np.mean((actual-source.truth(x))**2)), "wide_mse": float(np.mean((yw-source.truth(wide))**2)),
              "independent_duplicate_error": duplicate_error, "final_x": x.tolist(), "final_truth": source.truth(x).tolist(),
              "final_prediction": actual.tolist(), "wide_x": wide.tolist(), "wide_truth": source.truth(wide).tolist(),
              "wide_prediction": yw.tolist(), "trajectories": trajectories, "plan": plan,
              "qualification": record["qualification"], "investigation": record["investigation"]}
    if record["original_state"] is not None:
        answer = view.imagine(record["original_state"])
        checked = source.truth(record["original_state"])
        result["original_task"] = {"state": record["original_state"], "initial_answer": record["initial_answer"],
                                   "answer_after_investigation": answer, "independent_assessment": checked.tolist(),
                                   "initial_mse": float(np.mean((np.asarray(record["initial_answer"]["acceleration"])-checked)**2)),
                                   "final_mse": float(np.mean((np.asarray(answer["acceleration"])-checked)**2))}
    if with_controls:
        result["controls"] = controls(session.owner, source, record, x)
    return result


def summarize(rows):
    groups = []
    for family in ("radial", "directional", "omitted"):
        for policy in ("disagreement", "balanced"):
            selected = [r for r in rows if r["family"] == family and r["policy"] == policy]
            if not selected:
                continue
            group = {"family": family, "policy": policy, "worlds": len(selected),
                     "accepted": sum(r["accepted"] for r in selected), "directional_selected": sum(r["selected"] == "directional" for r in selected),
                     "mse": float(np.mean([r["mse"] for r in selected])), "wide_mse": float(np.mean([r["wide_mse"] for r in selected])),
                     "trajectory_position_rmse_m": float(np.mean([t["position_rmse_m"] for r in selected for t in r["trajectories"]])),
                     "trajectory_velocity_rmse_m_s": float(np.mean([t["velocity_rmse_m_s"] for r in selected for t in r["trajectories"]]))}
            if policy == "disagreement":
                group["controls"] = {k: float(np.mean([r["controls"][k]["mse"] for r in selected])) for k in selected[0]["controls"]}
            groups.append(group)
    eligible = [r for r in rows if r["policy"] == "disagreement"]
    gate = bool(eligible) and all((r["accepted"] and r["selected"] == "radial") if r["family"] == "radial" else
                                 (r["accepted"] and r["selected"] == "directional" and r["mse"] < r["controls"]["frozen_radial"]["mse"])
                                 if r["family"] == "directional" else not r["accepted"] for r in eligible)
    return {"rows": groups, "integration_gate": gate, "episodes": len(rows), "observed_pairs": 72*len(rows),
            "final_predictions": 256*len(rows), "independent_duplicate_max_error": max(r["independent_duplicate_error"] for r in rows)}


def run(phase):
    if (OUT / (phase + ".json")).exists():
        raise FileExistsError("This finite phase is complete; preserve it")
    if phase == "final":
        verify()
    parent, growth = restore_parent()
    del growth
    store = Store(STORE if phase == "final" else ROOT / "runs/sera-structural-development")
    previous = store.read()
    folder = RUN / phase
    pending = folder / "pending.json"
    saved = read(pending) if pending.exists() else previous["state"] if previous else None
    session = FieldSession(parent, saved)
    rows = [read(p) for p in (folder / "completed").glob("*.json")]
    done = {r["subject"] for r in rows}
    seeds = range(46101, 46113) if phase == "final" else (46001, 46002)
    for family in ("radial", "directional", "omitted"):
        for seed in seeds:
            for policy in ("disagreement", "balanced"):
                subject = f"{family}-{seed}-{policy}"
                if subject in done:
                    continue
                started = time.perf_counter()
                source = Source(family, seed, subject)
                if subject not in session.subjects:
                    session.start(subject, "Predict acceleration at the saved state and investigate whether direction changes the force law",
                                  source.identity, original_state=[.3, .7, -.1, .2, .1, 0.])
                investigate(session, subject, source.bank("initial", 12, narrow=True), source.bank("candidates", 128),
                            source.bank("selection", 16), source.bank("adequacy", 32), source.observe, policy,
                            lambda value: save_checkpoint(pending, value))
                result = evaluate(session, source, subject, policy, policy == "disagreement")
                result["seconds"] = time.perf_counter()-started
                # Commit owner before marking the episode complete; replay resumes the retained result.
                if store.read() != previous:
                    raise ValueError("Another writer changed the structural owner")
                previous = store.commit({"state": session.state(), "owner": session.identity(), "parent_current": sha(PARENT),
                                         "freeze": sha(OUT / "freeze.json") if phase == "final" else "development"}, previous)
                write(folder / "completed" / (subject + ".json"), result)
                rows.append(result)
                print(json.dumps({k: result[k] for k in ("subject", "accepted", "selected", "mse", "wide_mse", "seconds")}), flush=True)
    summary = summarize(rows) | {"owner": session.identity(), "phase": phase}
    write(OUT / (phase + ".json"), summary)
    print(json.dumps(summary, indent=2))
    del session, parent
    gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reconcile", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--phase", choices=("development", "final"))
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.reconcile:
        reconcile()
    elif a.freeze:
        freeze()
    elif a.phase:
        run(a.phase)
    else:
        raise ValueError("Choose a declared phase")


if __name__ == "__main__":
    main()
