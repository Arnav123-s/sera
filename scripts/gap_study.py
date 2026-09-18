"""Finite autonomous investigation, independent evaluation and exact owner continuation."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.discovery_observation import ObservationSession, measurement_rows
from experiments.gap_assessor import assess_development, credit, evaluate, qualify
from experiments.gap_inquiry import (
    ARMS,
    EXPECTED_OWNER,
    OUT,
    PARENT,
    ROOT,
    RUN,
    GapSession,
    contracts,
    digest,
    exhaustive_segmentation,
    global_controls,
    inventory,
    read,
    search,
    sha,
    unchanged_predictions,
    write,
)
from experiments.verified_completion.credit import decode
from sera.session_state import model_identity
from workbench.storage import Store


def parent():
    saved = Store(ROOT / PARENT).read()
    if saved is None or saved["owner"] != EXPECTED_OWNER:
        raise ValueError("Reconcile a changed latest parent before running this protocol")
    return saved


def verify_frozen():
    frozen = read(OUT / "freeze.json")
    if frozen["contracts"] != contracts() or frozen["driver"] != sha(Path(__file__)):
        raise ValueError("Frozen research source changed")
    for name, expected in frozen["inputs"].items():
        if sha(RUN / name) != expected:
            raise ValueError("Frozen input changed")
    return frozen


def prepare():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("Do not restart a prepared study")
    saved = parent()
    observation = ObservationSession(saved["parent"], saved["evidence"])
    if model_identity(observation.owner) != saved["owner"]:
        raise ValueError("Parent restoration mismatch")
    items, selected = inventory(observation)
    write(OUT / "inventory.json", {"owner": saved["owner"], "items": items, "chosen": selected,
                                   "target_supplied": False, "chooser": "fixed evidence-coverage priority"})
    if selected is None:
        raise ValueError("No executable retained evidence gap; preserve this result")
    # Search only the previously retained source area; select by recorded hash,
    # never by a teacher-supplied target-file argument.
    matches = [p for p in (ROOT / "runs/SD-study-001").glob("*.csv") if sha(p) == selected["source_hash"]]
    if len(matches) != 1:
        raise ValueError("The owned source identity must resolve uniquely")
    raw = matches[0].read_bytes()
    rows, indexed = measurement_rows(raw)
    goal = {"parent": saved["owner"], "source": selected["source_hash"], "target": selected["target"],
            "question": "Does the retained explanation cover its own unexamined observations, and what executable refinements explain any mismatch?"}
    goal_id = digest(goal)
    partitions = {name: {"ids": [], "rows": [], "time": [], "value": [], "source": selected["source_hash"], "goal": goal_id}
                  for name in ("development", "credit", "final")}
    for index, row in enumerate(rows):
        if index in selected["consumed"]:
            continue
        identifier = digest([selected["source_hash"], index])
        bucket = int(identifier[:8], 16) % 5
        name = "development" if bucket < 3 else "credit" if bucket == 3 else "final"
        part = partitions[name]
        part["ids"].append(identifier)
        part["rows"].append(index)
        part["time"].append(float(row[selected["input"]]))
        part["value"].append(float(row[selected["target"]]))
    if min(len(p["ids"]) for p in partitions.values()) < 8:
        raise ValueError("Insufficient independent source rows for the fixed protocol")
    development = partitions["development"]
    baseline = unchanged_predictions(observation, selected, development["time"])
    discrepancy = float(np.sqrt(np.mean((np.array(baseline) - development["value"]) ** 2)))
    tolerance = max(.05, .10 * float(np.std(development["value"])))
    discovery = {"goal": goal, "id": goal_id, "input": selected["input"], "target": selected["target"],
                 "source": selected["source"], "source_hash": selected["source_hash"],
                 "reason": selected["reason"], "development_rmse_of_retained_explanation": discrepancy,
                 "mismatch_detected": discrepancy > tolerance, "tolerance": tolerance,
                 "original_scope_preserved": True, "novel_mechanism_labels_supplied": [],
                 "gap_kind": "empirical coverage beyond the previously checked interval"}
    if not discovery["mismatch_detected"]:
        write(OUT / "gap.json", discovery)
        raise ValueError("No actionable mismatch; do not manufacture a discovery")
    for name, part in partitions.items():
        write(RUN / f"{name}.json", part)
        write(RUN / f"{name}-baseline.json", unchanged_predictions(observation, selected, part["time"]))
    write(OUT / "gap.json", discovery)
    write(OUT / "sources.json", {"source": selected["source"], "sha256": selected["source_hash"],
          "retained_path": matches[0].relative_to(ROOT).as_posix(), "rows": len(rows), "indexed": indexed,
          "excluded_previously_consumed_rows": selected["consumed"],
          "partitions": {k: {"rows": v["rows"], "count": len(v["rows"])} for k, v in partitions.items()},
          "teaching_examples_added": 0, "new_source_downloads": 0})
    write(OUT / "freeze.json", {"contracts": contracts(), "driver": sha(Path(__file__)), "parent": saved["owner"],
          "inputs": {p.name: sha(p) for p in sorted(RUN.glob("*.json"))}, "arms": list(ARMS),
          "programs_per_arm": 4096, "final_opened": False, "goal": goal_id})
    print(json.dumps(discovery, indent=2))


def train():
    verify_frozen()
    if (OUT / "selection.json").exists():
        raise FileExistsError("Preserve the completed selected study")
    saved, results = parent(), []
    started = time.perf_counter()
    for arm in ARMS:
        path = RUN / arm / "result.json"
        if path.exists():
            results.append(read(path))
            continue
        session = GapSession(saved)
        result = search(session, read(RUN / "development.json"), arm, RUN / arm)
        results.append(result)
        del session
        gc.collect()
    selected = min(results, key=lambda r: (r["portfolio"][0]["assessment"]["score"], r["arm"]))
    write(OUT / "selection.json", {"arm": selected["arm"], "candidate": selected["portfolio"][0]["id"],
          "basis": "lowest development score; fixed before independent credit/final", "contracts": contracts(),
          "portfolios": {r["arm"]: [c["id"] for c in r["portfolio"]] for r in results},
          "search_wall_seconds_this_invocation": time.perf_counter() - started,
          "counts": {r["arm"]: {k: r[k] for k in ("candidates", "attempts", "duplicates", "fitted")} for r in results}})
    print(json.dumps(read(OUT / "selection.json"), indent=2))


def control_predictions(public, independent, basis):
    result = {}
    for candidate in global_controls(public, basis):
        if candidate["status"] == "PROPOSED":
            result[f"global_degree_{candidate['program']['degree']}"] = evaluate(candidate, independent["time"])
    result["linear_interpolation"] = np.interp(independent["time"], public["time"], public["value"]).tolist()
    dp = exhaustive_segmentation(public, basis)
    values = []
    for t in independent["time"]:
        segment = dp["segments"][sum(t >= c for c in dp["cuts"])]
        values.append(sum(c * (t - segment["origin"]) ** j for j, c in enumerate(segment["weights"])))
    result["dynamic_segmentation"] = values
    return result, dp


def final():
    verify_frozen()
    if (OUT / "final.json").exists():
        raise FileExistsError("The final is one-shot; use audit for replay")
    selection = read(OUT / "selection.json")
    public, independent = read(RUN / "development.json"), read(RUN / "final.json")
    credit_data = read(RUN / "credit.json")
    final_base, credit_base = read(RUN / "final-baseline.json"), read(RUN / "credit-baseline.json")
    tolerance = read(OUT / "gap.json")["tolerance"]
    # Reward accounting shared across comparisons; selected arm first. Different
    # procedure names cannot farm credit for the same source observation.
    ordered = [selection["arm"]] + [a for a in ARMS if a != selection["arm"]]
    results, used = {}, set()
    for arm in ordered:
        result = read(RUN / arm / "result.json")
        assessed = []
        for candidate in result["portfolio"]:
            receipt = credit(candidate, credit_data, credit_base, tolerance, used)
            used.update(receipt["credited_ids"])
            evaluation = qualify(candidate, independent, final_base, tolerance)
            assessed.append({"candidate": candidate["id"], "program": candidate["program"],
                             "credit": receipt, "final": evaluation})
        results[arm] = assessed
    selected = results[selection["arm"]][0]
    basis = read(RUN / selection["arm"] / "result.json")["portfolio"][0]["basis"]
    controls, dp = control_predictions(public, independent, basis)
    control_result = {k: {"predictions": v, "rmse": float(np.sqrt(np.mean((np.array(v) - independent["value"]) ** 2)))} for k, v in controls.items()}
    write(OUT / "final.json", {"selected": selection["arm"], "candidate": selection["candidate"],
          "admitted": selected["final"]["accepted"] and selected["credit"]["assessment"]["accepted"],
          "arms": results, "controls": control_result, "dynamic_control": dp,
          "tolerance": tolerance, "source": public["source"], "final_ids": independent["ids"],
          "unique_credited_ids": sorted(used), "points": sum(r["credit"]["points"] for values in results.values() for r in values),
          "selection_unchanged": True, "contracts": contracts()})
    print(json.dumps({"selected": selection["arm"], "admitted": read(OUT / "final.json")["admitted"],
                      "selected_result": selected["final"], "controls": {k: v["rmse"] for k, v in control_result.items()}}, indent=2))


def audit():
    verify_frozen()
    public, final_data, credit_data = [read(RUN / f"{name}.json") for name in ("development", "final", "credit")]
    final_result = read(OUT / "final.json")
    goal = read(OUT / "gap.json")
    proposal_count, max_difference, used = 0, 0., set()
    for arm in [final_result["selected"]] + [a for a in ARMS if a != final_result["selected"]]:
        state = read(RUN / arm / "checkpoint.json")
        if not state["done"]:
            raise ValueError("Incomplete search checkpoint")
        if len({digest(e["program"]) for e in state["events"]}) != len(state["events"]):
            raise ValueError("Duplicate programs counted as new proposals")
        for event in state["events"]:
            proposal_count += 1
            if event["status"] == "PROPOSED":
                replay = assess_development(event, public)
                if replay != event["assessment"]:
                    raise ValueError("Development receipt differs")
                max_difference = max(max_difference, replay["independent_execution_max_difference"])
        candidates = read(RUN / arm / "result.json")["portfolio"]
        for candidate, recorded in zip(candidates, final_result["arms"][arm], strict=True):
            receipt = credit(candidate, credit_data, read(RUN / "credit-baseline.json"), goal["tolerance"], used)
            if receipt != recorded["credit"] or qualify(candidate, final_data, read(RUN / "final-baseline.json"), goal["tolerance"]) != recorded["final"]:
                raise ValueError("Independent final or credit replay differs")
            used.update(receipt["credited_ids"])
    groups = [set(v["ids"]) for v in (public, final_data, credit_data)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Evidence partitions overlap")
    saved = parent()
    session = GapSession(saved)
    protected = {k: v.detach().clone() for k, v in session.owner.state_dict().items() if not k.startswith("gap_")}
    chosen = read(RUN / final_result["selected"] / "result.json")
    session.owner.gap_policy.load_state_dict(decode(chosen["policy"]))
    session.optimizer.load_state_dict(decode(chosen["optimizer"]))
    for candidate, record in zip(chosen["portfolio"], final_result["arms"][final_result["selected"]], strict=True):
        if record["final"]["accepted"] and record["credit"]["assessment"]["accepted"]:
            session.retain(candidate, record["credit"], record["final"])
    session.open_goals = [{"id": goal["id"], "question": "What independently observed variable explains the retained time-local changes?",
                           "status": "EMPIRICAL_ALTERNATIVES_RETAINED_CAUSE_OPEN", "source": goal["source_hash"]}]
    snapshot = session.state()
    inherited_equal = all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items())
    # Source-gate access uses the typed growth route, not assumed class names.
    growth = session.base.base.base
    source_gate = growth.base.base.base.base.base.base.grounded.gate["autonomous"]
    if not inherited_equal or source_gate:
        raise ValueError("Prior knowledge or source gate changed")
    del session
    gc.collect()
    restored = GapSession(saved, snapshot)
    if restored.state() != snapshot:
        raise ValueError("Exact continuation failed")
    repeat_rejected = forged_rejected = False
    if snapshot["models"]:
        first = next(iter(snapshot["models"].values()))
        rec = next(r for r in final_result["arms"][final_result["selected"]] if r["candidate"] == first["id"])
        try:
            restored.retain(first, rec["credit"], rec["final"])
        except ValueError:
            repeat_rejected = True
        forged = copy.deepcopy(rec["credit"])
        forged["points"] += 1
        try:
            restored.retain(first, forged, rec["final"])
        except ValueError:
            forged_rejected = True
    result = {"passed": inherited_equal and (not snapshot["models"] or repeat_rejected and forged_rejected),
              "proposals_replayed": proposal_count, "max_scalar_discrepancy": max_difference,
              "inherited_tensors": len(protected), "inherited_equal": inherited_equal,
              "source_gate": source_gate, "exact_restore": True, "repeat_credit_rejected": repeat_rejected,
              "forged_credit_rejected": forged_rejected, "retained_candidates": len(snapshot["models"]),
              "same_actual_owner": restored.owner is restored.base.owner, "owner": snapshot["owner"],
              "prior_owner": saved["owner"], "contracts": contracts()}
    write(RUN / "successor.json", snapshot)
    write(OUT / "audit.json", result)
    print(json.dumps(result, indent=2))


def integrate():
    verify_frozen()
    result, replay = read(OUT / "final.json"), read(OUT / "audit.json")
    if not replay["passed"] or not result["admitted"]:
        write(OUT / "integration.json", {"admitted": False, "parent_kept": EXPECTED_OWNER, "reason": "Preserved independent gate outcome"})
        return
    snapshot = read(RUN / "successor.json")
    live = Store(ROOT / "runs/sera-gap-inquiry-live")
    if live.read() is not None:
        raise FileExistsError("Preserve the integrated gap owner")
    # Parent reference avoids duplicating its 236 MB state in every new snapshot.
    value = {"schema": "sera.knowledge-gaps.store.1", "parent_store": PARENT,
             "parent_owner": EXPECTED_OWNER, "state": snapshot, "owner": snapshot["owner"],
             "audit": sha(OUT / "audit.json"), "final": sha(OUT / "final.json")}
    live.commit(value, None)
    write(OUT / "integration.json", {"admitted": True, "store": "runs/sera-gap-inquiry-live", "owner": snapshot["owner"],
          "parent_owner": EXPECTED_OWNER, "models": len(snapshot["models"]), "same_actual_owner": True,
          "scope": "source-bound empirical interpolation and preserved alternative explanations"})
    print(json.dumps(read(OUT / "integration.json"), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "final", "audit", "integrate"))
    args = p.parse_args()
    torch.set_num_threads(1)
    globals()[args.action]()


if __name__ == "__main__":
    main()
