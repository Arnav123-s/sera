"""Freeze, fit and score every declared control; resume only intact completed cases."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import Observations, canonical, digest, fit, metrics, predict
from .data import bank


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def source_files():
    root = Path(__file__).parent
    return {name: root / name for name in ("__init__.py", "core.py", "data.py", "study.py")}


def peak_memory():
    try:
        import psutil
        memory = psutil.Process().memory_info()
        return {"peak_working_set_bytes": getattr(memory, "peak_wset", None),
                "current_rss_bytes": memory.rss}
    except ImportError:
        return {"peak_working_set_bytes": None, "boundary": "psutil unavailable"}


def run_case(case, protocol, output, decoder_sha):
    seed, family, noise, count = case
    name = f"{seed}-{family}-{noise}-{count}"
    folder = output / "cases" / name
    folder.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    support_full = bank(seed, family, noise, "support", max(protocol["support"]))
    support = {key: value[:count] for key, value in support_full.items()}
    selection = bank(seed, family, noise, "selection", protocol["selection"])
    calibration = bank(seed, family, noise, "calibration", protocol["calibration"])
    evidence = [Observations(rows["t"], rows["y"], role) for rows, role in
                ((support, "support"), (selection, "selection"), (calibration, "calibration"))]
    np.savez_compressed(folder / "observations.npz", **{
        f"{role}_{key}": value for role, rows in
        (("support", support), ("selection", selection), ("calibration", calibration))
        for key, value in rows.items()})
    records = []
    for method in protocol["methods"]:
        artifact, work = fit(method, *evidence, noise,
                             initialization_seed=protocol.get("initializer_seed", 42),
                             steps=protocol["neural_steps"])
        artifact["decoder_sha256"] = decoder_sha
        artifact_path = folder / f"{method}.json"
        atomic_json(artifact_path, artifact)
        records.append({"method": method, "artifact": artifact_path.name,
                        "artifact_sha256": file_hash(artifact_path), "work": work,
                        "artifact_bytes": artifact_path.stat().st_size})
    # All eight artifacts exist before this case's independent query banks are generated.
    queries = {role: bank(seed, family, noise, role, protocol["query_count"])
               for role in ("interpolation", "withheld_arc", "extrapolation")}
    queries["recall"] = support
    if any(set(rows["t"]) & set(support["t"]) for role, rows in queries.items() if role != "recall"):
        raise ValueError("Support/query overlap")
    np.savez_compressed(folder / "queries.npz", **{
        f"{role}_{key}": value for role, rows in queries.items() for key, value in rows.items()})
    for record in records:
        artifact = json.loads((folder / record["artifact"]).read_text())
        result, raw = {}, {}
        inference_start = time.perf_counter()
        for role, rows in queries.items():
            prediction = predict(artifact, rows["t"])
            result[role] = metrics(prediction, rows["truth"], rows["y"])
            for field, value in prediction.items():
                raw[f"{role}_{field}"] = value
        record["inference_wall_seconds"] = time.perf_counter() - inference_start
        record["evaluation"] = result
        record["calibration"] = artifact["calibration"]
        record["selected_names"] = [m["spec"]["name"] for m in artifact.get("models", [])]
        record["class_weights"] = artifact.get("class_weights", [])
        np.savez_compressed(folder / f"{record['method']}-predictions.npz", **raw)
        if file_hash(folder / record["artifact"]) != record["artifact_sha256"]:
            raise ValueError("Scoring changed a frozen artifact")
    report = {"case": {"seed": seed, "family": family, "noise": noise, "support": count},
              "records": records, "cpu_seconds": time.process_time() - cpu,
              "wall_seconds": time.perf_counter() - start, "memory": peak_memory(),
              "files": {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}}
    atomic_json(folder / "record.json", report)
    return {"case_id": name, "path": (folder / "record.json").relative_to(output).as_posix(),
            "sha256": file_hash(folder / "record.json"), "wall_seconds": report["wall_seconds"]}


def validate_completed(output, record):
    path = output / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("Completed record changed")
    report = json.loads(path.read_text())
    for name, expected in report["files"].items():
        if file_hash(path.parent / name) != expected:
            raise ValueError("Completed case artifact changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text())
    files = source_files()
    hashes = {name: file_hash(path) for name, path in files.items()}
    contract = {"protocol_sha256": digest(protocol), "source_sha256": digest(hashes)}
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["contract"] != contract:
            raise ValueError("Source or protocol changed: use a new cohort")
        if state.get("running"):
            raise ValueError("Interrupted case requires a preserved failure/recovery review")
        for record in state["completed"]:
            validate_completed(args.output, record)
    else:
        if any(args.output.iterdir()):
            raise ValueError("Fresh output must be empty")
        (args.output / "source").mkdir()
        for name, path in files.items():
            shutil.copyfile(path, args.output / "source" / name)
        atomic_json(args.output / "protocol.json", protocol)
        state = {"contract": contract, "source_files": hashes, "completed": [], "charged_seconds": 0.,
                 "running": None, "status": "FROZEN", "environment": {
                     "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                     "platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads()},
                 "accounting_boundary": "Per-case wall reservation refunded after completion; includes fit, selection, calibration, serialization and scoring. Research and Python startup excluded."}
        atomic_json(state_path, state)
    lock = args.output / ".writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        done = {record["case_id"] for record in state["completed"]}
        executed = 0
        for seed in protocol["seeds"]:
            for family in protocol["families"]:
                for noise in protocol["noise"]:
                    for count in protocol["support"]:
                        name = f"{seed}-{family}-{noise}-{count}"
                        if name in done:
                            continue
                        available = protocol["fit_budget_seconds"] - state["charged_seconds"]
                        if available < 25 or (args.limit_cases is not None and executed >= args.limit_cases):
                            state["status"] = "BUDGET_OR_REQUESTED_PAUSE"
                            atomic_json(state_path, state)
                            print(canonical({"status": state["status"], "completed_cases": len(done)}), flush=True)
                            return
                        state["running"] = {"case_id": name, "reserved_seconds": 25}
                        state["charged_seconds"] += 25
                        atomic_json(state_path, state)
                        started = time.perf_counter()
                        try:
                            record = run_case((seed, family, noise, count), protocol, args.output, hashes["core.py"])
                        except Exception as exc:
                            state["status"] = "FAILED"
                            state["failure"] = repr(exc)
                            atomic_json(state_path, state)
                            raise
                        state["charged_seconds"] += time.perf_counter() - started - 25
                        state["completed"].append(record)
                        state["running"] = None
                        state["status"] = "RUNNING"
                        atomic_json(state_path, state)
                        done.add(name)
                        executed += 1
                        if executed % 16 == 0:
                            print(canonical({"completed_cases": len(done), "charged_seconds": state["charged_seconds"]}), flush=True)
        state["status"] = "COMPLETED"
        state["memory"] = peak_memory()
        atomic_json(state_path, state)
        print(canonical({"status": "COMPLETED", "cases": len(done), "charged_seconds": state["charged_seconds"]}), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
