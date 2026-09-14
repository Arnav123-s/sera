"""Run independent seed processes with one Torch thread each and a common frozen protocol."""

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from sera.stage_three import StageConfig, run_stage_three
from sera.storage import write_json
from sera.training import environment


def worker(output, seed):
    return run_stage_three(Path(output), seeds=[seed], config=StageConfig())[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh cohort directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started, cpu = time.perf_counter(), time.process_time()
    results = []
    workers = min(3, os.cpu_count() or 1)
    orchestration = {"workers": workers, "torch_threads_per_worker": 1,
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "environment": environment(),
                     "scope": "Independent seeded Python processes share this host. Each model uses the same wall cap and one Torch thread. Recorded timings include observed host contention; no exclusive-core or equal-FLOP claim."}
    write_json(output / "orchestration.json", orchestration)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, str(output / "worker-runs" / str(seed)), seed): seed for seed in (0, 1, 2)}
            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()
                source = output / "worker-runs" / str(seed)
                manifest = json.loads((source / "manifest.json").read_text())
                if manifest["environment"]["source_sha256"] != orchestration["environment"]["source_sha256"]:
                    raise ValueError("A worker source differs from the common protocol")
                # Copy completed immutable artifacts; keep each worker invocation intact.
                shutil.copytree(source / str(seed), output / str(seed))
                results.append(result)
                print(f"cohort: completed seed {seed}", flush=True)
        results.sort(key=lambda row: row["seed"])
        manifest = json.loads((output / "worker-runs/0/manifest.json").read_text())
        manifest["seeds"] = [0, 1, 2]
        manifest["orchestration"] = orchestration
        write_json(output / "manifest.json", manifest)
        costs = {"wall_seconds": time.perf_counter() - started,
                 "process_cpu_seconds": time.process_time() - cpu + sum(row["costs"]["process_cpu_seconds"] for row in results),
                 "process_peak_rss_bytes": max(row["costs"]["process_peak_rss_bytes"] for row in results),
                 "peak_rss_scope": "Largest individual worker high-water RSS; not simultaneous aggregate memory",
                 "phases": [{"seed": row["seed"], **phase} for row in results for phase in row["costs"]["phases"]],
                 "orchestration": orchestration}
        write_json(output / "accounting.json", costs)
        write_json(output / "summary.json", {"manifest": manifest, "runs": results, "costs": costs})
    except BaseException as error:
        write_json(output / "cohort-failure.json", {"error": repr(error), "completed_seeds": [row["seed"] for row in results],
                                                    "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
