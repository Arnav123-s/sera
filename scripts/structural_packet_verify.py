"""Read-only v16 replay under the repository's existing resource supervisor."""

import argparse
import contextlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback

from experiments.gap_inquiry import ROOT, read, sha, write


def portable(packet, output):
    """Replay archived scientific inputs; disclose host regeneration differences."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    original = read(output / "state.json")
    tests = original["completed"]["tests"]
    if sha(output / "tests.log") != tests["log_sha256"]:
        raise ValueError("The completed packet tests changed")
    target = output / "portable"
    if target.exists():
        raise FileExistsError("Preserve the earlier portable attempt")
    target.mkdir()
    state = {"strict_receipt": sha(output / "state.json"), "source": original["source"],
             "completed": {"tests": tests | {"reused": True}}, "attempts": [],
             "scope": "Original archived arrays; separate 1e-14 absolute/relative regeneration check; scientific tolerances unchanged"}
    write(target / "state.json", state)
    sys.path.insert(0, str(packet / "code"))
    spec = importlib.util.spec_from_file_location("v16_independent_verifier", packet / "code/verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    make_data = verifier.make_data
    regeneration = []
    def archived_data(family, seed):
        config, generated = make_data(family, seed)
        with np.load(packet / "data" / f"{family}_{seed}.npz") as stored:
            if set(generated) != set(stored.files):
                raise ValueError("The regenerated array schema changed")
            archived = {key: stored[key].copy() for key in stored.files}
        for key, value in generated.items():
            expected = archived[key]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("The regenerated shape or type changed")
            difference = float(np.max(np.abs(value - expected))) if value.size else 0.
            regeneration.append({"family": family, "seed": seed, "array": key,
                                 "different_elements": int(np.count_nonzero(value != expected)),
                                 "elements": value.size, "max_absolute_difference": difference})
            write(target / "regeneration.json", regeneration)
            np.testing.assert_allclose(value, expected, atol=1e-14, rtol=1e-14)
        return config, archived
    verifier.make_data = archived_data
    for name in ("neural", "diagnosis", "operators", "compilation", "manifest"):
        started = time.perf_counter()
        attempt = {"job": name, "status": "RUNNING"}
        state["attempts"].append(attempt)
        write(target / "state.json", state)
        try:
            with (target / (name + ".log")).open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        if name == "manifest":
                            subprocess.run([sys.executable, str(packet / "code/verify_manifest.py")],
                                           stdout=log, stderr=subprocess.STDOUT, check=True)
                            result = {"manifest": "PASS"}
                        else:
                            result = getattr(verifier, name)()
                        print(json.dumps(result, indent=2))
                    except Exception:
                        traceback.print_exc()
                        raise
            attempt.update(status="PASS", seconds=time.perf_counter() - started)
            write(target / (name + ".json"), result)
            state["completed"][name] = {"log_sha256": sha(target / (name + ".log")),
                                         "receipt_sha256": sha(target / (name + ".json"))}
        except Exception:
            attempt.update(status="FAILED", seconds=time.perf_counter() - started)
            raise
        finally:
            write(target / "state.json", state)
    print(json.dumps({"completed": list(state["completed"]), "scope": state["scope"],
                      "regeneration_max_difference": max(r["max_absolute_difference"] for r in regeneration)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    packet = ROOT / "research/intake/v16-structure-core/SERA_v16"
    output = ROOT / "runs/SF-v16-verification"
    if args.portable:
        portable(packet, output)
        return
    subprocess.run([sys.executable, str(packet / "code/run_checks.py"), "--output", str(output)], check=True)
    receipt = json.loads((output / "state.json").read_text())
    if len(receipt["completed"]) != 6:
        raise ValueError("The packet's six declared checks have not completed")
    print(json.dumps({"completed": list(receipt["completed"]), "attempts": receipt["attempts"]}, indent=2))


if __name__ == "__main__":
    main()
