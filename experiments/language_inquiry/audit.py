"""Independent consequence arithmetic, portability diagnosis and owned integration."""
import argparse
import hashlib
import math
import platform
import subprocess
import sys
import zipfile

import numpy as np
import torch

from sera.storage import digest
from workbench.storage import Store

from .study import OUT, ROOT, read, sha, write


def numeric_audit():
    folder = OUT/"evaluation"
    frozen = read(folder/"frozen.json")
    with zipfile.ZipFile(folder/"frozen-sources.zip") as archive:
        for name, expected in frozen["sources"].items():
            assert hashlib.sha256(archive.read("experiments/language_inquiry/"+name)).hexdigest() == expected
    for name, expected in frozen["checkpoints"].items():
        assert sha(ROOT/name) == expected
    situations = {}
    for path in folder.glob("world-24*.json"):
        seed = int(path.stem.split("-")[-1])
        for record in read(path):
            situations[(seed, record["stage"])] = record
            assert digest(record["graph"]["payload"]) == record["graph"]["sha256"]
    maximum, target_maximum, checked = 0., 0., 0
    for row in read(folder/"world-records.json"):
        state = situations[(row["seed"], row["stage"])]
        model, frame = state["situation"], row["frame"]
        mu = np.array(model["mean"])
        cholesky = np.linalg.cholesky(np.array(model["covariance"])+np.eye(3)*1e-15)
        parameters = [mu]+[mu+2*cholesky[:, j] for j in range(3)]+[mu-2*cholesky[:, j] for j in range(3)]
        values = []
        for coefficient in parameters:
            angle, velocity = model["angle"], model["velocity"]
            for control in frame["actions"]:
                velocity = max(-.65, min(.65, coefficient[0]*velocity+coefficient[1]*control+coefficient[2]))
                angle += velocity
            values.append([model["center"][0]+model["radius"]*math.cos(angle),
                           model["center"][1]+model["radius"]*math.sin(angle)] if frame["field"] == "position" else [velocity])
        expected = .25*np.array(values[0])+.125*np.sum(values[1:], axis=0)
        maximum = max(maximum, float(np.max(np.abs(expected-np.array(row["prediction"]["value"])))))
        assessor = state["world_assessor_only"]
        spec = assessor["spec"]
        angle, velocity, time = (assessor["state"][key] for key in ("angle", "velocity", "time"))
        for control in row["expected_frame"]["actions"]:
            sign = -1 if time >= spec["change_step"] else 1
            torque = .04*math.sin(2*angle) if spec["family"] == "omitted_torque" else 0.
            velocity = max(-.65, min(.65, spec["rho"]*velocity+sign*spec["gain"]*control+spec["drift"]+torque))
            angle += velocity
            time += 1
        target = np.array([.25+math.sqrt(.73)*math.cos(angle), -.4+math.sqrt(.73)*math.sin(angle)]
                          if row["expected_frame"]["field"] == "position" else [velocity])
        target_maximum = max(target_maximum, float(np.max(np.abs(target-np.array(row["target"])))))
        assert abs(float(np.linalg.norm(expected-target))-row["error"]) < 1e-11
        assert row["within_005"] == (row["error"] <= .05)
        checked += 1
    assert maximum < 1e-11 and target_maximum < 1e-11
    language = read(folder/"language-records.json")
    assert all(r["exact"] == (r["frame"]["actions"] == r["actions"] and r["frame"]["field"] == r["field"]) for r in language)
    write(OUT/"independent-math-audit.json", {"passed": True, "world_method_records": checked,
          "language_records_recounted": len(language), "max_prediction_difference": maximum,
          "max_assessor_difference": target_maximum, "tolerance": 1e-11,
          "boundary": "Independent scalar arithmetic checks forecasts and evaluator targets, not physical adequacy or calibration"})


def portability():
    pack = ROOT/"research/intake/v5-compact-20260916/expanded/SERA_Research_v5"
    sys.path.insert(0, str(pack/"code"))
    from s5.dynamics import WorldLearner
    record = read(pack/"results/dynamics/records.json")[0]
    name = f"{record['family']}-{record['seed']}-{record['method']}"
    expected = read(pack/"results/dynamics"/(name+".json"))["world"]["payload"]
    m = WorldLearner(expected["method"], dt=expected["dt"], noise=expected["noise"], max_observations=expected["max_observations"])
    for event in expected["events"]:
        m.observe(event["before"], event["action"], event["after"], event["id"], event["origin"])
    actual = m.snapshot()["payload"]
    differences, discrete = [], []
    def compare(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                compare(a[key], b[key], path+"/"+key)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+"/"+str(i))
        elif a != b:
            if type(a) is float and type(b) is float:
                differences.append({"path": path, "expected": a, "actual": b, "absolute_difference": abs(a-b)})
            else:
                discrete.append({"path": path, "expected": a, "actual": b})
    compare(expected, actual)
    write(OUT/"packet-portability-diagnosis.json", {"first_failed_model": name,
          "float_differences": len(differences), "largest_differences": sorted(differences, key=lambda d: -d["absolute_difference"])[:12],
          "maximum_absolute_difference": max((d["absolute_difference"] for d in differences), default=0),
          "discrete_differences": discrete, "original_strict_replay": "FAILED_PRESERVED",
          "interpretation": "Diagnostic reconstruction only. The exact checker is unchanged; no broader v5 replay pass is claimed.",
          "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)}})


def integration():
    from .runtime import Runtime
    directory = ROOT/"runs/sera-inquiry"
    directory.mkdir(parents=True, exist_ok=False)
    selection = read(ROOT/"runs/LI-fits-final-001/selected.json")
    runtime = Runtime({"path": selection["checkpoint"], "sha256": selection["sha256"]})
    store, previous, trace = Store(directory), None, []
    preserved = {name: sha(ROOT/name/"current.json") for name in ("runs/sera-workbench", "runs/sera-task-transfer")}
    def commit(label, response):
        nonlocal previous
        previous = store.commit(runtime.snapshot(), previous)
        trace.append({"operation": label, "response": response, "revision": previous["revision"],
                      "owner": previous["owner"], "pointer": read(directory/"current.json")})
    commit("initialize_actual_owner", runtime.status())
    for name, anchor in (("compare", "current"), ("historical", "historical")):
        result = runtime.graph.add(runtime.owner, name, "predict the position of orbit after push right then wait then push left", anchor=anchor)
        commit("ask_"+name, result)
    commit("ask_ambiguous", runtime.graph.add(runtime.owner, "choice", "predict the velocity of orbit after do not wait"))
    runtime.graph.run(runtime.owner, 2)
    runtime.graph.pause("compare")
    commit("pause_after_partial_work", runtime.status())
    runtime = Runtime(previous["checkpoint"], previous)
    runtime.graph.resume(runtime.owner, "compare")
    runtime.graph.clarify("choice", [1])
    runtime.graph.jobs["choice"]["clarification"]["origin"] = "experiment_supplied"
    runtime.graph.run(runtime.owner, 8)
    commit("restored_and_completed", runtime.status())
    assert runtime.graph.jobs["compare"]["result"] == runtime.graph.jobs["historical"]["result"]
    commit("execute_one_local_simulator_action", runtime.execute("compare"))
    assert runtime.graph.jobs["compare"]["status"] == "stale"
    assert runtime.graph.jobs["historical"]["status"] == "complete"
    runtime = Runtime(previous["checkpoint"], previous)
    runtime.graph.rebase(runtime.owner, "compare")
    runtime.graph.run(runtime.owner, 3)
    commit("explicit_rebase_after_real_observation", runtime.status())
    commit("unknown_phrase_withheld", runtime.graph.add(runtime.owner, "before_lesson", "predict the position of orbit after coast"))
    lesson_folder = directory/"lesson-checkpoints"
    lesson_folder.mkdir()
    write(lesson_folder/"parent.json", runtime.snapshot())
    receipt = runtime.teach("coast", 0, checkpoint_folder=lesson_folder)
    assert receipt["admitted"] and receipt["old_correct"] == 13
    commit("supervised_correction", receipt)
    runtime = Runtime(previous["checkpoint"], previous)
    result = runtime.graph.add(runtime.owner, "unfinished", "predict the position of orbit after coast then push right then wait")
    assert result["interpretation"]["frame"]["actions"] == [0, 1, 0]
    runtime.graph.run(runtime.owner, 1)
    runtime.graph.pause("unfinished")
    runtime.graph.add(runtime.owner, "ready", "predict the velocity of orbit after coast then push left")
    runtime.graph.run(runtime.owner, 2)
    commit("new_word_used_and_unfinished_work_preserved", runtime.status())
    assert store.verify_history() == len(trace)
    assert all(sha(ROOT/name/"current.json") == identity for name, identity in preserved.items())
    write(OUT/"integration-transcript.json", trace)
    write(OUT/"integration.json", {"passed": True, "workspace": directory.relative_to(ROOT).as_posix(),
          "revisions": len(trace), "owner": previous["owner"], "parent_stores_unchanged": preserved,
          "parent_owner": runtime.fit["parent_owner"], "shared_owner": True, "status": runtime.status(),
          "boundary": "One supervised correction, one new paid simulator action and reusable exact computation; no learned investigator claim"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("math", "portability", "integration", "releases", "continuation"))
    a = p.parse_args()
    {"math": numeric_audit, "portability": portability, "integration": integration,
     "releases": release_checks, "continuation": continuation_check}[a.operation]()
    print(a.operation+" completed", flush=True)


def release_checks():
    records = []
    for name in ("verify_release", "verify_continuation", "verify_applicability", "verify_v3_release",
                 "verify_transfer_release", "verify_continuing_release", "verify_sparse_release", "verify_task_transfer_release"):
        result = subprocess.run([sys.executable, str(ROOT/"scripts"/(name+".py"))], cwd=ROOT,
                                text=True, encoding="utf-8", capture_output=True, timeout=40)
        records.append({"script": name, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    write(OUT/"prior-release-checks.json", records)
    if any(r["returncode"] for r in records):
        raise ValueError("A preserved release check failed; inspect its recorded output")


def continuation_check():
    result = subprocess.run([sys.executable, str(ROOT/"scripts/verify_continuation.py")], cwd=ROOT,
                            text=True, encoding="utf-8", capture_output=True, timeout=30)
    write(OUT/"continuation-check-repaired.json", {"returncode": result.returncode, "stdout": result.stdout,
          "stderr": result.stderr, "repair": "Evolving Git attributes/ignore rules checked against exact historical Git bytes; all frozen research evidence still checked in place"})
    if result.returncode:
        raise ValueError("Historical continuation check still fails")


if __name__ == "__main__":
    main()
