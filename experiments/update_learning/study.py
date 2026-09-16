"""Measured-outcome meta-regression with successive states and crossed K/eta controls."""

import argparse
import gzip
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np

from workbench.storage import digest, encoded
from workbench.streams import METHODS, choose, features, predictions

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/20_live_workbench"


def mechanism(seed, family, length):
    """Assessor-only physical mechanism; meta-features receive only observed values."""
    rng = np.random.default_rng(seed)
    values = [float(rng.normal()), float(rng.normal())]
    for i in range(2, length):
        if family == "switching":
            rho = (.88, -.35, .65, .2)[(i//23) % 4]
            target = (0., 2., -1., 3.)[(i//31) % 4]
            value = rho*values[-1]+(1-rho)*target+rng.normal(0, .12)
        elif family == "wave_drift":
            value = .015*i+np.sin(i/(3.5+.02*i))+.25*np.sin(i/1.7)+rng.normal(0, .08)
        elif family == "teaching":
            rho = .3+.6*np.sin(i/29)**2
            target = 1.5*np.sin(i/13)+(.7 if i % 47 > 25 else 0)
            value = rho*values[-1]+(1-rho)*target+rng.normal(0, .1)
        else:
            raise ValueError("Unknown assessor mechanism")
        values.append(float(value))
    return values


class Eta:
    def __init__(self, record=None):
        self.gram = np.eye(8)*.5 if record is None else np.array(record["gram"])
        self.rhs = np.zeros(8) if record is None else np.array(record["rhs"])
        self.examples = 0 if record is None else record["examples"]

    def weights(self):
        return np.linalg.solve(self.gram, self.rhs)

    def observe(self, features_, losses, scale):
        targets = np.log1p(np.asarray(losses)/max(scale**2, 1e-12))
        self.gram += features_.T@features_
        self.rhs += features_.T@targets
        self.examples += len(losses)

    def record(self):
        return {"gram": self.gram.tolist(), "rhs": self.rhs.tolist(), "examples": self.examples,
                "weights": self.weights().tolist()}


def collect(values, eta):
    losses, records = [], []
    for i in range(6, len(values)):
        history = values[:i]
        pred, _ = predictions(history)
        x = features(history, pred, losses)
        outcome = np.square(pred-values[i])
        scale = max(float(np.std(history[-32:])), 1e-6)
        # Meta-training starts only after the real next observation arrives.
        eta.observe(x, outcome, scale)
        losses.append(outcome.tolist())
        records.append({"index": i, "features": x.tolist(), "observed_loss": outcome.tolist(), "scale": scale})
    return records


def future_block(past, future, eta):
    original = list(past)
    arms = {name: {"history": list(past if name.startswith("newK") else past[-8:]), "losses": []}
            for name in ("oldK_oldEta", "oldK_newEta", "newK_oldEta", "newK_newEta", "newK_fixedAR16", "newK_persistence")}
    # Warm up only procedure-loss context, using that arm's permitted prior K.
    for arm in arms.values():
        h = arm["history"]
        for i in range(max(6, len(h)-12), len(h)):
            pred, _ = predictions(h[:i])
            arm["losses"].append(np.square(pred-h[i]).tolist())
    weights = eta.weights()
    results = {name: [] for name in arms}
    acquired = []
    for value in future:
        for name, arm in arms.items():
            history = arm["history"]
            pred, _ = predictions(history)
            x = features(history, pred, arm["losses"])
            selected, _ = choose(history, pred, arm["losses"], weights if "newEta" in name else None)
            if name.endswith("fixedAR16"):
                selected = METHODS.index("ar2_16")
            if name.endswith("persistence"):
                selected = 0
            outcome = np.square(pred-value)
            results[name].append({"prediction": float(pred[selected]), "observed": value,
                                  "squared_error": float(outcome[selected]), "method": METHODS[selected]})
            if name == "newK_newEta":
                acquired.append((x.tolist(), outcome.tolist(), max(float(np.std(history[-32:])), 1e-6)))
            arm["losses"].append(outcome.tolist())
            history.append(value)
    # The next generation inherits both this acquired K and independently updated eta.
    successor = Eta(eta.record())
    for x, losses, scale in acquired:
        successor.observe(np.array(x), losses, scale)
    return {"arms": results, "K_before_sha256": digest(original), "K_after": original+list(future),
            "eta_before": eta.record(), "eta_after": successor.record(), "meta_observations": acquired}, successor


def source_map():
    paths = [Path(__file__), ROOT/"workbench/streams.py", ROOT/"workbench/storage.py",
             ROOT/"scripts/run_workbench_bounded.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write(path, value):
    path.write_bytes(encoded(value))


def freeze(name, final, lives):
    directory = RELEASE/name
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"name": name, "final": final, "teaching_seeds": list(range(130001, 130007)),
                "seeds": list(range(132001 if final else 131001, (132001 if final else 131001)+lives)),
                "families": ["switching", "wave_drift"], "prefix": 24, "generations": 3,
                "observations_per_generation": 24, "source": source_map(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "selection": "One prespecified ridge meta-regressor; no post-final tuning. Only observed errors train eta.",
                "gate": {"minimum_gain_vs_fixed_error_selector": .05, "paired_95_percent_lower_positive": True,
                         "beats_fixed_ar16_and_persistence": True, "no_generation_mean_regression": True},
                "unit": "Independent continuing lifetime; repeated observations/generations are clustered.",
                "scope": "Learned selection of supplied numerical update windows, not action acquisition, new representation or general procedure synthesis.",
                "K_control": "Old K keeps the last eight permitted prior values; new K retains the full acquired past. Both acquire the identical future observations. Old history remains archived.",
                "continuity": "Each successor inherits all preceding observed values and meta-regression normal equations; no lifetime reset between generations.",
                "cost": "Same observations and six candidate fits per arm, all fitting/teaching counted. Unequal procedure overhead reported; no equal-FLOP claim."}
    if final:
        protocol["training_reference"] = {
            name: {"path": (RELEASE/"A09-PILOT-001"/name).relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256((RELEASE/"A09-PILOT-001"/name).read_bytes()).hexdigest()}
            for name in ("eta-trained.json", "teaching.json.gz")}
    write(directory/"protocol.json", protocol)
    with zipfile.ZipFile(directory/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in protocol["source"]:
            archive.write(ROOT/path, path)
    return directory


def run(directory):
    directory = Path(directory)
    if (directory/"summary.json").exists():
        raise ValueError("Completed cohort will not restart")
    protocol = json.loads((directory/"protocol.json").read_text())
    if protocol["source"] != source_map():
        raise ValueError("Frozen source changed")
    started = time.perf_counter()
    if protocol["final"]:
        for reference in protocol["training_reference"].values():
            if hashlib.sha256((ROOT/reference["path"]).read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("Previously trained procedure changed")
        eta = Eta(json.loads((ROOT/protocol["training_reference"]["eta-trained.json"]["path"]).read_text()))
    else:
        eta = Eta()
        teaching = []
        for seed in protocol["teaching_seeds"]:
            values = mechanism(seed, "teaching", 96)
            teaching.append({"seed": seed, "observations": values, "training": collect(values, eta)})
        (directory/"teaching.json.gz").write_bytes(gzip.compress(encoded(teaching), mtime=0))
    write(directory/"eta-trained.json", eta.record())
    records = []
    for index, seed in enumerate(protocol["seeds"]):
        family = protocol["families"][index % len(protocol["families"])]
        values = mechanism(seed, family, 96)
        history, procedure = values[:24], Eta(eta.record())
        generations = []
        for generation in range(3):
            offset = 24+generation*24
            result, procedure = future_block(history, values[offset:offset+24], procedure)
            history = result["K_after"]
            result["generation"] = generation+1
            generations.append(result)
        records.append({"seed": seed, "family": family, "generations": generations})
        # Every completed lifetime is an exact resumable state and result.
        write(directory/f"lifetime-{seed}.json", records[-1])
    names = list(records[0]["generations"][0]["arms"])
    means = {name: [float(np.mean([row["squared_error"] for generation in record["generations"]
                                  for row in generation["arms"][name]])) for record in records] for name in names}
    delta = np.array(means["newK_oldEta"])-np.array(means["newK_newEta"])
    rng = np.random.default_rng(135001)
    interval = np.quantile(np.mean(rng.choice(delta, (10000, len(delta)), replace=True), axis=1), [.025, .975]).tolist()
    average = {name: float(np.mean(value)) for name, value in means.items()}
    gain = 1-average["newK_newEta"]/average["newK_oldEta"]
    per_generation = [{name: float(np.mean([row["squared_error"] for record in records
                                          for row in record["generations"][g]["arms"][name]])) for name in names} for g in range(3)]
    gate = {"gain": gain >= .05, "interval": interval[0] > 0,
            "strong_controls": average["newK_newEta"] < min(average["newK_fixedAR16"], average["newK_persistence"]),
            "all_generations": all(g["newK_newEta"] < g["newK_oldEta"] for g in per_generation)}
    summary = {"status": "PASS" if all(gate.values()) else "REJECTED", "gate": gate, "average_mse": average,
               "paired_lifetime_mse": means, "eta_relative_gain": gain, "paired_95_interval": interval,
               "generation_mse": per_generation, "independent_lifetimes": len(records), "generations": 3,
               "observations_per_lifetime": 96, "future_predictions": len(records)*3*24*len(names),
               "teaching_observations": 576, "trained_eta_parameters": 8, "elapsed_seconds": time.perf_counter()-started,
               "new_teaching_observations": 0 if protocol["final"] else 576,
               "integration": "No promotion without all gates and an independent audit; rejection preserves the fixed observed-error selector."}
    write(directory/"summary.json", summary)
    print(json.dumps(summary, indent=2))


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory/"protocol.json").read_text())
    if protocol["source"] != source_map():
        raise ValueError("Frozen source changed")
    training = ROOT/protocol["training_reference"]["teaching.json.gz"]["path"] if protocol["final"] else directory/"teaching.json.gz"
    teaching = json.loads(gzip.decompress(training.read_bytes()))
    xs, ys = [], []
    for stream in teaching:
        for row in stream["training"]:
            xs.extend(row["features"])
            ys.extend(np.log1p(np.array(row["observed_loss"])/max(row["scale"]**2, 1e-12)))
    x = np.vstack([np.array(xs), np.eye(8)*np.sqrt(.5)])
    y = np.r_[ys, np.zeros(8)]
    independent = np.linalg.lstsq(x, y, rcond=None)[0]
    trained = json.loads((directory/"eta-trained.json").read_text())
    difference = float(np.max(np.abs(independent-np.array(trained["weights"]))))
    if difference > 1e-8:
        raise ValueError("Independent QR/SVD meta-fit disagrees")
    checked, maximum = 0, 0.
    for seed in protocol["seeds"]:
        record = json.loads((directory/f"lifetime-{seed}.json").read_text())
        previous = None
        for generation in record["generations"]:
            if previous is not None:
                if generation["K_before_sha256"] != digest(previous["K_after"]) or generation["eta_before"] != previous["eta_after"]:
                    raise ValueError("Successor continuity failed")
            for arm in generation["arms"].values():
                for row in arm:
                    difference_ = abs((row["prediction"]-row["observed"])**2-row["squared_error"])
                    maximum = max(maximum, difference_)
                    checked += 1
            previous = generation
    if maximum > 1e-10:
        raise ValueError("Prediction error accounting failed")
    write(directory/"audit.json", {"status": "PASS", "independent_meta_fit_max_difference": difference,
                                   "checked_predictions": checked, "max_squared_error_disagreement": maximum,
                                   "successive_K_and_eta_continuity": True,
                                   "scope": "Independent meta-fit and arithmetic/lineage audit; does not independently reimplement every AR fit."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run", "audit"))
    parser.add_argument("name")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--lives", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "freeze":
        print(freeze(args.name, args.final, args.lives))
    elif args.mode == "run":
        run(RELEASE/args.name)
    else:
        audit(RELEASE/args.name)


if __name__ == "__main__":
    main()
