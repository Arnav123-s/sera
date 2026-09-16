"""Independent complex-plane dynamics, QR fits, action choices and paired gates."""

import argparse
import cmath
import gzip
import itertools
import json
import math

import numpy as np
import torch

from experiments.acquisition_dependence.study import load_before

from .core import ContinuingSession
from .study import RELEASE, ROOT, check_contract, choose_owner, parent, read, sha, write


def angle_difference(value):
    return float((value+math.pi) % (2*math.pi)-math.pi)


def independent_rollout(spec, state, actions):
    angle, speed, time = state["angle"], state["velocity"], state["time"]
    points, states = [], []
    for action in actions:
        gain = spec["gain"] * (-1 if time >= spec["change_step"] else 1)
        force = .04*math.sin(2*angle) if spec["family"] == "omitted_torque" else 0
        speed = max(-.65, min(.65, spec["rho"]*speed+gain*action+spec["drift"]+force))
        angle, time = angle+speed, time+1
        z = complex(.25, -.4)+math.sqrt(.73)*cmath.exp(1j*angle)
        points.append([z.real, z.imag])
        states.append({"angle": angle, "velocity": speed, "time": time})
    return np.array(points), states


class IndependentFit:
    def __init__(self, center, window, adapt, detect):
        self.center, self.window, self.adapt, self.detect = center, window, adapt, detect
        self.mean = np.array([.85, .06, 0.])
        self.cov = np.diag([1/20, 1/200, 1/500])
        self.angle = self.velocity = 0.
        self.rows, self.n, self.start, self.changes = [], 0, 0, 0

    def update(self, point, action):
        angle = math.atan2(point[1]-self.center[1], point[0]-self.center[0])
        speed = angle_difference(angle-self.angle) if self.n else 0.
        if self.n >= 2:
            features = np.array([self.velocity, action, 1.])
            innovation = speed-features@self.mean
            sigma = math.sqrt(.012**2+features@self.cov@features)
            if self.adapt and self.detect and len(self.rows)-self.start >= 6 and abs(innovation) > max(.04, 4*sigma):
                self.start = len(self.rows)
                self.changes += 1
            self.rows.append((features, speed))
            if self.adapt:
                used = self.rows[max(self.start, len(self.rows)-self.window):]
                x = np.array([r[0] for r in used])
                y = np.array([r[1] for r in used])
                prior_root = np.diag(np.sqrt([20., 200., 500.]))
                design = np.vstack([prior_root, x/.012])
                target = np.concatenate([prior_root@np.array([.85, .06, 0.]), y/.012])
                self.mean = np.linalg.lstsq(design, target, rcond=None)[0]
                _, r = np.linalg.qr(design, mode="reduced")
                inverse = np.linalg.solve(r, np.eye(3))
                self.cov = inverse@inverse.T
        self.angle, self.velocity, self.n = angle, speed, self.n+1


def independent_plan(fit, radius, goals, plan):
    mean, covariance = np.array(plan["mean"]), np.array(plan["covariance"])
    if plan["policy"] == "reactive":
        goal = math.atan2(goals[0, 1]-fit.center[1], goals[0, 0]-fit.center[0])
        desired = np.clip(.55*angle_difference(goal-fit.angle), -.35, .35)
        gain = mean[1] if abs(mean[1]) > .01 else np.copysign(.01, mean[1] or 1.)
        action = int(np.clip(np.round((desired-mean[0]*fit.velocity-mean[2])/gain), -1, 1))
        if plan["action"] != action or plan["particles"]:
            raise ValueError("Reactive decision differs")
        return 0.
    horizon = len(plan["actions"])
    candidates = np.array(list(itertools.product((-1, 0, 1), repeat=horizon)))
    factor = np.linalg.cholesky(covariance+1e-15*np.eye(3))
    parameters = np.array([mean]+[mean+2*factor[:, j] for j in range(3)]+[mean-2*factor[:, j] for j in range(3)])
    weights = np.array([.25]+[.125]*6)
    z = np.full((len(candidates), 7), cmath.exp(1j*fit.angle))
    speed = np.full((len(candidates), 7), fit.velocity)
    total = np.zeros_like(speed)
    positions = []
    for t in range(horizon):
        speed = np.clip(speed*parameters[:, 0]+candidates[:, t, None]*parameters[:, 1]+parameters[:, 2], -.65, .65)
        z *= np.exp(1j*speed)
        xy = fit.center+radius*np.stack((z.real, z.imag), -1)
        positions.append(xy)
        total += .9**t*((xy-goals[t])**2).sum(-1)
        total += .9**t*(.005*candidates[:, t, None]**2+.05*speed**2)
    average = (weights*total).sum(-1)
    scores = average+.15*np.sqrt(np.maximum(0, (weights*(total-average[:, None])**2).sum(-1)))
    selected = np.flatnonzero((candidates == np.array(plan["actions"])).all(-1))
    if len(selected) != 1 or plan["action"] != plan["actions"][0]:
        raise ValueError("Invalid selected action sequence")
    selected = int(selected[0])
    if scores[selected]-scores.min() > 1e-9:
        raise ValueError("Selected plan is not an optimal frozen candidate")
    points = np.stack(positions, axis=1)[selected]
    difference = float(np.max(np.abs(points-np.asarray(plan["particles"]))))
    if difference > 1e-10 or not np.allclose(scores, plan["scores"], atol=1e-10, rtol=1e-10):
        raise ValueError("Independent imagined trajectories or branch costs differ")
    return difference


def interval(values, seed):
    values = np.array(values)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(-1)
    return np.quantile(draws, [.0125, .9875]).tolist()


def run(directory):
    protocol = check_contract(directory)["payload"]
    config = protocol["config"]
    summary = read(directory/"summary.json")
    original, incorrect = parent(), load_before()[0]
    errors = {"physical_position": 0., "posterior_mean": 0., "conditional_trajectory": 0., "cost": 0.}
    checked = 0
    details = []
    for cell in summary["records"]:
        path = directory/"lifetimes"/(cell["identity"]+".json.gz")
        if sha(path) != cell["raw_sha256"] or sha(ROOT/cell["checkpoint"]) != cell["checkpoint_sha256"]:
            raise ValueError("Saved outcome/checkpoint changed")
        item = json.loads(gzip.decompress(path.read_bytes()))
        spec, rows, snapshot = item["spec"], item["rows"], item["snapshot"]
        if len(rows) != config["steps"] or len(snapshot["events"]) != config["steps"]+3:
            raise ValueError("Lost continuing history")
        base, _, _ = choose_owner(original, incorrect, item["arm"], config["window"])
        restored = ContinuingSession.restore(base.components["r1"], snapshot)
        if any(not torch.equal(value, restored.owner.state_dict()[name]) for name, value in base.components["r1"].state_dict().items()):
            raise ValueError("Preexisting tensors changed")
        center, radius = restored.owner.geometry()
        fit = IndependentFit(center, snapshot["window"], snapshot["adapt"], snapshot["detect_change"])
        for event in snapshot["events"][:3]:
            fit.update(event["values"], event["action"])
        predicted_errors, covered, measured_costs, post_change_costs = [], [], [], []
        prior_after = None
        for index, row in enumerate(rows):
            plan = row["plan"]
            if row["index"] != index or row["before_assessor"]["time"] != index+2:
                raise ValueError("Reset or nonsequential time detected")
            if prior_after is not None and row["before_assessor"] != prior_after:
                raise ValueError("Hidden world continuity broken")
            if plan["policy"] != "analytic":
                errors["posterior_mean"] = max(errors["posterior_mean"], float(np.max(np.abs(fit.mean-plan["mean"]))))
                if not np.allclose(fit.cov, plan["covariance"], atol=1e-10, rtol=1e-10):
                    raise ValueError("Independent posterior covariance differs")
            else:
                expected = [spec["rho"], spec["gain"]*(-1 if index+2 >= spec["change_step"] else 1), spec["drift"]]
                if expected != plan["mean"]:
                    raise ValueError("Privileged reference coefficient mismatch")
            errors["conditional_trajectory"] = max(errors["conditional_trajectory"], independent_plan(fit, radius, np.array(row["goals"]), plan))
            physical, states = independent_rollout(spec, row["before_assessor"], plan["actions"])
            errors["physical_position"] = max(errors["physical_position"], float(np.max(np.abs(physical-row["counterfactual_assessor_positions"]))))
            prior_after = row["after_assessor"]
            if any(abs(states[0][k]-prior_after[k]) > 1e-12 for k in states[0]):
                raise ValueError("Independent world transition differs")
            noise_rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], prior_after["time"], 19231, 0]))
            sensed = physical[0]+noise_rng.normal(0, spec["sensor_noise"], 2)
            if not np.allclose(sensed, row["observed"], atol=1e-12, rtol=0):
                raise ValueError("Sensor reading is not the paid noisy world observation")
            event = snapshot["events"][index+3]
            if event["values"] != row["observed"] or event["action"] != plan["action"] or event["kind"] != "observation":
                raise ValueError("Conditional result entered the factual channel")
            cost = float(np.square(physical[0]-row["goals"][0]).sum()+.005*plan["action"]**2+.05*states[0]["velocity"]**2)
            errors["cost"] = max(errors["cost"], abs(cost-row["cost"]))
            measured_costs.append(cost)
            if row["before_assessor"]["time"] >= spec["change_step"]:
                post_change_costs.append(cost)
            if plan["particles"]:
                particles = np.asarray(plan["particles"])
                weights = np.asarray(plan["weights"])
                mean = np.einsum("hpd,p->hd", particles, weights)
                variance = np.einsum("hpd,p->hd", (particles-mean[:, None])**2, weights)
                predicted_errors.append(np.mean((mean-physical)**2, axis=1).tolist())
                covered.append((np.abs(mean-physical) <= 1.96*np.sqrt(variance+.002**2)).all(-1).astype(int).tolist())
            fit.update(row["observed"], plan["action"])
            errors["posterior_mean"] = max(errors["posterior_mean"], float(np.max(np.abs(fit.mean-row["posterior_after_observation"]))))
            checked += 1
        if abs(np.mean(measured_costs)-cell["cost"]) > 1e-12 or abs(np.mean(post_change_costs)-cell["post_change_cost"]) > 1e-12:
            raise ValueError("Aggregate costs or post-change boundary differ from actual transitions")
        if fit.changes != snapshot["buffers"]["interaction_changes"]:
            raise ValueError("Change detector used unobserved information")
        work = snapshot["work"]
        priced = cell["cost"]+(.00001*work["conditional_transition_particles"]+.001*work["posterior_fits"]+.001*work["paid_sensor_measurements"])/len(rows)
        details.append({**cell, "priced_cost": priced, "changes": fit.changes,
                        "rollout_mse_by_horizon": np.mean(predicted_errors, axis=0).tolist() if predicted_errors else [],
                        "conditional_interval_coverage_by_horizon": np.mean(covered, axis=0).tolist() if covered else []})
    if max(errors.values()) > 1e-9:
        raise ValueError(f"Independent audit mismatch: {errors}")
    aggregates = {family: {arm: {metric: float(np.mean([r[metric] for r in details if r["family"] == family and r["arm"] == arm]))
                                for metric in ("cost", "post_change_cost", "priced_cost", "seconds", "changes")}
                          for arm in config["arms"]} for family in config["families"]}
    pairs = {}
    for i, arm in enumerate(("adaptive_reactive", "adaptive_one_step", "frozen_mpc", "incorrect_geometry", "full_history_mpc")):
        values = []
        for seed in config["families"]["gain_reversal"]:
            a = next(r for r in details if r["family"] == "gain_reversal" and r["seed"] == seed and r["arm"] == "adaptive_mpc")
            b = next(r for r in details if r["family"] == "gain_reversal" and r["seed"] == seed and r["arm"] == arm)
            values.append(b["cost"]-a["cost"])
        pairs[arm] = {"absolute_gains": values, "interval_97_5_percent": interval(values, 199910+i),
                      "relative_cost_reduction": float(np.mean(values))/aggregates["gain_reversal"][arm]["cost"]}
    nominal = aggregates["gain_reversal"]
    gate = {"both_physical_gains": all(pairs[a]["relative_cost_reduction"] >= .1 and pairs[a]["interval_97_5_percent"][0] > 0
                                        for a in ("adaptive_reactive", "adaptive_one_step")),
            "post_change_correction": 1-nominal["adaptive_mpc"]["post_change_cost"]/nominal["frozen_mpc"]["post_change_cost"] >= .1,
            "priced_gain": 1-nominal["adaptive_mpc"]["priced_cost"]/nominal["adaptive_reactive"]["priced_cost"] >= .1,
            "replay_and_retention": all(r["exact_reload"] and r["old_tensors_unchanged"] for r in details)}
    gate["component_pass"] = all(gate.values())
    result = {"status": "PASS", "lifetimes": len(details), "independent_action_checks": checked,
              "maximum_differences": errors, "aggregate": aggregates, "paired_nominal_comparisons": pairs,
              "gate": gate, "records": details,
              "scope": "Conditional component with common interaction/decision caps; actual computation differs and is separately priced. No statistical answer certificate, learned eta, or neural adaptation."}
    write(directory/"independent-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    run(RELEASE/parser.parse_args().name)
