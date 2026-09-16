"""Assessor-owned simulator; the controller receives only sensors, actions and goals."""

import copy
import math

import numpy as np

from .core import SPEED_LIMIT

CENTER = np.array([.25, -.4])
RADIUS = math.sqrt(.8**2+.3**2)


def specification(seed, family, steps):
    if family not in ("gain_reversal", "omitted_torque"):
        raise ValueError("Unknown assessor family")
    rng = np.random.default_rng(seed)
    rho, gain, drift = float(rng.uniform(.72, .9)), float(rng.uniform(.055, .095)), float(rng.uniform(-.02, .02))
    angles = rng.uniform(-math.pi, math.pi, math.ceil((steps+8)/12))
    return {"seed": seed, "family": family, "steps": steps, "angle": float(rng.uniform(-math.pi, math.pi)),
            "velocity": float(rng.uniform(-.18, .18)), "rho": rho, "gain": gain, "drift": drift,
            "change_step": steps//2, "sensor_noise": .002,
            "goals": (CENTER+RADIUS*np.stack((np.cos(angles), np.sin(angles)), -1)).repeat(12, axis=0).tolist()}


class World:
    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        self.angle, self.velocity, self.time = spec["angle"], spec["velocity"], 0

    def coefficients(self):
        sign = -1. if self.time >= self.spec["change_step"] else 1.
        return np.array([self.spec["rho"], sign*self.spec["gain"], self.spec["drift"]])

    def position(self):
        return CENTER+RADIUS*np.array([math.cos(self.angle), math.sin(self.angle)])

    def measure(self, replicate=0):
        rng = np.random.default_rng(np.random.SeedSequence([self.spec["seed"], self.time, 19231, replicate]))
        return self.position()+rng.normal(0., self.spec["sensor_noise"], 2)

    def step(self, action):
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError("Unknown action; no reset or hidden-state query is permitted")
        rho, gain, drift = self.coefficients()
        torque = .04*math.sin(2*self.angle) if self.spec["family"] == "omitted_torque" else 0.
        self.velocity = float(np.clip(rho*self.velocity+gain*action+drift+torque, -SPEED_LIMIT, SPEED_LIMIT))
        self.angle += self.velocity
        self.time += 1
        return self.position()

    def goals(self, horizon):
        return np.asarray(self.spec["goals"][self.time:self.time+horizon])

    def assessment(self):
        return {"angle": self.angle, "velocity": self.velocity, "time": self.time}


def assess_plan(world, actions):
    """Counterfactual assessor diagnostics never enter factual state or learning."""
    alternative = copy.deepcopy(world)
    return np.array([alternative.step(int(action)) for action in actions])
