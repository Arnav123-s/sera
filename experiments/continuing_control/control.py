"""Fixed receding-horizon and strong non-rollout controls; eta is not learned."""

import itertools

import numpy as np

from sera.contracts import EvidenceKind

from .core import SPEED_LIMIT, wrap

HORIZON = 4
SEQUENCES = {h: np.array(list(itertools.product((-1, 0, 1), repeat=h)), dtype=np.float64) for h in (1, HORIZON)}


def decide(session, goals, policy, *, oracle_coefficients=None):
    if len(session.events) < 3:
        raise ValueError("Three paid observations are required before decisions")
    if policy not in ("mpc", "one_step", "reactive", "analytic"):
        raise ValueError("Unknown controller")
    goals = np.asarray(goals, dtype=np.float64)
    if goals.shape != (HORIZON, 2) or not np.isfinite(goals).all():
        raise ValueError("A finite public goal schedule is required")
    owner = session.owner
    mean = owner.interaction_mean.detach().numpy().copy()
    covariance = owner.interaction_covariance.detach().numpy().copy()
    if policy == "analytic":
        mean = np.asarray(oracle_coefficients, dtype=np.float64)
        if mean.shape != (3,) or not np.isfinite(mean).all():
            raise ValueError("Oracle control requires explicitly privileged coefficients")
        covariance = np.zeros((3, 3))
    elif oracle_coefficients is not None:
        raise ValueError("Privileged dynamics are forbidden for a learned controller")
    center, radius = owner.geometry()
    angle, velocity = float(owner.interaction_angle), float(owner.interaction_velocity)
    session.work["controller_decisions"] += 1
    if policy == "reactive":
        # A damped position controller with a learned input gain; no imagined transition.
        target = np.arctan2(goals[0, 1]-center[1], goals[0, 0]-center[0])
        desired_velocity = float(np.clip(.55*wrap(target-angle), -.35, .35))
        gain = mean[1] if abs(mean[1]) > .01 else np.copysign(.01, mean[1] or 1.)
        raw = (desired_velocity-mean[0]*velocity-mean[2])/gain
        action = int(np.clip(np.round(raw), -1, 1))
        return {"kind": EvidenceKind.PREDICTION.value, "state_token": session.token(), "action": action,
                "actions": [action], "particles": [], "weights": [], "scores": [],
                "policy": policy, "mean": mean.tolist(), "covariance": covariance.tolist()}
    horizon = 1 if policy == "one_step" else HORIZON
    sequences = SEQUENCES[horizon]
    # Positive-weight moment-matching quadrature; conditional covariance, not a safety certificate.
    chol = np.linalg.cholesky(covariance+np.eye(3)*1e-15)
    parameters = np.vstack([mean, *(mean+2*chol[:, j] for j in range(3)), *(mean-2*chol[:, j] for j in range(3))])
    weights = np.array([.25]+[.125]*6)
    angles = np.full((len(sequences), 7), angle)
    velocities = np.full_like(angles, velocity)
    costs = np.zeros_like(angles)
    positions = []
    for t in range(horizon):
        velocities = np.clip(parameters[:, 0]*velocities+parameters[:, 1]*sequences[:, t, None]+parameters[:, 2],
                             -SPEED_LIMIT, SPEED_LIMIT)
        angles += velocities
        xy = center+radius*np.stack((np.cos(angles), np.sin(angles)), -1)
        positions.append(xy)
        costs += .9**t*(np.square(xy-goals[t]).sum(-1)+.005*sequences[:, t, None]**2+.05*velocities**2)
    average = costs@weights
    deviation = np.sqrt(np.maximum(0., (costs-average[:, None])**2@weights))
    scores = average+.15*deviation
    choice = int(np.argmin(scores))
    session.work["conditional_transition_particles"] += int(len(sequences)*7*horizon)
    particles = np.stack(positions, axis=1)[choice]
    return {"kind": EvidenceKind.PREDICTION.value, "state_token": session.token(),
            "action": int(sequences[choice, 0]), "actions": sequences[choice].astype(int).tolist(),
            "particles": particles.tolist(), "weights": weights.tolist(), "scores": scores.tolist(),
            "policy": policy, "mean": mean.tolist(), "covariance": covariance.tolist()}
