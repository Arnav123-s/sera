"""Matched finite controls: existing R1 features, existing R2 and classical fits."""

import copy

import numpy as np
import torch

from experiments.intervention_assess import independent_fit, nll, predict
from experiments.intervention_model import features
from experiments.structural_field import r1_features
from sera.r2 import ControlledInstrument


def group_data(rows):
    groups = [g for r in rows for g in r["groups"]]
    controls = [g["applied"] for g in groups]
    n = np.array([g["shots"] for g in groups], dtype=float)
    y = np.array([g["plus"]/g["shots"] for g in groups])
    return controls, n, y


def r2_probabilities(model, controls):
    operators = model.operators()
    psi = model.action_raw.new_tensor([2**-.5, 2**-.5, 0., 0.])
    rho = (psi[:, None] @ psi.conj()[None]).expand(len(controls), -1, -1).clone()
    sequences = []
    for control in controls:
        actions = []
        for tick in range(control["ticks"]):
            if tick in control["pulses"]:
                actions.append(1)
            actions.append(0)
        sequences.append(actions)
    for step in range(max(map(len, sequences))):
        actions = torch.tensor([s[step] if step < len(s) else 0 for s in sequences])
        mask = torch.tensor([step < len(s) for s in sequences])[:, None, None]
        evolved = model.control(rho, actions, operators)
        rho = torch.where(mask, evolved, rho)
    return model.probabilities(rho, operators)[:, 0]


def fit_r2(acquisition, selection, queries, seed):
    controls, n, y = group_data(acquisition)
    vc, vn, vy = group_data(selection)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ControlledInstrument(dimension=4, rank=2)
        # The preparation is supplied; both controlled channels and the final
        # instrument are learned. No intermediate measurement is inserted.
        with torch.no_grad():
            model.action_raw.mul_(.08)
            model.action_raw[:, :4] += torch.eye(4, dtype=model.action_raw.dtype)
        optimizer = torch.optim.Adam([model.action_raw, model.basis_raw], lr=.015)
        tn, ty = torch.tensor(n, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        tvn, tvy = torch.tensor(vn, dtype=torch.float32), torch.tensor(vy, dtype=torch.float32)
        best, best_state, history = float("inf"), None, []
        for step in range(180):
            p = r2_probabilities(model, controls).clamp(1e-7, 1-1e-7)
            loss = -(tn*(ty*p.log()+(1-ty)*torch.log1p(-p))).sum()/tn.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 19:
                with torch.no_grad():
                    vp = r2_probabilities(model, vc).clamp(1e-7, 1-1e-7)
                    score = float(-(tvn*(tvy*vp.log()+(1-tvy)*torch.log1p(-vp))).sum()/tvn.sum())
                history.append({"step": step+1, "selection_nll": score, "training_nll": float(loss.detach())})
                if score < best:
                    best, best_state = score, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        with torch.no_grad():
            prediction = r2_probabilities(model, queries).double().tolist()
        serialized = {k: {"dtype": str(v.dtype), "real": v.real.tolist(),
                          "imag": v.imag.tolist() if v.is_complex() else None} for k, v in best_state.items()}
        return {"prediction": prediction, "weights": serialized, "history": history,
                "optimizer_steps": 180, "kind": "existing ControlledInstrument architecture, freshly fitted matched control",
                "validity": model.validity()}


def controls(owner, acquisition, selection, queries, initial, seed):
    result = {"initial_passive": {"weights": initial, "prediction": predict(queries, initial).tolist()}}
    for kind in ("temporal", "pulse_loss"):
        w, report = independent_fit(acquisition, kind)
        result["classical_"+kind] = {"weights": w.tolist(), "prediction": predict(queries, w).tolist(), **report}
    choice = min(("temporal", "pulse_loss"), key=lambda k: nll(selection, result["classical_"+k]["weights"]))
    result["classical_selection"] = {"selected": choice, "prediction": result["classical_"+choice]["prediction"]}
    # A deliberately weaker command-trusting control isolates the value of the monitor.
    trusted = copy.deepcopy(acquisition)
    for row in trusted:
        row["groups"] = [{"applied": row["requested"], "shots": sum(g["shots"] for g in row["groups"]),
                          "plus": sum(g["plus"] for g in row["groups"])}]
    w, report = independent_fit(trusted, "pulse_loss")
    result["trust_command"] = {"weights": w.tolist(), "prediction": predict(queries, w).tolist(), **report}
    ac, an, ay = group_data(acquisition)
    vc, vn, vy = group_data(selection)

    def numeric(programs):
        return np.array([features(p)+[p["ticks"]/16, (p["pulses"][0]/p["ticks"] if p["pulses"] else 0), 1.] for p in programs])

    ax, vx, qx = numeric(ac), numeric(vc), numeric(queries)
    context = [{"x": x.tolist(), "y": [float(y), 1-float(y)]} for x, y in zip(ax, ay, strict=True)]
    fa, fv, fq = (r1_features(owner, context, x) for x in (ax, vx, qx))
    weighted = an/an.mean()
    best, selected_w, selected_ridge = float("inf"), None, None
    scores = {}
    for ridge in (.001, .01, .1):
        w = np.linalg.solve(fa.T@(weighted[:, None]*fa)+np.eye(fa.shape[1])*ridge, fa.T@(weighted*ay))
        yp = np.clip(fv@w, 1e-7, 1-1e-7)
        score = float(-np.sum(vn*(vy*np.log(yp)+(1-vy)*np.log1p(-yp)))/vn.sum())
        scores[str(ridge)] = score
        if score < best:
            best, selected_w, selected_ridge = score, w, ridge
    result["current_r1_readout"] = {"prediction": np.clip(fq@selected_w, 0, 1).tolist(),
                                     "new_readout_weights": selected_w.tolist(), "feature_dimensions": fa.shape[1],
                                     "ridge": selected_ridge, "selection_scores": scores,
                                     "context_groups": len(context), "core_parameters_changed": 0}
    result["current_r2_fitted"] = fit_r2(acquisition, selection, queries, seed)
    return result
