"""Self-identified empirical gaps, generic program search and one continuing owner."""

import copy
import itertools
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.discovery_observation import ObservationSession, ObservingR1, point
from experiments.self_chosen.equations import learned_map
from experiments.verified_completion.common import ROOT, digest, model_hash, read, sha, write
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity

OUT = ROOT / "research-continuation/43_knowledge_gaps"
RUN = ROOT / "runs/KG-study-001"
PARENT = "runs/sera-observed-discovery-live"
EXPECTED_OWNER = "91f6c21ffcf80e8aab05ca8eea8e34499a27803cd823aa5a087839ac44957230"
ARMS = ("learned", "residual", "balanced")
STEPS = 4096


def contracts():
    return {"runtime": sha(Path(__file__)), "assessor": sha(Path(__file__).with_name("gap_assessor.py")),
            "protocol": sha(OUT / "PROTOCOL.md")}


def inventory(observation):
    """Read actual retained state; no externally selected target or answer."""
    discovery = observation.base
    items = []
    for record in discovery.records:
        numerical = "coefficients" in record["proposal"]
        items.append({"id": record["id"], "kind": "conditional_equation" if numerical else "language_association",
                      "domain": record["question"]["domain"], "target": record["proposal"]["target"],
                      "priority": 0., "reason": "Retained scoped evidence; no new contradictory observation in this record"})
    for key, record in observation.evidence.items():
        ev = record["evaluation"]
        consumed = sorted(set(ev["calibration_rows"] + ev["held_out_rows"]))
        unexplored = ev["source_rows"] - len(consumed)
        items.append({"id": key, "kind": "observed_source", "source": record["source"],
                      "source_hash": record["source_hash"], "target": ev["used_columns"][1],
                      "input": ev["used_columns"][0], "consumed": consumed,
                      "priority": unexplored / ev["source_rows"], "unexplored": unexplored,
                      "reason": "The retained prediction covers only part of its own recorded observation source"})
    eligible = [r for r in items if r["priority"] > 0]
    if not eligible:
        return items, None
    return items, max(eligible, key=lambda r: (r["priority"], r["id"]))


def inherited_basis(owner):
    """Powers are obtained by repeatedly executing its learned integral weights."""
    from fractions import Fraction

    coefficients, current = [[1.]], [Fraction(1)]
    for _ in range(4):
        current = learned_map(owner, "integral", current)
        while current and current[-1] == 0:
            current.pop()
        coefficients.append(list(map(float, current)))
    return coefficients


def normalize_program(program):
    kind, degree = program["kind"], int(program["degree"])
    cuts = sorted(set(float(v) for v in program["cuts"]))
    if kind not in {"segments", "continuous", "smooth"} or not 0 <= degree <= 4 or len(cuts) > 5:
        raise ValueError("Program exceeds the declared retained polynomial/conditional grammar")
    if kind == "continuous" and degree == 0 or kind == "smooth" and degree < 2:
        raise ValueError("Continuity condition needs a compatible degree")
    # No boundary means the same global polynomial, whatever its alias.
    return {"kind": kind if cuts else "segments", "degree": degree, "cuts": cuts}


def design(program, times, basis, center, scale):
    p = normalize_program(program)
    u = (np.asarray(times, dtype=float) - center) / scale
    cuts = (np.array(p["cuts"]) - center) / scale
    def powers(z):
        return np.column_stack([np.polynomial.polynomial.polyval(z, c) for c in basis[:p["degree"] + 1]])
    if not len(cuts):
        return powers(u)
    if p["kind"] == "segments":
        columns = []
        intervals = np.searchsorted(cuts, u, side="right")
        for j in range(len(cuts) + 1):
            local = u - (cuts[j - 1] if j else 0.)
            columns.append(powers(local) * (intervals == j)[:, None])
        return np.concatenate(columns, axis=1)
    columns = [powers(u)]
    start = 1 if p["kind"] == "continuous" else 2
    for cut in cuts:
        local = np.maximum(u - cut, 0.)
        columns.append(powers(local)[:, start:])
    return np.concatenate(columns, axis=1)


def predict(candidate, times, weights=None):
    x = design(candidate["program"], times, candidate["basis"], candidate["center"], candidate["scale"])
    return x @ np.array(candidate["weights"] if weights is None else weights)


def fit(program, public, basis):
    program = normalize_program(program)
    t, y = np.array(public["time"]), np.array(public["value"])
    center, scale = float(np.mean(t)), float(np.ptp(t))
    if scale <= 0:
        raise ValueError("At least two distinct independent coordinates are needed")
    matrix = design(program, t, basis, center, scale)
    if matrix.shape[1] + 3 > len(t):
        return {"status": "REJECTED", "reason": "insufficient observations for the program", "program": program}
    if program["kind"] == "segments":
        counts = np.bincount(np.searchsorted(program["cuts"], t, side="right"), minlength=len(program["cuts"]) + 1)
        if min(counts) < program["degree"] + 2:
            return {"status": "REJECTED", "reason": "unanchored conditional branch", "program": program}
    weights, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=1e-11)
    if rank != matrix.shape[1]:
        return {"status": "REJECTED", "reason": "unidentifiable program coefficients", "program": program}
    cv = np.zeros(len(t))
    for fold in range(3):
        mask = np.arange(len(t)) % 3 == fold
        w, _, rank, _ = np.linalg.lstsq(matrix[~mask], y[~mask], rcond=1e-11)
        if rank != matrix.shape[1]:
            return {"status": "REJECTED", "reason": "unidentifiable development fold", "program": program}
        cv[mask] = matrix[mask] @ w
    result = {"status": "PROPOSED", "program": program, "basis": basis, "center": center, "scale": scale,
              "weights": weights.tolist(), "fit_predictions": (matrix @ weights).tolist(),
              "development_predictions": cv.tolist(), "parameters": len(weights), "linear_solves": 4,
              "source": public["source"], "goal": public["goal"], "input_ids": public["ids"]}
    result["id"] = digest(result)
    return result


def program_features(program):
    return [float(program["kind"] == k) for k in ("segments", "continuous", "smooth")] + [
        program["degree"] / 4., len(program["cuts"]) / 5.,
        program["degree"] * len(program["cuts"]) / 20., float(not program["cuts"]), 1.]


class GapR1(ObservingR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not ObservingR1:
            raise ValueError("Continue the actual observation owner")
        owner.__class__ = cls
        owner.gap_policy = nn.Linear(8, 1, dtype=torch.float64)
        nn.init.zeros_(owner.gap_policy.weight)
        nn.init.zeros_(owner.gap_policy.bias)
        owner.gap_models = nn.ParameterDict()
        owner.gap_contract = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "gap_inquiry": self.gap_contract,
                "gap_model_keys": sorted(self.gap_models)}


class GapSession:
    def __init__(self, parent, saved=None):
        self.parent = parent
        self.base = ObservationSession(parent["parent"], parent["evidence"])
        if model_identity(self.base.owner) != parent["owner"]:
            raise ValueError("Changed parent owner")
        self.owner = GapR1.attach(self.base.owner)
        self.models, self.credits, self.open_goals = {}, [], []
        self.optimizer = torch.optim.Adam(self.owner.gap_policy.parameters(), lr=.003)
        if saved is not None:
            if saved["contracts"] != contracts() or saved["parent_owner"] != parent["owner"]:
                raise ValueError("Changed gap state lineage")
            self.owner.gap_policy.load_state_dict(decode(saved["policy"]))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            self.models, self.credits, self.open_goals = copy.deepcopy((saved["models"], saved["credits"], saved["open_goals"]))
            for key, candidate in self.models.items():
                self.owner.gap_models[key] = nn.Parameter(torch.tensor(candidate["weights"], dtype=torch.float64), requires_grad=False)
            if self.identity() != saved["owner"]:
                raise ValueError("Gap owner did not restore exactly")

    def identity(self):
        return model_identity(self.owner)

    def state(self):
        return {"schema": "sera.knowledge-gaps.1", "parent_owner": self.parent["owner"], "contracts": contracts(),
                "policy": encode(self.owner.gap_policy.state_dict()), "optimizer": encode(self.optimizer.state_dict()),
                "models": copy.deepcopy(self.models), "credits": copy.deepcopy(self.credits),
                "open_goals": copy.deepcopy(self.open_goals), "owner": self.identity()}

    def improve(self, program, score, reference):
        feature = torch.tensor(program_features(program), dtype=torch.float64)
        target = -math.log1p(score / max(reference, 1e-12))
        loss = (self.owner.gap_policy(feature).squeeze() - target) ** 2
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.owner.gap_policy.parameters(), 2.)
        self.optimizer.step()

    def retain(self, candidate, receipt, final):
        from experiments.gap_assessor import validate_receipt

        validate_receipt(candidate, receipt)
        if not final["accepted"] or final["candidate"] != candidate["id"]:
            raise ValueError("Final independent qualification is required")
        if any(c["candidate"] == receipt["candidate"] or c["identity"] == receipt["identity"] for c in self.credits):
            raise ValueError("Repeated credit")
        if any(set(c["credited_ids"]) & set(receipt["credited_ids"]) for c in self.credits):
            raise ValueError("This evidence already received novelty credit")
        self.models[candidate["id"]] = copy.deepcopy(candidate)
        self.owner.gap_models[candidate["id"]] = nn.Parameter(torch.tensor(candidate["weights"], dtype=torch.float64), requires_grad=False)
        self.credits.append(copy.deepcopy(receipt))


def random_program(rng, knots):
    kind = str(rng.choice(["segments", "continuous", "smooth"]))
    degree = int(rng.integers(0 if kind == "segments" else 1 if kind == "continuous" else 2, 5))
    count = int(rng.integers(0, min(5, len(knots)) + 1))
    return normalize_program({"kind": kind, "degree": degree,
                              "cuts": rng.choice(knots, count, replace=False).tolist()})


def mutate(rng, program, knots):
    p = copy.deepcopy(program)
    operation = int(rng.integers(0, 5))
    if operation == 0 and len(p["cuts"]) < 5:
        p["cuts"].append(float(rng.choice(knots)))
    elif operation == 1 and p["cuts"]:
        p["cuts"].pop(int(rng.integers(len(p["cuts"]))))
    elif operation == 2 and p["cuts"]:
        p["cuts"][int(rng.integers(len(p["cuts"])))] = float(rng.choice(knots))
    elif operation == 3:
        p["degree"] = int(rng.integers(0, 5))
    else:
        p["kind"] = str(rng.choice(["segments", "continuous", "smooth"]))
    try:
        return normalize_program(p)
    except ValueError:
        return random_program(rng, knots)


def portfolio(candidates, public, maximum=8):
    ordered = sorted((c for c in candidates if c["status"] == "PROPOSED"), key=lambda c: (c["assessment"]["score"], c["parameters"], c["id"]))
    chosen = []
    for candidate in ordered:
        if not chosen:
            chosen.append(candidate)
            continue
        similar = False
        for previous in chosen:
            delta = np.sqrt(np.mean((np.array(candidate["development_predictions"]) - previous["development_predictions"]) ** 2))
            same_assumptions = (candidate["program"]["kind"], candidate["program"]["degree"]) == (previous["program"]["kind"], previous["program"]["degree"])
            if same_assumptions and delta < .02 * max(float(np.std(public["value"])), 1e-9):
                similar = True
        if not similar:
            chosen.append(candidate)
        if len(chosen) == maximum:
            break
    return chosen


def search(session, public, arm, directory, limit=STEPS):
    from experiments.gap_assessor import assess_development

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"
    basis = inherited_basis(session.owner)
    times = np.array(public["time"])
    knots = ((times[:-1] + times[1:]) / 2).tolist()[2:-2]
    rng = np.random.default_rng(4301)
    events, seen, attempts, best = [], set(), 0, []
    reference = float(np.var(public["value"]))
    if checkpoint.exists():
        state = read(checkpoint)
        if state["contracts"] != contracts() or state["public"] != digest(public) or state["arm"] != arm:
            raise ValueError("Changed ongoing investigation")
        rng.bit_generator.state = state["rng"]
        events, attempts = state["events"], state["attempts"]
        seen = {digest(e["program"]) for e in events}
        session.owner.gap_policy.load_state_dict(decode(state["policy"]))
        session.optimizer.load_state_dict(decode(state["optimizer"]))
        best = sorted([e for e in events if e["status"] == "PROPOSED"], key=lambda e: e["assessment"]["score"])[:64]

    def save(done=False):
        write(checkpoint, {"contracts": contracts(), "public": digest(public), "arm": arm,
                           "rng": rng.bit_generator.state, "events": events, "attempts": attempts,
                           "policy": encode(session.owner.gap_policy.state_dict()),
                           "optimizer": encode(session.optimizer.state_dict()), "done": done})

    while len(events) < limit and attempts < 131072:
        attempts += 1
        if not best or arm == "balanced" or rng.random() < .25:
            program = random_program(rng, knots)
        else:
            parent = best[int(rng.integers(min(16, len(best))))]["program"]
            options = [mutate(rng, parent, knots) for _ in range(8)]
            if arm == "learned":
                with torch.no_grad():
                    scores = session.owner.gap_policy(torch.tensor([program_features(p) for p in options], dtype=torch.float64)).squeeze(-1).numpy()
                program = options[int(np.argmax(scores))]
            else:
                program = options[0]
        identity = digest(program)
        if identity in seen:
            continue
        seen.add(identity)
        event = fit(program, public, basis)
        event["sequence"] = len(events)
        # Durable proposal precedes assessor output. This is a single pending slot;
        # interrupted uncommitted work resumes from the preceding checkpoint.
        write(directory / "pending.json", event)
        if event["status"] == "PROPOSED":
            event["assessment"] = assess_development(event, public)
            if arm == "learned":
                session.improve(program, event["assessment"]["score"], reference)
            best = sorted(best + [event], key=lambda e: (e["assessment"]["score"], e["id"]))[:64]
        events.append(event)
        if len(events) % 128 == 0:
            save()
            print(f"{arm}: {len(events)} distinct programs; best development score {best[0]['assessment']['score'] if best else None}", flush=True)
    save(done=True)
    selected = portfolio(events, public)
    result = {"arm": arm, "candidates": len(events), "attempts": attempts, "duplicates": attempts - len(events),
              "fitted": sum(e["status"] == "PROPOSED" for e in events), "portfolio": selected,
              "policy": encode(session.owner.gap_policy.state_dict()), "optimizer": encode(session.optimizer.state_dict()),
              "ranker": model_hash(session.owner.gap_policy), "contracts": contracts()}
    write(directory / "result.json", result)
    return result


def unchanged_predictions(parent_session, gap, times):
    record = parent_session.evidence[gap["id"]]
    relation = next(r for r in parent_session.base.records if r["id"] == record["conjecture"])
    origin = record["evaluation"]["origin_seconds"]
    # At the initial coordinate the retained initial-position parameter is the
    # native applicable route; the inverse acceleration expression divides by t.
    return [record["weights"][0] if float(t) == origin else
            point(relation["proposal"], float(t) - origin, record["weights"]) for t in times]


def global_controls(public, basis):
    return [fit({"kind": "segments", "degree": d, "cuts": []}, public, basis) for d in range(5)]


def exhaustive_segmentation(public, basis):
    """Exact additive training-objective control over up to six segments."""
    times, y = np.array(public["time"]), np.array(public["value"])
    n, penalty = len(times), max(float(np.var(y)), 1e-12) * math.log(len(y)) / len(y)
    costs, best = {}, {(0, 0): (0., [])}
    for left, right in itertools.combinations(range(n + 1), 2):
        if right - left < 3:
            continue
        variants = []
        for d in range(min(3, right - left - 2) + 1):
            x = np.vander(times[left:right] - times[left], d + 1, increasing=True)
            w = np.linalg.lstsq(x, y[left:right], rcond=None)[0]
            variants.append((float(np.sum((x @ w - y[left:right]) ** 2)) + penalty * (d + 1), d, w.tolist()))
        costs[left, right] = min(variants)
    for segments in range(1, 7):
        for right in range(3, n + 1):
            variants = []
            for left in range(right):
                if (segments - 1, left) in best and (left, right) in costs:
                    prev, path = best[segments - 1, left]
                    cost, d, w = costs[left, right]
                    variants.append((prev + cost, path + [{"left": left, "right": right, "degree": d, "weights": w, "origin": float(times[left])}]))
            if variants:
                best[segments, right] = min(variants, key=lambda v: v[0])
    score, path = min((v for (k, end), v in best.items() if end == n and k), key=lambda v: v[0])
    cuts = [(times[a["right"] - 1] + times[b["left"]]) / 2 for a, b in zip(path, path[1:])]
    return {"kind": "dynamic_segmentation", "segments": path, "cuts": cuts, "objective": score, "penalty": penalty}


__all__ = ["ARMS", "EXPECTED_OWNER", "OUT", "PARENT", "ROOT", "RUN", "STEPS", "GapSession", "contracts",
           "digest", "read", "sha", "write"]
