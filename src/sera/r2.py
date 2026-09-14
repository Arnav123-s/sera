"""Controlled event instruments guide executable program proposals and learn from their traces."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.contracts import EvidenceKind, Provenance
from sera.data import seed_for
from sera.experience import EvidenceReplay, Trajectory
from sera.programs import BudgetExhausted
from sera.storage import digest


class ControlledInstrument(nn.Module):
    """A learned CPTP channel per action and a shared projective observation instrument.

    The learned measurement basis supplies the same effects for conditioning and
    generation. Multiple action Kraus operators allow noninvertible dynamics.
    """

    def __init__(self, dimension=4, rank=4, complex_valued=True):
        super().__init__()
        if dimension != 4 or not 1 <= rank <= 8:
            raise ValueError("The color-world instrument needs dimension four and bounded rank")
        self.dimension, self.rank, self.complex_valued = dimension, rank, complex_valued
        dtype = torch.complex64 if complex_valued else torch.float32
        self.action_raw = nn.Parameter(torch.randn(4, rank * dimension, dimension, dtype=dtype))
        self.basis_raw = nn.Parameter(torch.eye(dimension, dtype=dtype)
                                     + 0.02 * torch.randn(dimension, dimension, dtype=dtype))
        self.proposal = nn.Sequential(nn.Linear(36, 48), nn.Tanh(), nn.Linear(48, 4))

    def export_config(self):
        return {"type": "r2", "dimension": self.dimension, "rank": self.rank,
                "complex_valued": self.complex_valued}

    def operators(self):
        action_q, action_r = torch.linalg.qr(self.action_raw, mode="reduced")
        basis, basis_r = torch.linalg.qr(self.basis_raw)
        if min(float(action_r.diagonal(dim1=-2, dim2=-1).abs().min().detach()),
               float(basis_r.diagonal().abs().min().detach())) < 1e-7:
            raise ValueError("Rank-deficient controlled instrument")
        columns = basis.T
        effects = columns[:, :, None] * columns.conj()[:, None, :]
        return action_q.reshape(4, self.rank, 4, 4), effects

    def initial(self, batch):
        return torch.eye(4, dtype=self.action_raw.dtype).expand(batch, -1, -1).clone() / 4

    def control(self, rho, actions, operators):
        kraus = operators[0][actions]
        return (kraus @ rho[:, None] @ kraus.mH).sum(1)

    def branches(self, rho, operators):
        effects = operators[1]
        return effects[None] @ rho[:, None] @ effects.mH[None]

    def probabilities(self, rho, operators):
        return self.branches(rho, operators).diagonal(dim1=-2, dim2=-1).sum(-1).real

    def observe(self, rho, colors, operators):
        branches = self.branches(rho, operators)
        selected = branches[torch.arange(len(colors)), colors.clamp_min(0)]
        probability = selected.diagonal(dim1=-2, dim2=-1).sum(-1).real
        visible = colors >= 0
        if torch.any((probability <= 0) & visible):
            raise ValueError("Impossible teacher-forced observation")
        normalized = selected / probability.clamp_min(1e-20)[:, None, None]
        return torch.where(visible[:, None, None], normalized, branches.sum(1)), probability

    def proposal_logits(self, rho, goals):
        imaginary = rho.imag if rho.is_complex() else torch.zeros_like(rho)
        features = torch.cat((rho.real.flatten(1), imaginary.flatten(1), F.one_hot(goals, 4)), -1)
        return self.proposal(features)

    def sequence(self, observations, actions, goals):
        operators = self.operators()
        rho, _ = self.observe(self.initial(len(observations)), observations[:, 0], operators)
        likelihoods, proposals = [], []
        for t in range(actions.shape[1]):
            proposals.append(self.proposal_logits(rho, goals))
            rho = self.control(rho, actions[:, t], operators)
            likelihoods.append(self.probabilities(rho, operators))
            rho, _ = self.observe(rho, observations[:, t + 1], operators)
        return torch.stack(likelihoods, 1), torch.stack(proposals, 1), rho

    @torch.no_grad()
    def validity(self):
        actions, effects = self.operators()
        identity = torch.eye(4, dtype=actions.dtype)
        return {"channel_completeness": float(((actions.mH @ actions).sum(1) - identity).abs().max()),
                "instrument_completeness": float((effects.sum(0) - identity).abs().max())}


def fit_instrument(model, evidence: EvidenceReplay, *, plans=(), steps=300, seed=0, work=None):
    if not isinstance(evidence, EvidenceReplay) or not evidence.records or steps < 1:
        raise ValueError("Instrument learning requires admitted trajectories")
    # Program credit obeys the same provenance and held-out split contract as likelihood data.
    # Validate without deduplicating or changing the original sampling distribution.
    EvidenceReplay(plans)
    rng = np.random.default_rng(seed_for("r2-update-order", seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    history = []
    for _ in range(steps):
        records = [evidence.records[i] for i in rng.integers(len(evidence.records), size=32)]
        observations = torch.tensor([r.observations for r in records])
        actions = torch.tensor([r.actions for r in records])
        goals = torch.tensor([r.goal for r in records])
        likelihoods, _, _ = model.sequence(observations, actions, goals)
        visible = observations[:, 1:] >= 0
        selected = likelihoods.gather(-1, observations[:, 1:].clamp_min(0)[..., None]).squeeze(-1)
        loss = -selected[visible].clamp_min(1e-12).log().mean()
        if plans:
            plan_losses = []
            for i in rng.integers(len(plans), size=min(8, len(plans))):
                trace = plans[i]
                if trace.provenance.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}:
                    raise ValueError("Program credit requires verified execution")
                _, proposed, _ = model.sequence(torch.tensor([trace.observations]),
                                                torch.tensor([trace.actions]),
                                                torch.tensor([trace.goal]))
                plan_losses.append(F.cross_entropy(proposed.flatten(0, 1),
                                                    torch.tensor(trace.actions)))
            loss = loss + torch.stack(plan_losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite instrument loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
        if work is not None:
            work.add("instrument_optimizer_steps")
            work.add("instrument_transition_draws", actions.numel())
            work.add("verified_program_credit_draws", min(8, len(plans)))
    model.eval()
    return {"steps": steps, "initial_batch_loss": history[0], "final_batch_loss": history[-1],
            "history": history, "validity": model.validity()}


def flatten_program(body, library=None, *, fuel=64, depth=0, stack=()):
    """Typed action/sequence/repeat/call grammar. Every expansion consumes bounded fuel."""
    library = {} if library is None else library
    if fuel < 0 or depth > 8 or not isinstance(body, dict):
        raise BudgetExhausted("Program depth or fuel exhausted")
    op = body.get("op")
    if op == "action":
        action = body.get("value")
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError("Action instruction outside the alphabet")
        if fuel < 1:
            raise BudgetExhausted("Program action fuel exhausted")
        return (action,)
    if op == "call":
        key = body.get("skill")
        if key not in library or key in stack:
            raise ValueError("Unknown or recursive program dependency")
        return flatten_program(library[key]["body"], library, fuel=fuel, depth=depth + 1,
                               stack=stack + (key,))
    if op == "repeat":
        count = body.get("count")
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError("Repeat count outside the grammar")
        items = [body["body"]] * count
    elif op == "sequence":
        items = body.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 16:
            raise ValueError("Invalid sequence body")
    else:
        raise ValueError("Unknown program opcode")
    actions = ()
    for item in items:
        actions += flatten_program(item, library, fuel=fuel - len(actions), depth=depth + 1,
                                   stack=stack)
    return actions


def program_body(actions):
    return {"op": "sequence", "items": [{"op": "action", "value": int(a)} for a in actions]}


@torch.no_grad()
def ranked_programs(model, start, goal, *, max_length=3, library=None, max_candidates=256, work=None):
    if (type(max_length) is not int or not 1 <= max_length <= 5
            or type(max_candidates) is not int or not 1 <= max_candidates <= 4096):
        raise ValueError("Invalid program proposal budget")
    library = {} if library is None else library
    units = [((a,), {"op": "action", "value": a}) for a in range(4)]
    units += [(flatten_program(record["body"], library), {"op": "call", "skill": key})
              for key, record in sorted(library.items()) if len(record["actions"]) > 1]
    operators = model.operators()
    if work is not None:
        work.add("instrument_operator_factorizations")
    initial, _ = model.observe(model.initial(1), torch.tensor([start]), operators)
    candidates, seen = [], set()
    for length in range(1, max_length + 1):
        for selection in itertools.product(range(len(units)), repeat=length):
            actions = sum((units[i][0] for i in selection), ())
            if actions in seen or len(actions) > 12:
                continue
            seen.add(actions)
            rho, logp = initial, 0.0
            for action in actions:
                logp += float(model.proposal_logits(rho, torch.tensor([goal])).log_softmax(-1)[0, action])
                rho = model.control(rho, torch.tensor([action]), operators)
                # Unobserved events remain marginalized; no fictitious measurement label.
                rho = model.branches(rho, operators).sum(1)
                if work is not None:
                    work.add("program_proposal_model_transitions")
            goal_probability = float(model.probabilities(rho, operators)[0, goal])
            value = logp / len(actions) + math.log(max(goal_probability, 1e-12)) - 0.04 * len(actions)
            body = {"op": "sequence", "items": [units[i][1] for i in selection]}
            candidates.append((value, actions, body))
            if work is not None:
                work.add("program_candidates_constructed")
            if len(candidates) >= max_candidates:
                return sorted(candidates, key=lambda row: row[0], reverse=True)
    return sorted(candidates, key=lambda row: row[0], reverse=True)


def search_program(spec, start, goal, *, model=None, library=None, budget=8, max_length=3,
                   work=None, split="program-support"):
    if not spec.resettable:
        raise ValueError("This program search requires declared reset access")
    if type(budget) is not int or not 1 <= budget <= 4096:
        raise ValueError("Search budget must be an integer in [1, 4096]")
    if type(max_length) is not int or not 1 <= max_length <= 5:
        raise ValueError("Program length budget must be an integer in [1, 5]")
    library = {} if library is None else library
    if model is None:
        action_sequences = itertools.chain.from_iterable(
            itertools.product(range(4), repeat=length) for length in range(1, max_length + 1))
        candidates = [(0.0, actions, program_body(actions))
                      for actions in itertools.islice(action_sequences, budget)]
    else:
        candidates = ranked_programs(model, start, goal, max_length=max_length, library=library, work=work)
    if model is None and work is not None:
        work.add("program_candidates_constructed", len(candidates))
    traces = []
    for attempt, (_, actions, body) in enumerate(candidates[:budget], 1):
        observations = spec.execute(start, actions)
        trace = Trajectory(spec.identifier, observations, tuple(actions),
                           tuple(float(o == goal) for o in observations[1:]), goal,
                           Provenance(f"execution:{spec.identifier}", digest([start, goal, actions]),
                                      EvidenceKind.VERIFIED), split)
        traces.append(trace)
        if work is not None:
            work.add("program_environment_resets")
            work.add("program_environment_actions", len(actions))
            work.add("program_candidates_executed")
        if observations[-1] != goal:
            continue
        # Validate the expanded typed body against a separately executed trace.
        expanded = flatten_program(body, library)
        verification = spec.execute(start, expanded)
        if work is not None:
            work.add("program_verification_resets")
            work.add("program_verification_actions", len(expanded))
        if expanded != actions or verification[-1] != goal:
            raise ValueError("Program verification failed")
        record = {"body": body, "actions": list(actions), "world_id": spec.identifier,
                  "start": start, "goal": goal, "signature": "color + action_program -> color",
                  "dependencies": sorted({node["skill"] for node in body["items"]
                                          if node["op"] == "call"}),
                  "evidence": [trace.identifier], "verification": digest(verification),
                  "scope": "Verified in a deterministic resettable world for this start/goal"}
        return {"success": True, "attempts": attempt, "record": record,
                "skill_id": digest(record), "traces": traces, "successful_trace": trace}
    return {"success": False, "attempts": min(budget, len(candidates)), "traces": traces}
