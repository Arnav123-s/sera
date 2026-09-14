"""Controlled event instruments guide executable program proposals and learn from their traces."""

from __future__ import annotations

import copy
import itertools
import math
import time

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
    """A learned CPTP channel per action and a shared event instrument.

    The learned measurement basis supplies the same effects for conditioning and
    generation. The general mode allows several Kraus operators per event;
    the historical rank-one projective mode remains a restricted control.
    """

    def __init__(self, dimension=4, rank=4, complex_valued=True, event_kind="projective",
                 event_rank=2, proposal_width=48):
        super().__init__()
        if (dimension not in {4, 8, 16} or not 1 <= rank <= 8 or not 1 <= event_rank <= 8
                or event_kind not in {"projective", "kraus"}
                or (event_kind == "projective" and dimension != 4)):
            raise ValueError("Invalid event-instrument dimension, mode or Kraus rank")
        self.dimension, self.rank, self.complex_valued = dimension, rank, complex_valued
        self.event_kind, self.event_rank, self.proposal_width = event_kind, event_rank, proposal_width
        dtype = torch.complex64 if complex_valued else torch.float32
        self.action_raw = nn.Parameter(torch.randn(4, rank * dimension, dimension, dtype=dtype))
        if event_kind == "projective":
            self.basis_raw = nn.Parameter(torch.eye(dimension, dtype=dtype)
                                         + 0.02 * torch.randn(dimension, dimension, dtype=dtype))
        else:
            self.event_raw = nn.Parameter(torch.randn(4 * event_rank * dimension, dimension, dtype=dtype))
        self.proposal = nn.Sequential(nn.Linear(2 * dimension**2 + 4, proposal_width), nn.Tanh(),
                                      nn.Linear(proposal_width, 4))

    def export_config(self):
        config = {"type": "r2", "dimension": self.dimension, "rank": self.rank,
                  "complex_valued": self.complex_valued}
        # Preserve the executable identity of historical projective checkpoints.
        if self.event_kind != "projective" or self.proposal_width != 48:
            config.update(event_kind=self.event_kind, event_rank=self.event_rank,
                          proposal_width=self.proposal_width)
        return config

    def operators(self):
        action_q, action_r = torch.linalg.qr(self.action_raw, mode="reduced")
        basis, basis_r = torch.linalg.qr(self.basis_raw if self.event_kind == "projective" else self.event_raw,
                                        mode="reduced")
        if min(float(action_r.diagonal(dim1=-2, dim2=-1).abs().min().detach()),
               float(basis_r.diagonal().abs().min().detach())) < 1e-7:
            raise ValueError("Rank-deficient controlled instrument")
        if self.event_kind == "projective":
            columns = basis.T
            effects = columns[:, :, None] * columns.conj()[:, None, :]
        else:
            effects = basis.reshape(4, self.event_rank, self.dimension, self.dimension)
        return action_q.reshape(4, self.rank, self.dimension, self.dimension), effects

    def initial(self, batch):
        return torch.eye(self.dimension, dtype=self.action_raw.dtype,
                         device=self.action_raw.device).expand(batch, -1, -1).clone() / self.dimension

    def control(self, rho, actions, operators):
        kraus = operators[0][actions]
        return (kraus @ rho[:, None] @ kraus.mH).sum(1)

    def branches(self, rho, operators):
        effects = operators[1]
        if self.event_kind == "kraus":
            return (effects[None] @ rho[:, None, None] @ effects.mH[None]).sum(2)
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
        identity = torch.eye(self.dimension, dtype=actions.dtype, device=actions.device)
        completeness = ((effects.mH @ effects).sum((0, 1))
                        if self.event_kind == "kraus" else effects.sum(0))
        return {"channel_completeness": float(((actions.mH @ actions).sum(1) - identity).abs().max()),
                "instrument_completeness": float((completeness - identity).abs().max())}


def fit_instrument(model, evidence: EvidenceReplay, *, plans=(), steps=300, seed=0, work=None,
                   max_seconds=None, learning_rate=.015, validation=None, validation_interval=100):
    if not isinstance(evidence, EvidenceReplay) or not evidence.records or steps < 1:
        raise ValueError("Instrument learning requires admitted trajectories")
    # Program credit obeys the same provenance and held-out split contract as likelihood data.
    # Validate without deduplicating or changing the original sampling distribution.
    EvidenceReplay(plans)
    if not math.isfinite(learning_rate) or learning_rate <= 0 or validation_interval < 1:
        raise ValueError("Invalid instrument optimizer configuration")
    if validation is not None:
        if not validation or any(not row.split.startswith("validation") for row in validation):
            raise ValueError("Checkpoint selection requires an explicit validation split")
        if evidence.identifiers.intersection(row.identifier for row in validation):
            raise ValueError("Instrument support and validation overlap")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Training time budget must be positive and finite")
    started = time.perf_counter()
    rng = np.random.default_rng(seed_for("r2-update-order", seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    checkpoints, best, best_state = [], float("inf"), None
    lengths = {}
    for record in evidence.records:
        lengths.setdefault(len(record.actions), []).append(record)
    for step in range(steps):
        if history and max_seconds is not None and time.perf_counter() - started >= max_seconds:
            break
        length = len(evidence.records[int(rng.integers(len(evidence.records)))].actions)
        pool = lengths[length]
        records = [pool[i] for i in rng.integers(len(pool), size=32)]
        observations = torch.tensor([r.observations for r in records])
        actions = torch.tensor([r.actions for r in records])
        goals = torch.tensor([r.goal for r in records])
        likelihoods, _, _ = model.sequence(observations, actions, goals)
        visible = observations[:, 1:] >= 0
        selected = likelihoods.gather(-1, observations[:, 1:].clamp_min(0)[..., None]).squeeze(-1)
        # An all-missing batch provides no likelihood labels. It may still
        # contain independently verified program credit; never average an empty set.
        loss = (-selected[visible].clamp_min(1e-12).log().mean() if visible.any()
                else likelihoods.sum() * 0)
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
            work.add("instrument_visible_target_draws", int(visible.sum()))
            work.add("verified_program_credit_draws", min(8, len(plans)))
        if validation is not None and (step % validation_interval == 0 or step == steps - 1):
            from sera.belief import score_predictor
            checked = score_predictor(model, validation, work=work)
            checkpoints.append({"step": step + 1, **checked})
            if checked["nll"] < best:
                best, best_state = checked["nll"], copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"steps": len(history), "requested_steps": steps, "max_seconds": max_seconds,
            "learning_rate": learning_rate, "validation_checkpoints": checkpoints,
            "selected_validation_nll": best if best_state is not None else None,
            "training_seconds": time.perf_counter() - started,
            "initial_batch_loss": history[0], "final_batch_loss": history[-1],
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


def program_dependencies(body):
    if body.get("op") == "call":
        return {body["skill"]}
    if body.get("op") == "repeat":
        return program_dependencies(body["body"])
    if body.get("op") == "sequence":
        return set().union(*(program_dependencies(item) for item in body["items"]))
    return set()


def validate_library(library, world_id):
    if len(library) > 256:
        raise ValueError("Program library exceeds the declared search budget")
    for record in library.values():
        if record.get("world_id") != world_id:
            raise ValueError("Program dependency belongs to another world domain")
        dependencies = program_dependencies(record["body"])
        if dependencies != set(record.get("dependencies", [])) or not dependencies.issubset(library):
            raise ValueError("Program dependency declarations do not match the body")
        if "dependency_versions" in record:
            if record["dependency_versions"] != {key: digest(library[key]) for key in sorted(dependencies)}:
                raise ValueError("Program dependency version changed")
        if tuple(record["actions"]) != flatten_program(record["body"], library):
            raise ValueError("Stored program expansion differs from its verified actions")


def program_units(library):
    # Equivalent action expansions share one proposal; the shorter callable vocabulary comes first.
    units, seen = [], set()
    for key, record in sorted(library.items(), key=lambda row: (len(row[1]["actions"]), row[0])):
        actions = flatten_program(record["body"], library)
        if 1 < len(actions) <= 12 and actions not in seen:
            units.append((actions, {"op": "call", "skill": key}))
            seen.add(actions)
    return units + [((action,), {"op": "action", "value": action}) for action in range(4)]


def enumerate_programs(library, max_length, max_candidates, *, work=None):
    units, seen, examined, produced = program_units(library), set(), 0, 0
    for length in range(1, max_length + 1):
        for selection in itertools.product(range(len(units)), repeat=length):
            examined += 1
            if work is not None:
                work.add("program_combinations_examined")
            actions = sum((units[index][0] for index in selection), ())
            if actions not in seen and len(actions) <= 12:
                body = {"op": "sequence", "items": [units[index][1] for index in selection]}
                try:
                    flatten_program(body, library)
                except BudgetExhausted:
                    if work is not None:
                        work.add("program_candidates_rejected_depth")
                    if examined >= 16384:
                        return
                    continue
                seen.add(actions)
                produced += 1
                yield actions, body
                if produced >= max_candidates:
                    return
            if examined >= 16384:
                return


@torch.no_grad()
def ranked_programs(model, start, goal, *, max_length=3, library=None, max_candidates=256, work=None):
    if (type(max_length) is not int or not 1 <= max_length <= 5
            or type(max_candidates) is not int or not 1 <= max_candidates <= 4096):
        raise ValueError("Invalid program proposal budget")
    library = {} if library is None else library
    operators = model.operators()
    if work is not None:
        work.add("instrument_operator_factorizations")
    initial, _ = model.observe(model.initial(1), torch.tensor([start]), operators)
    candidates = []
    for actions, body in enumerate_programs(library, max_length, max_candidates, work=work):
        rho, logp = initial, 0.0
        for action in actions:
            logp += float(model.proposal_logits(rho, torch.tensor([goal])).log_softmax(-1)[0, action])
            rho = model.control(rho, torch.tensor([action]), operators)
            goal_probability = float(model.probabilities(rho, operators)[0, goal])
            # Unobserved events remain marginalized; no fictitious measurement label.
            rho = model.branches(rho, operators).sum(1)
            if work is not None:
                work.add("program_proposal_model_transitions")
        value = logp / len(actions) + math.log(max(goal_probability, 1e-12)) - 0.04 * len(actions)
        candidates.append((value, actions, body))
        if work is not None:
            work.add("program_candidates_constructed")
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
    validate_library(library, spec.identifier)
    if model is None:
        candidates = [(0.0, actions, program_body(actions))
                      for actions, _ in enumerate_programs({}, max_length, budget, work=work)] if not library else [
                          (0.0, actions, body) for actions, body in
                          enumerate_programs(library, max_length, budget, work=work)]
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
        dependencies = sorted(program_dependencies(body))
        failure_cases = [{"start": start, "goal": goal, "actions": list(row.actions),
                          "observed_output": row.observations[-1], "evidence": row.identifier}
                         for row in traces if row.observations[-1] != goal]
        record = {"schema_version": 2, "body": body, "actions": list(actions), "world_id": spec.identifier,
                  "start": start, "goal": goal, "signature": "color + action_program -> color",
                  "dependencies": dependencies,
                  "dependency_versions": {key: digest(library[key]) for key in dependencies},
                  "input_domain": {"world_id": spec.identifier, "verified_start": start, "reset_required": True},
                  "examples": [{"input": start, "output": goal, "observations": list(observations)}],
                  "failure_cases": failure_cases,
                  "tests": [{"input": start, "expected": goal, "observed": verification[-1],
                             "trace_sha256": digest(verification)}],
                  "evidence": [trace.identifier], "verification": digest(verification),
                  "scope": "Verified in a deterministic resettable world for this start/goal"}
        return {"success": True, "attempts": attempt, "record": record,
                "skill_id": digest(record), "traces": traces, "successful_trace": trace}
    return {"success": False, "attempts": min(budget, len(candidates)), "traces": traces}


@torch.no_grad()
def search_transformation(spec, targets, *, library=None, model=None, budget=8,
                          action_budget=128, max_length=3, candidate_budget=128, work=None):
    """Execute a bounded program against four demonstrated input/output constraints.

    Held-out composition evaluation can request a whole transformation, rather
    than selecting an untrained start/goal pair from an otherwise known task.
    """
    if not spec.resettable or len(targets) != 4 or any(type(v) is not int or not 0 <= v < 4 for v in targets):
        raise ValueError("Transformation requests need four valid outputs and reset access")
    if (type(budget) is not int or not 1 <= budget <= 4096 or not 4 <= action_budget <= 65536
            or not 1 <= max_length <= 5 or not 1 <= candidate_budget <= 4096):
        raise ValueError("Invalid transformation search budget")
    library = {} if library is None else library
    validate_library(library, spec.identifier)
    candidates = list(enumerate_programs(library, max_length, candidate_budget, work=work))
    if work is not None:
        work.add("transformation_candidates_constructed", len(candidates))
    if model is not None:
        operators = model.operators()
        initial, _ = model.observe(model.initial(4), torch.arange(4), operators)
        scored = []
        for actions, body in candidates:
            rho = initial
            for action in actions:
                rho = model.control(rho, torch.full((4,), action), operators)
                probabilities = model.probabilities(rho, operators)
                rho = model.branches(rho, operators).sum(1)
                if work is not None:
                    work.add("transformation_model_transitions", 4)
            likelihood = probabilities[torch.arange(4), torch.tensor(targets)].clamp_min(1e-12).log().mean()
            scored.append((float(likelihood) - .01 * len(actions), actions, body))
        candidates = [(actions, body) for _, actions, body in sorted(scored, key=lambda row: row[0], reverse=True)]
    used, attempts = 0, []
    for actions, body in candidates[:budget]:
        if used + 4 * len(actions) > action_budget:
            break
        traces = [spec.execute(start, actions) for start in range(4)]
        used += 4 * len(actions)
        if work is not None:
            work.add("transformation_environment_actions", 4 * len(actions))
            work.add("transformation_environment_resets", 4)
        outputs = [row[-1] for row in traces]
        attempts.append({"actions": list(actions), "outputs": outputs, "traces": traces})
        if outputs == list(targets):
            return {"success": True, "body": body, "dependencies": sorted(program_dependencies(body)),
                    "attempts": attempts, "environment_actions": used, "targets": list(targets)}
    return {"success": False, "attempts": attempts, "environment_actions": used, "targets": list(targets)}
