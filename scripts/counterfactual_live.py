"""Qualified current interface after the independent completion-challenge repair."""

from fractions import Fraction as Q

from experiments.counterfactual_core import (
    EXPANSIONS,
    POWERS,
    forward,
    operators,
    primitives,
    propose,
    rational,
)
from experiments.gap_inquiry import ROOT, read
from scripts import counterfactual_use as original
from scripts.counterfactual_qualification import restore as restore_qualified

ORIGINAL_PERFORM = original.perform


def restore():
    audit = read(ROOT / "research-continuation/45_counterfactual_inquiry/audit.json")
    session, growth = restore_qualified()
    if not audit["passed"] or session.identity() != audit["owner"]:
        raise ValueError("Use the qualified challenged owner")
    return session, growth


def perform(session, growth, request):
    if request.get("kind") == "what_if" and "context" in request:
        frontier = read(ROOT / "runs/CI-qualified-001/frontier.json")
        axis, target = original.names(request, frontier)
        q = next(q for q in frontier["questions"] if q["domain"] == request["domain"] and q["axis"] == axis and q["target"] == target)
        context = {n: Q(str(v)) for n, v in request["context"].items()}
        if set(context) != set(q["context"]):
            raise ValueError("Provide the complete retained world context")
        expected = forward(operators(session.owner), q["domain"], {n: rational(context[n]) for n in primitives(q["domain"])})
        if any(Q(str(v)) != context[n] for n, v in expected.items()):
            raise ValueError("Conflicting baseline premises; preserve them for investigation rather than silently changing the world")
        if context["t"] <= 0 or "m" in context and context["m"] <= 0 or q["domain"] == "polynomials" and context["t"].denominator != 1:
            raise ValueError("The supplied baseline violates the retained domain guards")
    if request.get("kind") == "what_if":
        earlier_proposer = original.propose
        try:
            original.propose = lambda question, matrices: propose(question, matrices, (*POWERS, *EXPANSIONS))
            return ORIGINAL_PERFORM(session, growth, request)
        finally:
            original.propose = earlier_proposer
    return ORIGINAL_PERFORM(session, growth, request)


def main():
    original.restore, original.perform = restore, perform
    original.main()


if __name__ == "__main__":
    main()
