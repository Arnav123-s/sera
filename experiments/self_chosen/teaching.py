"""Three supplied elementary examples teach an action policy, then transfer is tested."""

import torch
from torch.nn import functional as F

from .equations import evaluate, learned_map

ACTIONS = ("test", "diversify", "revise")
EXAMPLES = (
    {"id": "double", "description": "2 maps to 4 and 3 maps to 6. Propose twice the input; test 5 before credit.",
     "context": [1., 0., 0.], "action": "test", "independent_check": {"input": 5, "prediction": 10, "answer": 10}},
    {"id": "alias", "description": "Adding two and then one equals adding three. Renaming the same expression earns no new discovery; seek another useful route.",
     "context": [0., 1., 0.], "action": "diversify", "independent_check": {"same_canonical_expression": True, "new_discovery_credit": 0}},
    {"id": "exception", "description": "A sequence increased on two checks, then decreased. Keep the counterexample and revise the always-increasing claim.",
     "context": [0., 0., 1.], "action": "revise", "independent_check": {"observations": [1, 2, 3, 2], "always_increasing": False}},
)


def teach(owner):
    x = torch.tensor([r["context"] for r in EXAMPLES], dtype=torch.float64)
    y = torch.arange(3)
    optimizer = torch.optim.SGD(owner.discovery_action.parameters(), lr=.3)
    before = owner.discovery_action(x).detach().argmax(-1).tolist()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(owner.discovery_action(x), y)
        loss.backward()
        optimizer.step()
    # No extra teaching: new context magnitudes and distractor components.
    checks = []
    with torch.no_grad():
        for label in range(3):
            for magnitude in (.35, .6, 1.4, 2.):
                context = [.1, .05, .08]
                context[label] = magnitude
                chosen = int(owner.discovery_action(torch.tensor(context, dtype=torch.float64)).argmax())
                checks.append({"context": context, "expected": ACTIONS[label], "chosen": ACTIONS[chosen], "correct": chosen == label})
    return {"examples": list(EXAMPLES), "before": before, "after": owner.discovery_action(x).detach().argmax(-1).tolist(),
            "updates": 120, "transfer": checks, "transfer_correct": sum(c["correct"] for c in checks),
            "meaning": "learned selection of three investigation actions from supplied structured status; measured process transfer"}


def action(owner, status):
    context = [float(status == s) for s in ("unverified", "duplicate", "counterexample")]
    with torch.no_grad():
        return ACTIONS[int(owner.discovery_action(torch.tensor(context, dtype=torch.float64)).argmax())]


def run_examples(owner):
    doubled = evaluate(learned_map(owner, "sum", [2]), 5)
    left = evaluate(learned_map(owner, "integral", [2]), 7)+evaluate(learned_map(owner, "integral", [1]), 7)
    right = evaluate(learned_map(owner, "integral", [3]), 7)
    observations = EXAMPLES[2]["independent_check"]["observations"]
    failed_increase = any(b <= a for a, b in zip(observations, observations[1:]))
    if doubled != 5+5 or left != right or left != 21 or not failed_increase:
        raise AssertionError("The actual owner did not execute the elementary examples correctly")
    return {"double": {"actual_owner_result": str(doubled), "independent_addition": 5+5},
            "alias": {"two_integrals": str(left), "one_integral": str(right), "independent_addition": 21},
            "exception": {"counterexample_detected": failed_increase, "observations": observations}}
