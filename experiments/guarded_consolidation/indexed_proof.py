"""Same exhaustive proof, shared forward evaluations across right-hand sides.

For each a,b, enumerate all x once and retain c -> preimages. Then check the
guarded program at every c. No proof case or ambiguous preimage is discarded.
This is an engineering improvement to verification, not learned eta.
"""

import itertools

from .core import P, evaluate, forward, guard


def prove_indexed(tree, left, work):
    def validate_relation(node):
        work.add("proof_dependency_nodes")
        if node == "x":
            return
        if (not isinstance(node, (tuple, list)) or len(node) != 3
                or node[0] not in {"add", "sub", "mul"} or node[2] not in {"a", "b"}):
            raise ValueError("Indexed proof requires a forward relation independent of the right-hand side c")
        validate_relation(node[1])

    validate_relation(left)
    accepted = 0
    for a, b in itertools.product(range(P), repeat=2):
        preimages = [[] for _ in range(P)]
        inputs = dict(a=a, b=b, c=0)
        for x in range(P):
            observed = forward(left, inputs, x, work)
            preimages[observed].append(x)
            work.add("proof_index_insertions")
        for c in range(P):
            inputs["c"] = c
            if guard(tree, inputs, work):
                prediction = evaluate(tree, inputs, work)
                work.add("proof_index_lookups")
                if preimages[c] != [prediction]:
                    raise ValueError("Finite proof found an incorrect or ambiguous answer")
                accepted += 1
    if not accepted:
        raise ValueError("Vacuous all-abstain program")
    return {"domain": "F11^3", "checked_inputs": P**3, "accepted_inputs": accepted,
            "scope": "Exhaustive correctness inside supplied finite semantics and preconditions only"}
