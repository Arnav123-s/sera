"""Finite actual-owner acquisition cycle; held-out assessment is a separate command."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from experiments.stream_curriculum.data import first_rows
from experiments.stream_curriculum.model import encode
from workbench.storage import Store
from .algebra import SIZE, independent
from .lessons import calculus
from .runtime import OUT, StudySession, fingerprint
from .sources import Arxiv, ROOT


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def retained(session):
    texts = [r["text"] for r in first_rows("train", 32)]
    return [v.clone() for v in session.owner.request_logits(encode(texts))]


def problems(seed, count):
    rng = random.Random(seed)
    tasks = []
    for domain, degrees in (("sum", (1, 3, 5, 7, 9)), ("integral", tuple(range(6)))):
        for i in range(count):
            p = [0] * SIZE
            for k in degrees:
                p[k] = rng.randint(-7, 7)
            tasks.append({"id": f"{domain}-{i}", "domain": domain, "p": p})
    return tasks


def assessment(session, tasks):
    rows = []
    for t in tasks:
        answer = session.propose(t["domain"], t["p"])
        rows.append({**t, "answer": answer,
                     "independent": independent(t["domain"], t["p"], answer["proposal"])})
    return {"correct": sum(r["independent"] for r in rows), "total": len(rows), "records": rows}


def develop(directory):
    directory.mkdir(parents=True, exist_ok=False)
    parent = json.loads((OUT / "parent-request.json").read_text())
    client = Arxiv(ROOT / "runs/SS-sources")
    goal = [0, 2, 0, 3, 0, 1]
    source = json.loads((OUT / "calculus-source.json").read_text())
    tasks = problems(2702, 32)
    summaries = {}
    for policy in ("verified", "read_only", "unchecked_control"):
        session = StudySession(parent)
        old = {n: v.clone() for n, v in session.owner.state_dict().items() if not n.startswith("study_")}
        baseline = retained(session)
        store = Store(directory / policy)
        previous = [None]
        def persist(value):
            previous[0] = store.commit(value, previous[0])
        frozen = assessment(session, tasks)
        original = session.autonomous_round("power-sum", "sum", goal, client, policy=policy, persist=persist)
        after_sum = session.snapshot()
        restored = StudySession(parent, after_sum)
        assert restored.snapshot() == after_sum
        integral = session.autonomous_round("antiderivative", "integral", [1, -2, 3, 1], client,
                                             policy=policy, lessons=calculus(source), persist=persist)
        after = assessment(session, tasks)
        assert all(torch.equal(v, session.owner.state_dict()[n]) for n, v in old.items())
        assert all(torch.equal(a, b) for a, b in zip(baseline, retained(session)))
        result = {"policy": policy, "source": fingerprint(), "frozen": frozen,
                  "original_goal": original, "integral_goal": integral, "after": after,
                  "sum_retention": session.propose("sum", goal), "exact_restore": True,
                  "old_tensor_count_unchanged": len(old), "request_outputs_bit_exact": 32,
                  "motion": session.motion([2, 3, 1], "3", "5", "-1"),
                  "events": session.events, "snapshot": session.snapshot()}
        write(directory / f"{policy}.json", result)
        summaries[policy] = {"before": frozen["correct"], "after": after["correct"],
                             "total": after["total"], "original_goal": original["status"],
                             "rejected_source_lessons": sum(not e["check"]["accepted"] for e in session.events)}
        print(json.dumps({policy: summaries[policy]}), flush=True)
    write(directory / "summary.json", summaries)
    write(directory / "freeze.json", {"source": fingerprint(), "protocol_sha256": hashlib.sha256((OUT / "protocol.md").read_bytes()).hexdigest(),
                                      "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.json")}})


def final(directory, development):
    directory.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((development / "freeze.json").read_text())
    if frozen["source"] != fingerprint():
        raise ValueError("Implementation changed after development freeze")
    for name, expected in frozen["files"].items():
        if hashlib.sha256((development / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Development record changed")
    parent = json.loads((OUT / "parent-request.json").read_text())
    tasks = problems(2791, 128)
    summaries = {}
    for policy in ("verified", "read_only", "unchecked_control"):
        saved = json.loads((development / f"{policy}.json").read_text())["snapshot"]
        session = StudySession(parent, saved)
        result = assessment(session, tasks)
        motion = []
        rng = random.Random(2792)
        for _ in range(64):
            a = [rng.randint(-7, 7) for _ in range(3)]
            t, x, v = rng.randint(1, 7), rng.randint(-7, 7), rng.randint(-7, 7)
            answer = session.motion(a, str(t), str(x), str(v))
            valid = False
            if answer["status"] == "VERIFIED":
                from fractions import Fraction as Q
                # Independent closed form used only for scoring, never learner input.
                target_v = Q(v) + sum(Q(c, i + 1) * t ** (i + 1) for i, c in enumerate(a))
                target_x = Q(x) + v * t + sum(Q(c, (i + 1) * (i + 2)) * t ** (i + 2) for i, c in enumerate(a))
                valid = Q(answer["position"]) == target_x and Q(answer["velocity"]) == target_v
            motion.append({"acceleration": a, "time": t, "x0": x, "v0": v, "answer": answer, "independent": valid})
        write(directory / f"{policy}.json", {"polynomials": result, "motion": motion})
        summaries[policy] = {"polynomial_correct": result["correct"], "polynomial_total": result["total"],
                             "motion_correct": sum(m["independent"] for m in motion), "motion_total": len(motion)}
    # Privileged direct solver is assessed separately; no source acquisition or learning.
    import sympy as sp
    x, k = sp.symbols("x k", integer=True)
    correct = 0
    for task in tasks:
        p = sum(c * x ** i for i, c in enumerate(task["p"]))
        q = sp.summation(p.subs(x, k), (k, 0, x - 1)) if task["domain"] == "sum" else sp.integrate(p, x)
        values = [str(sp.expand(q).coeff(x, i)) for i in range(SIZE)]
        correct += independent(task["domain"], task["p"], values)
    summaries["installed_symbolic_reference"] = {"correct": correct, "total": len(tasks), "reads": 0}
    write(directory / "summary.json", summaries)
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("develop", "final"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == "develop":
        develop(args.output)
    else:
        final(args.output, args.development)


if __name__ == "__main__":
    main()
