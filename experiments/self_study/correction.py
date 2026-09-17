"""Repair a rejected source claim through a bounded, independently checked search."""
import argparse
import copy
import hashlib
import json
import random
from fractions import Fraction as Q
from pathlib import Path

import torch

from .algebra import SIZE, check, independent, vector
from .lessons import expression
from .runtime import OUT, StudySession
from .sources import ROOT


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def assess(session, seed, count):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        p = [0] * SIZE
        for j in (1, 3, 5, 7, 9, 11):
            p[j] = rng.randint(-7, 7)
        answer = session.propose("sum", p)
        rows.append({"input": p, "answer": answer, "independent": independent("sum", p, answer["proposal"])})
    return {"correct": sum(r["independent"] for r in rows), "count": count, "records": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("develop", "final"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.study.read_text())
    saved = data["snapshot"]
    session = StudySession(json.loads((OUT / "parent-request.json").read_text()), saved)
    if args.mode == "final":
        if data["source"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Correction implementation changed after development")
        result = assess(session, 2811, 128)
        write(args.output / "result.json", result)
        print(json.dumps({"correct": result["correct"], "count": result["count"]}))
        return
    failed = [e["lesson"] for e in session.events if not e["check"]["accepted"] and e["lesson"]["domain"] == "sum"]
    if len(failed) != 1:
        raise ValueError("Expected one preserved failed mathematical claim")
    lesson = failed[0]
    target = lesson["input"]
    before = session.propose("sum", target)
    protected = {n: v.clone() for n, v in session.owner.state_dict().items() if not n.startswith("study_")}
    candidates, accepted = [], []
    for power in range(1, 7):
        basis = expression(f"N^{{{power}}}", [0, Q(-1, 2), Q(1, 2)])
        for delta in range(-8, 9):
            if delta == 0:
                continue
            proposed = [str(a + Q(delta, 6) * b) for a, b in zip(vector(lesson["output"]), basis)]
            proof = check("sum", target, proposed)
            verified = independent("sum", target, proposed)
            assert proof["accepted"] == verified
            candidates.append({"N_power": power, "coefficient_delta": delta, "denominator": 6,
                               "output": proposed, "certificate": proof, "independent": verified})
            if verified:
                accepted.append(candidates[-1])
    write(args.output / "candidates.json", candidates)
    if len(accepted) != 1:
        raise ValueError("No unique certified correction in the frozen family")
    repaired = copy.deepcopy(lesson)
    repaired.update(id=lesson["id"] + "-independently-corrected", output=accepted[0]["output"],
                    interpretation="bounded single-coefficient correction, independently certified",
                    derivation={"original_source_claim_rejected": True, "candidate_count": len(candidates),
                                "change": {k: accepted[0][k] for k in ("N_power", "coefficient_delta", "denominator")}})
    goal = session.autonomous_round("eleventh-power-correction", "sum", target, None,
                                    reads=1, lessons=[repaired])
    retained = session.propose("integral", [1, -2, 3, 1])
    assert all(torch.equal(v, session.owner.state_dict()[n]) for n, v in protected.items())
    result = {"before": before, "after": goal, "retained_integral": retained,
              "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "development": assess(session, 2801, 32), "snapshot": session.snapshot(),
              "correction": repaired, "protected_tensors": len(protected)}
    write(args.output / "result.json", result)
    print(json.dumps({"candidates": len(candidates), "unique_verified": len(accepted),
                      "correction": repaired["derivation"]["change"],
                      "before": before["status"], "after": goal["status"],
                      "development": result["development"]["correct"]}))


if __name__ == "__main__":
    main()
