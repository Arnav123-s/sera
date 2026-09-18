"""Trusted, separate-process practice checker and fresh portfolio assessor.

The issuer owns reference answers and an append-only attempt history. Its local
key is outside learner snapshots. This is a process/API boundary, not isolation
from arbitrary programs running as the same Windows user.
"""

import argparse
import hashlib
import hmac
import json
import math
import random
import secrets
import sys
from fractions import Fraction
from pathlib import Path

from .common import digest, read, write

SCOPES = ("time", "impulse", "work_positive", "work_negative")


def lower_bound(successes, n, alpha):
    if successes == 0:
        return 0.
    if successes == n:
        return alpha ** (1/n)
    lo, hi = 0., 1.
    for _ in range(70):
        p = (lo+hi)/2
        terms = [math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                 + k*math.log(p)+(n-k)*math.log1p(-p) for k in range(successes, n+1)]
        peak = max(terms)
        tail = math.exp(peak)*sum(math.exp(v-peak) for v in terms)
        if tail < alpha:
            lo = p
        else:
            hi = p
    return (lo+hi)/2


def challenge(seed, n):
    """Build reference outcomes from a latent endpoint, not candidate execution."""
    rng = random.Random(seed)
    rows, truth = [], []
    for _ in range(n):
        scope = rng.randrange(4)
        mass, initial = Fraction(rng.randint(1, 10), 2), Fraction(rng.randint(-8, 8), 2)
        endpoint = Fraction(rng.randint(1, 12), 2)*(1 if scope != 3 else -1)
        row = {"mechanism": "constant_mechanics", "units": "SI", "scope": SCOPES[scope],
               "v0": str(initial), "assumptions": {"constant_mass": True},
               "origin": "SUPPLIED_CONDITIONAL_MECHANICS"}
        if scope == 0:
            acceleration, duration = Fraction(rng.randint(-6, 6), 6), Fraction(rng.randint(1, 24), 6)
            row.update(a=str(acceleration), t=str(duration))
            row["assumptions"]["constant_acceleration"] = True
            endpoint = initial+acceleration*duration
        elif scope == 1:
            row.update(m=str(mass), j=str(mass*(endpoint-initial)))
        else:
            row.update(m=str(mass), w=str(mass*(endpoint*endpoint-initial*initial)/2))
            row["assumptions"]["positive_final_velocity" if scope == 2 else "negative_final_velocity"] = True
        rows.append(row)
        truth.append(float(endpoint))
    return rows, truth


class Issuer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        keyfile = self.path / "issuer.key"
        if not keyfile.exists():
            with keyfile.open("xb") as file:
                file.write(secrets.token_bytes(32))
        self.key = keyfile.read_bytes()
        self.file = self.path / "ledger.json"
        self.state = read(self.file) if self.file.exists() else {"quests": {}, "pending": {}, "receipts": {}}

    def save(self):
        temporary = self.path / "ledger.tmp"
        write(temporary, self.state)
        temporary.replace(self.file)

    def begin(self, request):
        mode = request["mode"]
        if mode not in {"practice", "boss"}:
            raise ValueError("A practice or sealed assessment purpose is required")
        if len(self.state["pending"]) >= 8:
            raise ValueError("Resume outstanding checks first")
        key = request["quest"]
        entry = self.state["quests"].setdefault(key, {"index": len(self.state["quests"])+1, "attempts": 0})
        alpha, attempt = None, None
        if mode == "boss":
            retention = request["retention"]
            if (retention.get("owner") != request["owner"] or retention.get("protected_equal") is not True
                    or retention.get("language_max_error") != 0. or retention.get("probes") != 128
                    or retention.get("old_definition_gate") is not False):
                raise ValueError("Actual-owner retained-capability receipt required")
            entry["attempts"] += 1
            attempt, index = entry["attempts"], entry["index"]
            alpha = .05/(index*(index+1)*attempt*(attempt+1))
        identifier = secrets.token_hex(20)
        seed = secrets.randbits(64)
        rows, truth = challenge(seed, 256 if mode == "boss" else 64)
        record = {"id": identifier, "request": request, "index": entry["index"], "attempt": attempt,
                  "alpha": alpha, "seed": seed, "cases": rows, "truth": truth}
        self.state["pending"][identifier] = record
        self.save()  # Spend the assessment attempt before revealing any inputs.
        return {"id": identifier, "cases": rows, "inputs_sha256": digest(rows)}

    def grade(self, request):
        record = self.state["pending"][request["id"]]
        original = record["request"]
        if request["owner"] != original["owner"] or request["candidate"] != original["candidate"]:
            raise ValueError("Assessment must use the frozen owner and candidate")
        predictions = request["predictions"]
        if len(predictions) != len(record["truth"]):
            raise ValueError("Every assessed input needs a prediction or abstention")
        good, wrong, returned = [], [], []
        for index, (value, expected) in enumerate(zip(predictions, record["truth"], strict=True)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Finite numeric prediction required")
            returned.append(index)
            (good if abs(value-expected) <= 1e-8 else wrong).append(index)
        valid_scopes = sorted({record["cases"][i]["scope"] for i in good}) if not wrong else []
        lower = None if original["mode"] == "practice" else lower_bound(len(good), len(predictions), record["alpha"])
        body = {"id": record["id"], "mode": original["mode"], "quest": original["quest"],
                "owner": original["owner"], "candidate": original["candidate"], "decision": original.get("decision"),
                "retention": original.get("retention"), "inputs_sha256": digest(record["cases"]),
                "prediction_sha256": digest(predictions), "correct": len(good), "wrong": len(wrong),
                "returned": len(returned), "n": len(predictions), "valid_scopes": valid_scopes,
                "counterexamples": [{"input": record["cases"][i], "prediction": predictions[i],
                                      "expected": record["truth"][i]} for i in wrong[:8]],
                "global_quest_index": record["index"], "attempt": record["attempt"], "alpha": record["alpha"],
                "lower_bound": lower, "qualified": lower is not None and lower >= .95,
                "evidence_kind": "SUPPLIED_CONDITIONAL_MECHANICS"}
        receipt = body | {"signature": hmac.new(self.key, digest(body).encode(), hashlib.sha256).hexdigest()}
        # Preserve failed answers and hidden reference outcomes, including spent attempts.
        write(self.path / "completed" / (record["id"]+".json"), record | {"predictions": predictions, "receipt": receipt})
        self.state["receipts"][record["id"]] = receipt
        del self.state["pending"][record["id"]]
        self.save()
        return receipt

    def verify(self, receipts):
        for receipt in receipts:
            body = {k: v for k, v in receipt.items() if k != "signature"}
            signature = hmac.new(self.key, digest(body).encode(), hashlib.sha256).hexdigest()
            if (not hmac.compare_digest(signature, receipt["signature"])
                    or self.state["receipts"].get(receipt["id"]) != receipt):
                raise ValueError("Changed or unissued independent receipt")
        return {"verified": len(receipts)}

    def recover(self, request):
        matching = [r for r in self.state["pending"].values() if r["request"] == request]
        matching += [r for r in self.state["receipts"].values()
                     if all(r[k] == request.get(k) for k in ("mode", "quest", "candidate", "owner", "decision"))]
        if not matching:
            return None
        record = matching[-1]
        if "signature" in record:
            return {"receipt": record}
        return {"id": record["id"], "cases": record["cases"], "inputs_sha256": digest(record["cases"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    issuer = Issuer(args.store)
    if request["action"] == "begin":
        result = issuer.begin(request["request"])
    elif request["action"] == "grade":
        result = issuer.grade(request["request"])
    elif request["action"] == "verify":
        result = issuer.verify(request["receipts"])
    elif request["action"] == "recover":
        result = issuer.recover(request["request"])
    else:
        raise ValueError("Unknown assessor operation")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
