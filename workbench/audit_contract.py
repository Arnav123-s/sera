"""Corrective output validation on preserved predictions, without retraining."""

import hashlib
import json
from pathlib import Path

from experiments.grounded_language.study import RELEASE, packed, read, write

from .math_contract import check_translation


def main():
    import gzip
    output = RELEASE/"CONTRACT-001"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in (151001, 151009, 151027):
        folder = RELEASE/f"L10-FINAL-{seed}-interface-assessment"
        raw = json.loads(gzip.decompress((folder/"raw.json.gz").read_bytes()))
        for example, prediction in zip(raw["wording"]["after"], raw["wording"]["decisions"]):
            checked = check_translation(example["text"], prediction)
            accepted = checked["status"] == "ACCEPTED"
            correct = prediction.get("labels") == example["target"]
            if accepted and not correct:
                raise ValueError("The supplied contract admitted a mistranslation")
            rows.append({"seed": seed, "text": example["text"], "target": example["target"],
                         "previously_accepted": prediction["status"] == "ACCEPTED", "correct": correct,
                         "now_accepted": accepted})
    continuing = read(RELEASE/"L11-ONREQUEST-001/result.json")
    for row in continuing["records"]:
        assert check_translation(row["request"], row["answer"])["status"] == "ACCEPTED"
    packed(output/"cases.json.gz", rows)
    accepted = [row for row in rows if row["now_accepted"]]
    report = {"status": "PASS", "checked_cases": len(rows),
              "previously_accepted": sum(row["previously_accepted"] for row in rows),
              "previously_accepted_errors": sum(row["previously_accepted"] and not row["correct"] for row in rows),
              "accepted": len(accepted), "accepted_errors": sum(not row["correct"] for row in accepted),
              "coverage": len(accepted)/len(rows), "continuing_requests_preserved": len(continuing["records"]),
              "checker_sha256": hashlib.sha256(Path(__file__).with_name("math_contract.py").read_bytes()).hexdigest(),
              "boundary": "Post-audit correction validated against archived cases and an exhaustive finite-grammar unit check. No new neural training, prospective accuracy claim, statistical risk certificate, or general semantic verifier."}
    write(output/"result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
