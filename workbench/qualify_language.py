"""Apply the frozen release decision after complete independent cohort audit."""

import json

from experiments.grounded_language.model import fingerprint
from experiments.grounded_language.study import RELEASE, read, sha, write


def main():
    audit = read(RELEASE/"independent-audit.json")
    protocol = read(RELEASE/"cohort-protocol.json")
    assert audit["status"] == "PASS"
    interfaces = [row for row in audit["comparisons"] if row["arm"] == "interface"]
    gates = {"known_translation": all(row["known"]["exact_translation"] >= .95 for row in interfaces),
             "accepted_translation": all((row["known"]["accepted_translation_accuracy"] or 0) >= .99 for row in interfaces),
             "coverage": all(row["known"]["coverage"] >= .5 for row in interfaces),
             "old_skills": all(row["old_skill_maximum_drop"] <= .02 and row["old_skill_exact_scores"] for row in interfaces),
             "three_declared_seeds": len(interfaces) == 3}
    candidate = read(RELEASE/protocol["intended_release_candidate"]/"result.json")
    report = {"status": "QUALIFIED_RESTRICTED_INTERFACE" if all(gates.values()) else "REJECTED",
              "candidate": protocol["intended_release_candidate"], "gates": gates,
              "language_source": fingerprint(), "checkpoint": candidate["selected_checkpoint"],
              "threshold": candidate["calibration"]["threshold"],
              "audit_sha256": sha(RELEASE/"independent-audit.json"),
              "cohort_protocol_sha256": sha(RELEASE/"cohort-protocol.json"),
              "allowed": "Six supplied sentence shapes, coefficients zero to ten, two operation orders, modulus eleven. Additional wording must pass a new explicit lesson check; prior wording is retested and can be withdrawn.",
              "not_qualified": ["unrestricted English", "new domains or general mathematical proofs", "positive pretrained-core transfer", "shared-core replacement", "learned curriculum or optimizer"]}
    if (RELEASE/"qualification.json").exists():
        raise FileExistsError("The qualification decision is immutable")
    write(RELEASE/"qualification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
