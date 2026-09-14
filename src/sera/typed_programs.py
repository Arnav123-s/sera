"""Bounded arithmetic procedure induction from admitted examples and fresh checks."""

from __future__ import annotations

import re

from sera.contracts import EvidenceKind
from sera.storage import digest

OPERATIONS = ("sum", "minimum", "maximum", "first", "last", "length", "alternating_sum")


def numeric_input(observations):
    if all(row.modality == "numeric" and row.units == "dimensionless" and len(row.values) == 1
           and row.scale == 1 and (row.available is None or all(row.available)) for row in observations):
        numbers = [row.values[0] for row in observations]
    elif len(observations) == 1 and observations[0].modality == "text_bytes":
        row = observations[0]
        if row.scale != 1 or (row.available is not None and not all(row.available)):
            raise ValueError("Program text input must be fully available unscaled bytes")
        if any(value != int(value) or not 0 <= value <= 255 for value in row.values):
            raise ValueError("Invalid program text byte")
        spelling = bytes(int(value) for value in row.values).decode("ascii")
        if not re.fullmatch(r"\s*\d+(?:\s*\+\s*\d+)+\s*", spelling):
            raise ValueError("Input is outside the declared decimal-addition grammar")
        numbers = [int(value) for value in re.findall(r"\d+", spelling)]
    else:
        raise ValueError("Input is outside the declared arithmetic representation")
    if not 1 <= len(numbers) <= 64 or any(number != int(number) or abs(number) > 10000 for number in numbers):
        raise ValueError("Arithmetic input exceeds its finite integer contract")
    return [int(number) for number in numbers]


def execute_rule(rule, observations):
    if set(rule) != {"operation", "modulus"} or rule["operation"] not in OPERATIONS or rule["modulus"] not in (2, 3, 4):
        raise ValueError("Unregistered typed arithmetic instruction")
    values = numeric_input(observations)
    result = {"sum": lambda: sum(values), "minimum": lambda: min(values), "maximum": lambda: max(values),
              "first": lambda: values[0], "last": lambda: values[-1], "length": lambda: len(values),
              "alternating_sum": lambda: sum((-1)**index * value for index, value in enumerate(values))}[rule["operation"]]()
    return result % rule["modulus"]


def validate_program(record):
    if record.get("schema_version") != 1 or record.get("rule", {}).get("operation") not in OPERATIONS:
        raise ValueError("Invalid typed program schema")
    body = {key: value for key, value in record.items() if key != "identity"}
    if digest(body) != record.get("identity") or not record.get("support_ids") or not record.get("validation_ids"):
        raise ValueError("Typed program evidence or integrity failure")
    if set(record["support_ids"]) & set(record["validation_ids"]):
        raise ValueError("Typed program checks overlap teaching examples")
    if record["rule"].get("modulus") not in (2, 3, 4):
        raise ValueError("Invalid arithmetic modulus")


def induce(records, validation, *, task, work=None):
    support, checks = [row for row in records if row.task == task], [row for row in validation if row.task == task]
    if not support or not checks or any(not row.split.startswith("validation") for row in checks):
        raise ValueError("Procedure induction requires admitted support and separate validation")
    if any(row.evidence.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}
           or row.split.startswith(("test", "query", "evaluation", "promotion")) for row in support + checks):
        raise ValueError("Procedure examples are not admitted development evidence")
    if {row.identifier for row in support} & {row.identifier for row in checks}:
        raise ValueError("Procedure support and validation overlap")
    tried, finalists = [], []
    for operation in OPERATIONS:
        for modulus in (2, 3, 4):
            rule = {"operation": operation, "modulus": modulus}
            mismatches = []
            for row in support:
                if work is not None:
                    work.add("typed_program_support_executions")
                try:
                    output = execute_rule(rule, row.observations)
                except (ValueError, UnicodeError):
                    output = None
                if output != row.target:
                    mismatches.append(row.identifier)
            tried.append({"rule": rule, "failure_count": len(mismatches), "first_counterexample": next(iter(mismatches), None)})
            if not mismatches:
                finalists.append(rule)
    if len(finalists) != 1:
        return {"accepted": False, "reason": "No unique support-consistent procedure", "attempts": tried}
    rule = finalists[0]
    if work is not None:
        work.add("typed_program_validation_executions", len(checks))
    if not all(execute_rule(rule, row.observations) == row.target for row in checks):
        return {"accepted": False, "reason": "Independent validation rejected the procedure", "attempts": tried}
    record = {"schema_version": 1, "task": task, "rule": rule, "support_ids": [row.identifier for row in support],
              "signature": "bounded integer sequence or decimal expression -> modulo class",
              "validation_ids": [row.identifier for row in checks], "attempts": tried,
              "scope": "A supplied 21-candidate integer grammar selected by examples. The parser and candidate operations are engineered; the selected rule is induced. Validation is empirical within the declared domain."}
    record["identity"] = digest(record)
    return {"accepted": True, "record": record}
