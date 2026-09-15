"""Independent saved-data audit; explicitly invoked through the bounded pytest worker.

No compiler, library, inverse reasoning, or study implementation is imported.
Only the forward finite relation defines the assessment answers here.
"""

import gzip
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def operations(counts):
    return sum(v for k, v in counts.items() if not k.endswith("_bytes"))


def audit(run, release):
    started = time.perf_counter()
    protocol = json.loads((run/"protocol.json").read_text())
    cfg = protocol["payload"]
    require(digest(cfg) == protocol["sha256"], "Protocol integrity")
    require(protocol == json.loads((release/"protocol.json").read_text()), "Published protocol differs")
    with zipfile.ZipFile(release/"frozen-sources.zip") as archive:
        require(set(archive.namelist()) == set(cfg["sources"]), "Frozen source inventory")
        for name, sha in cfg["sources"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha, "Frozen implementation changed")
    summary = json.loads((run/"summary.json").read_text())
    raw = (run/"evidence.json.gz").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == summary["evidence_sha256"], "Evidence integrity")
    evidence = json.loads(gzip.decompress(raw))
    checks, cases = 0, 0
    all_panels = list(evidence["panels"].items())+[("library_growth", p) for p in evidence["library_growth"].values()]
    for panel, data in all_panels:
        tasks = data["tasks"]
        require(len(tasks) == cfg["counts"][panel] and len({t["id"] for t in tasks}) == len(tasks), "Incomplete/duplicate assessment")
        expected = []
        for task in tasks:
            require(task["panel"] == panel, "Wrong task panel")
            require(len(task["chain"]) == (3 if panel == "withheld_compositions" else 1), "Missing composition")
            previous = None
            for item in task["chain"]:
                bindings = {role: item["values"][name] for role, name in item["bindings"].items()}
                if previous is not None:
                    bindings["c"] = previous
                a, b, c = [bindings[n] for n in ("a", "b", "c")]
                sign = -1 if panel in ("after_correction", "library_growth") else 1
                solutions = [x for x in range(11) if (a*x+sign*b) % 11 == c]
                previous = solutions[0] if len(solutions) == 1 else None
                if previous is None:
                    break
            expected.append(previous)
        require(expected == data["assessor_truth"], "Independent forward assessment mismatch")
        cases += len(tasks)
        for method, record in data["methods"].items():
            answers = record["answers"]
            require(len(answers) == len(tasks), "Missing answers")
            require(sum(a is not None for a in answers) == record["answered"], "Answer count")
            require(sum(a is not None and a != b for a, b in zip(answers, expected)) == record["wrong_answers"], "False-answer count")
            require(sum(a is None and b is not None for a, b in zip(answers, expected)) == record["missed_unique_solutions"], "Abstention count")
            require(operations(record["counts"]) == record["operations"], "Operation proxy mismatch")
            if method != "omit_guard":
                require(answers == expected, "Guarded or baseline answer differs")
            checks += len(answers)
    overhead = sum(operations(summary[k]) for k in ("learning_work", "correction_work", "restore_work"))
    require(overhead == summary["acquisition_verification_restore_overhead"], "Acquisition/proof/restore costs omitted")
    for method in cfg["methods"]:
        online = sum(v["methods"][method]["operations"] for v in evidence["panels"].values())
        require(online == summary["online_stream_operations"][method], "Online cost mismatch")
        total = online+(overhead if method.startswith("compiled") or method == "omit_guard" else 0)
        require(total == summary["full_stream_operations"][method], "Full cost mismatch")
    initial, corrected = [json.loads((run/f"{name}-library.json").read_text()) for name in ("initial", "corrected")]
    for record in (initial, corrected):
        graph = record["graph"]
        require(digest(graph) == record["graph_sha256"], "Graph identity")
        definitions = {d["name"]: d for d in graph["definitions"]}
        for definition in definitions.values():
            for dep in definition["dependencies"]:
                require(dep["name"] in definitions and digest(definitions[dep["name"]]) == dep["sha256"], "Stale transitive dependency")
    before = {d["name"]: d for d in initial["graph"]["definitions"]}
    after = {d["name"]: d for d in corrected["graph"]["definitions"]}
    require(before["compiled/offset"] == after["compiled/offset"], "Unrelated knowledge changed")
    require(before["compiled/affine"] != after["compiled/affine"], "Correction did not replace obsolete program")
    require(initial["owner_sha256"] == corrected["owner_sha256"] == summary["shared_owner_sha256"], "Owner changed")
    return {"status": "PASS", "case_occurrences": cases, "paired_policy_answers_checked": checks,
            "finite_field": 11, "full_costs_reconciled": True, "graphs_and_dependencies_verified": True,
            "wall_seconds": time.perf_counter()-started,
            "scope": "Independent forward arithmetic for all saved cases, complete cohorts, decision and cost counts, frozen sources, graph hashes and transitive dependencies. Source review and unit tests separately validate runtime correction/reload. No neural transfer claim."}


def test_saved_evidence():
    run = Path(os.environ["SERA_V3_AUDIT_RUN"])
    release = Path(os.environ["SERA_V3_AUDIT_RELEASE"])
    result = audit(run, release)
    with Path(os.environ["SERA_V3_AUDIT_OUTPUT"]).open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))
