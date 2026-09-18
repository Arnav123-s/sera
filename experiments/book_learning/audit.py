"""Independent retention, exact resume, source isolation and conditional execution."""

import copy
import json
import random
from fractions import Fraction

import numpy as np
import torch
from torch.nn import functional as F

from experiments.concept_refinement.audit import languages
from experiments.human_reading.data import ROOT, normalized, read, sha, write
from experiments.human_reading.runtime import ReadingSession
from sera.session_state import model_identity
from workbench.storage import Store

from .data import DOMAINS, OUT, RUN
from .ground_data import GROUND, ROLES
from .ground_model import KINDS, GroundR1, restore_books
from .ground_model import apply as ground_apply
from .ground_model import delta as ground_delta
from .grounding import targets
from .grounding_v2 import RUN as NAMED_RUN
from .grounding_v2 import SELECTION, text
from .model import apply as book_apply
from .model import delta as book_delta
from .runtime import PREMISES, UNITS, GroundedSession
from .study import labels


def language_replay():
    owner = restore_books().owner
    saved = torch.load(RUN / "science/step-0080.pt", map_location="cpu", weights_only=True)
    book_apply(owner, saved["delta"])
    for p in owner.book_heads.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(owner.book_heads.parameters(), lr=.003, weight_decay=.0001)
    optimizer.load_state_dict(saved["optimizer"])
    rng = torch.Generator()
    rng.set_state(saved["rng"])
    order, cursor = saved["order"].clone(), saved["cursor"]
    x = torch.load(RUN / "features-science-train.pt", map_location="cpu", weights_only=True)["features"]
    y = labels(read(RUN / "science-train.json"), saved["vocabulary"])
    for _ in range(81, 121):
        if cursor == len(order):
            order, cursor = torch.randperm(len(y), generator=rng), 0
        ids = order[cursor:cursor+64]
        cursor += len(ids)
        optimizer.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(owner.book_logits(x[ids], k), y[ids]) for k in ("shared", "pooled"))
        loss.backward()
        optimizer.step()
    final = torch.load(RUN / "science/step-0120.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(v, book_delta(owner)[k]) for k, v in final["delta"].items())


def binding_replay():
    owner = GroundR1.attach(restore_books().owner)
    saved = torch.load(NAMED_RUN / "step-0080.pt", map_location="cpu", weights_only=True)
    ground_apply(owner, saved["delta"])
    optimizer = torch.optim.AdamW(owner.ground_heads.parameters(), lr=.01, weight_decay=.01)
    optimizer.load_state_dict(saved["optimizer"])
    rows = read(GROUND / "teaching.json")["train"]
    x, y = owner.ground_features([text(r) for r in rows]), targets(rows)
    weight = 1/y.bincount(minlength=len(ROLES)).float()
    for _ in range(81, 161):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(owner.ground_logits(x, k), y, weight=weight) for k in KINDS)
        loss.backward()
        optimizer.step()
    final = torch.load(NAMED_RUN / "step-0160.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(v, ground_delta(owner)[k]) for k, v in final["delta"].items())


def data_checks():
    isolation = {}
    for domain in DOMAINS:
        rows = [read(RUN / f"{domain}-{p}.json") for p in ("train", "dev", "final")]
        groups = [{r["group"] for r in part} for part in rows]
        texts = [{normalized(" ".join(r["context"]+[r["target"]])) for r in part} for part in rows]
        for i in range(3):
            for j in range(i):
                assert not groups[i] & groups[j] and not texts[i] & texts[j]
        isolation[domain] = {"windows": [len(p) for p in rows], "groups": [len(p) for p in groups], "disjoint": True}
    results = read(OUT / "final.json")["results"]
    for domain in DOMAINS:
        for result in results[domain].values():
            assert abs(np.mean(result["losses"])-result["cross_entropy"]) < 2e-6
            pairs = [(p, y) for p, y in zip(result["predictions"], result["targets"], strict=True) if y != 0]
            assert abs(sum(p == y for p, y in pairs)/len(pairs)-result["covered_accuracy"]) < 1e-7
    teaching = read(GROUND / "teaching.json")
    taught_texts = {normalized(r["text"]) for p in ("train", "dev") for r in teaching[p]}
    final_rows = read(NAMED_RUN / "final.json")["rows"]
    assert not taught_texts & {normalized(r["text"]) for r in final_rows}
    for result in read(OUT / "named-final.json")["results"].values():
        records = result["predictions"]
        assert abs(sum(r["correct"] for r in records)/len(records)-result["accuracy"]) < 1e-7
        assert all(r["correct"] == (r["gold"] == r["predicted"]) for r in records)
    return isolation


def main():
    destination = OUT / "integration-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed integration audit")
    torch.set_num_threads(1)
    isolation = data_checks()
    predecessor = ReadingSession()
    old = {k: v.clone() for k, v in predecessor.owner.state_dict().items()}
    before_language, probe_ids = languages(predecessor.owner)
    before_math = predecessor.base.exact_motion([2, 3, 1], "3", "5", "-1")
    subject = next(iter(predecessor.base.subjects))
    before_empirical = predecessor.base.predict(subject, [.8]*4)
    session = GroundedSession()
    assert all(torch.equal(v, session.owner.state_dict()[k]) for k, v in old.items())
    after_language, after_ids = languages(session.owner)
    assert probe_ids == after_ids
    assert all(torch.equal(a, b) for key in before_language for a, b in zip(before_language[key], after_language[key], strict=True))
    assert session.base.exact_motion([2, 3, 1], "3", "5", "-1") == before_math
    after_empirical = session.base.predict(subject, [.8]*4)
    assert {k: v for k, v in before_empirical.items() if k != "parent_owner"} == {k: v for k, v in after_empirical.items() if k != "parent_owner"}
    question, passage, source = "What is force?", "A force is a push or pull. A measurement has a unit.", "engineering retention probe"
    before_read = predecessor.read_passage(question, passage, source)
    after_read = session.read_passage(question, passage, source)
    assert {k: v for k, v in before_read.items() if k != "owner"} == {k: v for k, v in after_read.items() if k != "owner"}
    bank, eligible, rejected = [], {}, []
    for row in session.bank:
        if not row["term"]:
            try:
                session.bind(row["term"], row["text"], row["source"])
            except ValueError as error:
                bank.append({"id": row["id"], "label": row["label"], "binding": {"admitted": False, "reason": str(error)}})
                continue
            raise AssertionError("Nameless source was admitted")
        binding = session.bind(row["term"], row["text"], row["source"])
        bank.append({"id": row["id"], "label": row["label"], "binding": binding})
        if binding["admitted"]:
            eligible.setdefault(row["label"], row)
        elif row["label"] != "other":
            rejected.append(row["id"])
    write(OUT / "teaching-reuse-audit.json", bank)
    initial_owner, observations = model_identity(session.owner), copy.deepcopy(session.base.subjects)
    rng = random.Random(3041)
    results, example = [], None
    for role, row in eligible.items():
        for i in range(60):
            value = Fraction(rng.randint(1, 9) if role == "mass" else rng.randint(-9, 9), rng.randint(1, 4))
            time = Fraction(rng.randint(0, 8), rng.randint(1, 3))
            req = {"id": "conditional-"+role, "goal": "Apply the taught "+role+" meaning to conditional motion",
                   "term": row["term"], "definition": row["text"], "source": row["source"],
                   "value": str(value), "unit": UNITS[role], "time": str(time),
                   "mass": "3", "force": "6", "assumptions": {**dict.fromkeys(PREMISES, True), "net_force": True}}
            result = session.imagine(req)
            assert result["status"] == "CONDITIONAL_RESULT", result
            x0, v0, a = Fraction(0), Fraction(0), Fraction(0)
            if role == "position":
                x0 = value
            elif role == "velocity":
                v0 = value
            elif role == "acceleration":
                a = value
            elif role == "force":
                a = value/3
            else:
                a = 6/value
            answer = result["branches"]["base"]["result"]
            assert Fraction(answer["position"]) == x0+v0*time+a*time*time/2
            assert Fraction(answer["velocity"]) == v0+a*time
            results.append({"role": role, "case": i, "value": str(value), "time": str(time), "position": answer["position"],
                            "velocity": answer["velocity"], "branches_checked": len(result["branches"])})
        print(json.dumps({"role": role, "checked": 60}), flush=True)
        if role == "force":
            req.update(id="force-demo", goal="Where is the body after two seconds with 6 N net force and 3 kg mass?", value="6", time="2")
            example = session.imagine(req)
            write(OUT / "example-request.json", req)
            write(OUT / "example-result.json", example)
    assert model_identity(session.owner) == initial_owner and session.base.subjects == observations
    directory = ROOT / "runs/GD-integration-audit-001"
    directory.mkdir(exist_ok=False)
    write(directory / "conditional-cases.json", results)
    store = Store(directory / "session")
    store.commit(session.snapshot(), None)
    restored = GroundedSession(store.read())
    assert restored.snapshot() == session.snapshot() and store.verify_history() == 1
    language_replay()
    binding_replay()
    report = {"status": "PASS", "actual_shared_owner": True, "owner": initial_owner,
              "predecessor_tensors_exact": len(old), "four_language_probes_exact": {k: len(v) for k, v in probe_ids.items()},
              "old_reader_exact": True, "polynomial_route_exact": True, "empirical_forecast_exact": True,
              "book_resume_80_to_120_exact": True, "binding_resume_80_to_160_exact": True,
              "snapshot_restore_exact": True, "source_isolation": isolation,
              "teaching_reuse": {"entries": len(bank), "physical_entries": sum(r["label"] != "other" for r in bank),
                                 "admitted": sum(r["binding"]["admitted"] for r in bank), "rejected_physical": rejected},
              "conditional_integration": {"cases": len(results), "branches": sum(r["branches_checked"] for r in results),
                                          "roles": list(eligible), "numerical_source": "generated diagnostics; not human language teaching",
                                          "cases_sha256": sha(directory / "conditional-cases.json")},
              "imagined_observations_added": 0, "novel_definition_gate_unchanged": read(SELECTION)["gate"],
              "limits": "Exact source reuse and supplied premises; conditional algebra does not verify meaning or occurrence."}
    write(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
