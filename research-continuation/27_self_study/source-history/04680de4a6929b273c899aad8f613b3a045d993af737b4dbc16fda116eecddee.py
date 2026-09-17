"""Independent evidence recount and checked integration of the admitted successor."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from workbench.storage import Store

from .algebra import SIZE, independent, vector
from .runtime import OUT, StudySession, fingerprint
from .sources import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def spans(tags):
    intervals, start = set(), 0
    while start < len(tags):
        if tags[start] == "O":
            start += 1
            continue
        label = tags[start][2:]
        end = start + 1
        while end < len(tags) and tags[end] == "I-" + label:
            end += 1
        intervals.add((label, start, end))
        start = end
    return intervals


def language_recount():
    counts, total = {}, 0
    for path in sorted((OUT / "language-evaluation").glob("SS-*.json")):
        counts[path.stem] = {}
        for locale, result in read(path).items():
            rows = result["records"]
            intent = frames = tp = fp = fn = tokens = matched_tokens = 0
            for row in rows:
                expected, actual = spans(row["target_tags"]), spans(row["predicted_tags"])
                right = row["target_intent"] == row["predicted_intent"]
                intent += right
                frames += right and expected == actual
                tp += len(expected & actual)
                fp += len(actual - expected)
                fn += len(expected - actual)
                tokens += len(row["target_tags"])
                matched_tokens += sum(a == b for a, b in zip(row["target_tags"], row["predicted_tags"]))
            audited = {"examples": len(rows), "intent_correct": intent,
                       "intent_accuracy": intent / len(rows), "exact_frames": frames,
                       "frame_accuracy": frames / len(rows), "slot_true_positive": tp,
                       "slot_false_positive": fp, "slot_false_negative": fn,
                       "slot_span_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                       "token_accuracy": matched_tokens / tokens}
            assert audited == {k: v for k, v in result.items() if k != "records"}
            counts[path.stem][locale] = audited
            total += len(rows)
    return {"records": total, "scores": counts}


def symbolic_check(domain, p, q):
    x = sp.Symbol("x")
    def poly(values):
        return sum(sp.Rational(str(value)) * x ** i for i, value in enumerate(values))
    left, right = poly(p), poly(q)
    residual = right.subs(x, x + 1) - right - left if domain == "sum" else sp.diff(right, x) - left
    return right.subs(x, 0) == 0 and sp.Poly(residual, x).is_zero


def math_recount():
    results = {}
    for policy in ("verified", "read_only", "unchecked_control"):
        data = read(OUT / "evaluation-001" / f"{policy}.json")
        checks = [symbolic_check(r["domain"], r["p"], r["answer"]["proposal"])
                  for r in data["polynomials"]["records"]]
        assert checks == [r["independent"] for r in data["polynomials"]["records"]]
        assert sum(checks) == data["polynomials"]["correct"]
        motion_checks = []
        for r in data["motion"]:
            right = False
            if r["answer"]["status"] == "VERIFIED":
                t = sp.Rational(r["time"])
                velocity = r["v0"] + sum(sp.Rational(c, i + 1) * t ** (i + 1)
                                         for i, c in enumerate(r["acceleration"]))
                position = r["x0"] + r["v0"] * t + sum(sp.Rational(c, (i + 1) * (i + 2)) * t ** (i + 2)
                                                      for i, c in enumerate(r["acceleration"]))
                right = (sp.Rational(r["answer"]["velocity"]) == velocity
                         and sp.Rational(r["answer"]["position"]) == position)
            assert right == r["independent"]
            motion_checks.append(right)
        results[policy] = {"polynomial_correct": sum(checks), "polynomial_records": len(checks),
                           "motion_correct": sum(motion_checks), "motion_records": len(motion_checks)}
    correction = read(OUT / "correction-final/result.json")
    checks = [symbolic_check("sum", r["input"], r["answer"]["proposal"]) for r in correction["records"]]
    assert checks == [r["independent"] for r in correction["records"]]
    assert sum(checks) == correction["correct"]
    results["correction"] = {"correct": sum(checks), "records": len(checks)}
    return results


def integrate(directory, output):
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Live descendant must remain under runs")
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory):
        store = Store(directory)
        if store.read() is not None:
            raise ValueError("Preserve an existing live successor; integrate only once into a fresh store")
        parent = read(OUT / "parent-request.json")
        selected = read(OUT / "language-evaluation/selected-session.json")
        migrated = StudySession(parent, selected)
        assert model_identity(migrated.owner) == selected["owner"]
        assert migrated.weights() == selected["weights"]
        before = migrated.snapshot()
        store.commit(before, None)
        corrected = read(OUT / "correction-development/result.json")["snapshot"]
        session = StudySession(parent, corrected)
        # Correction must preserve the evaluated language tensors and integral map.
        protected = 0
        for name, value in migrated.owner.state_dict().items():
            if not name.startswith("study_maps.sum."):
                assert torch.equal(value, session.owner.state_dict()[name]), name
                protected += 1
        retained = [("sum", [0, 2, 0, 3, 0, 1]), ("integral", [1, -2, 3, 1])]
        for domain, p in retained:
            assert session.propose(domain, p)["proposal"] == migrated.propose(domain, p)["proposal"]
        previous = store.commit(session.snapshot(), store.read())
        rh = read(OUT / "rh-goal.json")
        session.register_research_goal("RH", rh)
        class NoSource:
            def paper(self, *_args, **_kwargs):
                raise AssertionError("A retained capability unexpectedly fetched its source")
        lineage = [previous]
        def persist(value):
            lineage[0] = store.commit(value, lineage[0])
        rh_child = session.autonomous_round("RH-logarithmic-series-prerequisite", "integral", [1] * 6,
                                            NoSource(), persist=persist)
        assert rh_child["status"] == "SOLVED_WITH_CERTIFICATE" and not rh_child["source_ids"]
        with_goal = session.snapshot()
        restored = StudySession(parent, store.read())
        assert restored.snapshot() == with_goal
        # Changed parent statements are rejected, while the original stays open.
        altered = copy.deepcopy(rh)
        altered["statement"] = "a different quantified proposition"
        try:
            restored.register_research_goal("RH", altered)
        except ValueError:
            pass
        else:
            raise AssertionError("A parent research statement was silently changed")
        errors = {}
        for domain, learner in session.owner.study_maps.items():
            lessons = [e["lesson"] for e in session.events if e["learned"] and e["lesson"]["domain"] == domain]
            x = np.array([[float(v) for v in vector(l["input"])] for l in lessons])
            y = np.array([[float(v) for v in vector(l["output"])] for l in lessons])
            fitted = np.linalg.lstsq(np.concatenate([x, np.eye(SIZE) * .001]),
                                     np.concatenate([y, np.zeros((SIZE, SIZE))]), rcond=None)[0]
            errors[domain] = float(np.max(np.abs(fitted - learner.weight.numpy())))
            assert errors[domain] < 1e-8
        examples = {}
        for text in ("set an alarm for nine am", "apaga las luces por favor",
                     "mets une alarme pour neuf heures", "schalte das licht aus"):
            examples[text] = restored.interpret(text)
        calculus = restored.propose("integral", [1, 2, 3])
        motion = restored.motion([2, 3, 1], "3", "5", "-1")
        assert independent("integral", [1, 2, 3], calculus["proposal"])
        assert motion["position"] == "125/4" and motion["velocity"] == "55/2"
        record = {"status": "INTEGRATED", "source": fingerprint(), "owner": model_identity(restored.owner),
                  "store": directory.relative_to(ROOT).as_posix(), "history_revisions": store.verify_history(),
                  "migration_owner_and_weights_bit_exact": True,
                  "correction_preserved_tensors": protected, "least_squares_max_errors": errors,
                  "original_sum_and_integral_retained": True, "restore_exact": True,
                  "parent_goal_quantifiers_immutable": True, "rh_parent": rh, "rh_child": rh_child,
                  "source_calls_for_retained_child": 0, "request_examples": examples,
                  "calculus": calculus, "motion": motion, "checkpoint": restored.request_update,
                  "snapshot": restored.snapshot()}
        write(output / "integration.json", record)
        print(json.dumps({k: record[k] for k in ("status", "owner", "history_revisions", "least_squares_max_errors")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("recount", "integrate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-study-live")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "integrate":
        integrate(args.store.resolve(), args.output)
    else:
        result = {"language": language_recount(), "mathematics": math_recount(),
                  "method": "Recount saved predictions only; no new final-set model inference or selection",
                  "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write(args.output / "recount.json", result)
        print(json.dumps({"language_records": result["language"]["records"], "mathematics": result["mathematics"]}))


if __name__ == "__main__":
    main()
