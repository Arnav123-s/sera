"""Independent finite semantics and preserved-output audit; no training or tuning."""

import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/21_grounded_language"
NUMBERS = {word: i for i, word in enumerate("zero one two three four five six seven eight nine ten".split())}
# Independently written grammar used only by the assessor, never neural training.
PATTERNS = (
    (0, r"(?:multiply|scale) x by (?P<a>\w+) then subtract (?P<b>\w+) to get (?P<c>\w+)"),
    (0, r"take (?P<a>\w+) times x subtract (?P<b>\w+) and obtain (?P<c>\w+)"),
    (0, r"subtract (?P<b>\w+) from the product of (?P<a>\w+) and x equals (?P<c>\w+)"),
    (0, r"the product of x and (?P<a>\w+) minus (?P<b>\w+) equals (?P<c>\w+)"),
    (1, r"subtract (?P<b>\w+) from x then (?:multiply|scale) by (?P<a>\w+) to get (?P<c>\w+)"),
    (1, r"take x minus (?P<b>\w+) (?:multiply|scale) by (?P<a>\w+) and obtain (?P<c>\w+)"),
    (1, r"the product of (?P<a>\w+) and the difference of x and (?P<b>\w+) equals (?P<c>\w+)"),
    (1, r"(?:multiply|scale) the difference of x and (?P<b>\w+) by (?P<a>\w+) equals (?P<c>\w+)"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unpack(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic(text):
    for mode, pattern in PATTERNS:
        match = re.fullmatch(pattern+" modulo eleven", text)
        if match:
            parsed = [NUMBERS[value] if value in NUMBERS else int(value) for value in (match[k] for k in ("a", "b", "c"))]
            assert all(0 <= value <= 10 for value in parsed)
            return [mode, *parsed]
    raise AssertionError("A teaching/evaluation sentence violates the independent grammar")


def solutions(labels):
    order, scale, offset, result = labels
    assert order in (0, 1) and all(type(value) is int and 0 <= value <= 10 for value in (scale, offset, result))
    valid = []
    for candidate in range(11):
        left = (candidate-offset)*scale if order else candidate*scale-offset
        if left % 11 == result:
            valid.append(candidate)
    return valid


def audit_rows(rows, summary):
    for row in rows:
        assert semantic(row["text"]) == row["target"]
        assert row["solutions"] == solutions(row["decoded"])
        assert row["formal_check"] is True
        assert row["translation_correct"] == (row["target"] == row["decoded"])
        assert row["answer_correct"] == (solutions(row["target"]) == solutions(row["decoded"]))
        assert row["accepted"] == (not row["missing_words"] and row["confidence"] >= summary["threshold"])
    assert len(rows) == summary["examples"]
    assert mean(r["translation_correct"] for r in rows) == summary["exact_translation"]
    assert mean(r["answer_correct"] for r in rows) == summary["answer_accuracy"]
    accepted = [r for r in rows if r["accepted"]]
    assert len(accepted) == summary["accepted"]
    assert (mean(r["translation_correct"] for r in accepted) if accepted else None) == summary["accepted_translation_accuracy"]
    return len(rows)


def audit_teaching(record, split):
    allowed = {"train": {2, 3, 4}, "development": {1}}
    for row in record:
        assert semantic(row["text"]) == row["labels"]
        _, a, b, c = row["labels"]
        assert (121*a+11*b+c) % 5 in allowed[split]


def run():
    torch.set_num_threads(1)
    cohort = read(RELEASE/"cohort-protocol.json")
    for name, expected in cohort["sources"].items():
        assert sha(ROOT/name) == expected, name
    rows_checked, checkpoints, comparisons = 0, 0, []
    paired_banks = {}
    for seed in cohort["seeds"]:
        for arm in cohort["arms"]:
            name = f"L10-FINAL-{seed}-{arm}"
            directory = RELEASE/name
            result = read(directory/"result.json")
            assert result["status"] == "COMPLETE" and result["shared_owner_aliases"]
            assert result["protocol_sha256"] == sha(directory/"protocol.json")
            with zipfile.ZipFile(directory/"sources.zip") as archive:
                for source_name, expected in read(directory/"protocol.json")["sources"].items():
                    assert hashlib.sha256(archive.read(source_name)).hexdigest() == expected
            bank_sha = sha(directory/"teaching.json.gz")
            if seed in paired_banks:
                assert paired_banks[seed] == bank_sha
            paired_banks[seed] = bank_sha
            data = unpack(directory/"teaching.json.gz")
            for split, records in data.items():
                audit_teaching(records, split)
            raw = unpack(directory/"evaluation.json.gz")
            for part, records in raw.items():
                rows_checked += audit_rows(records, result["metrics"][part])
                assert all((121*r["target"][1]+11*r["target"][2]+r["target"][3]) % 5 == 0 for r in records)
            for receipt in result["checkpoints"]+[result["selected_checkpoint"]]:
                assert sha(ROOT/receipt["path"]) == receipt["sha256"]
                checkpoints += 1
            selected = torch.load(ROOT/result["selected_checkpoint"]["path"], map_location="cpu", weights_only=True)
            if arm == "interface":
                assert all(key.startswith("language_") for key in selected["delta"])
            assessment_dir = RELEASE/(name+"-assessment")
            assessment = read(assessment_dir/"result.json")
            assert assessment["status"] == "COMPLETE" and assessment["selected_predecessor_unchanged"]
            with zipfile.ZipFile(assessment_dir/"sources.zip") as archive:
                for source_name, expected in read(assessment_dir/"protocol.json")["sources"].items():
                    assert hashlib.sha256(archive.read(source_name)).hexdigest() == expected
            lesson_raw = unpack(assessment_dir/"raw.json.gz")
            for kind in ("word", "wording"):
                for part in ("before", "after", "known_retention"):
                    rows_checked += audit_rows(lesson_raw[kind][part], assessment[kind][part])
                data = unpack(assessment_dir/(kind+"-teaching.json.gz"))
                audit_teaching(data["teaching"], "train")
                audit_teaching(data["development"], "development")
                detail = assessment[kind]
                indices = [i for i, r in enumerate(lesson_raw[kind]["decisions"]) if r["status"] == "ACCEPTED"]
                assert len(indices) == detail["guarded"]["accepted"]
                if indices:
                    assert mean(lesson_raw[kind]["after"][i]["translation_correct"] for i in indices) == detail["guarded"]["translation_accuracy"]
                for i in indices:
                    decision = lesson_raw[kind]["decisions"][i]
                    assert decision["labels"] == lesson_raw[kind]["after"][i]["decoded"]
                    assert decision["solutions"] == solutions(decision["labels"])
                for receipt in detail["checkpoints"]+[detail["selected_checkpoint"]]:
                    assert sha(ROOT/receipt["path"]) == receipt["sha256"]
                    checkpoints += 1
                if kind == "word":
                    assert set(detail["acquisition"]["changed_tensors"]) <= {"language_embedding.weight"}
            retained = unpack(assessment_dir/"retention.json.gz")
            drops = {key: value-retained["successor"]["groups"][key] for key, value in retained["predecessor"]["groups"].items()}
            assert drops == assessment["retention"]["drops"]
            comparisons.append({"seed": seed, "arm": arm, "known": result["metrics"]["known"],
                                "novel": result["metrics"]["novel"], "memory_reset": result["metrics"]["memory_reset"],
                                "training_parameters": result["trainable_parameters"], "train_worker_seconds": result["worker_seconds_this_invocation"],
                                "old_skill_maximum_drop": max(drops.values()), "old_skill_exact_scores": assessment["retention"]["exact_scores_equal"],
                                "word_before": assessment["word"]["before"]["exact_translation"], "word_after": assessment["word"]["after"],
                                "wording_before": assessment["wording"]["before"]["exact_translation"], "wording_after": assessment["wording"]["after"],
                                "wording_old_language": assessment["wording"]["known_retention"]["exact_translation"],
                                "wording_guard": assessment["wording"]["guarded"]})
    result = {"status": "PASS", "independently_parsed_and_checked_predictions": rows_checked,
              "checkpoint_hashes_verified": checkpoints, "models": len(comparisons), "comparisons": comparisons,
              "boundary": "Independent grammar/arithmetic/metric audit of every saved case; hash audit of saved neural checkpoints. This does not retrain models or independently reproduce logits/cross-entropy."}
    path = RELEASE/"independent-audit.json"
    if path.exists():
        raise FileExistsError("Preserve the completed audit")
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    run()
