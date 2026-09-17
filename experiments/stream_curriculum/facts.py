"""Account for split overlap, supplied schema, fixed controls and observed errors."""

import json
from collections import Counter

from .data import PREPARED, ROOT, read, token_id, write


def main():
    fit = ROOT / "runs/SC-pilot-001"
    records = read(fit / "after.json")
    teach, dev = records["teaching"]["records"], records["development"]["records"]
    normalize = str.casefold
    texts = {normalize(row["text"]) for row in teach}
    intents = {row["target_intent"] for row in teach}
    novel = [row for row in dev if normalize(row["text"]) not in texts]
    most_common = Counter(row["target_intent"] for row in teach).most_common(1)[0][0]
    controls = {}
    for name, rows in (("teaching", teach), ("development", dev)):
        correct = sum(r["target_intent"] == most_common for r in rows)
        exact = sum(r["target_intent"] == most_common and all(t == "O" for t in r["target_tags"]) for r in rows)
        controls[name] = {"examples": len(rows), "intent_correct": correct,
                          "exact_frames": exact, "intent_accuracy": correct / len(rows),
                          "frame_accuracy": exact / len(rows), "slot_span_f1": 0.0}
    source_rows = [json.loads(line) for line in (PREPARED / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    words = {word.lower() for row in source_rows for word in row["tokens"]}
    vocabulary = read(PREPARED / "vocabulary.json")
    result = {"schema": "sera.stream-data-audit.1", "full_training_rows": len(source_rows),
              "full_training_domains": dict(Counter(r["scenario"] for r in source_rows)),
              "supplied_intents": len(vocabulary["intents"]), "supplied_bio_tags": len(vocabulary["tags"]),
              "training_word_types": len(words), "occupied_hash_buckets": len({token_id(w) for w in words}),
              "pilot_domains": len({r["scenario"] for r in teach}), "pilot_intents": len(intents),
              "development_rows_with_teaching_text": len(dev) - len(novel),
              "development_novel_text_rows": len(novel),
              "novel_text_intent_correct": sum(r["predicted_intent"] == r["target_intent"] for r in novel),
              "development_rows_with_untaught_intent": sum(r["target_intent"] not in intents for r in dev),
              "fixed_control": {"rule": "Most frequent pilot teaching intent; all tokens outside entities",
                                "intent": most_common, "scores": controls},
              "final_test_opened": False}
    write(ROOT / "research-continuation/26_stream_curriculum/data-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
