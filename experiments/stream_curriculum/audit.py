"""Independently count outcomes and check exact optimizer/session continuation."""

import argparse
import copy
import json

import torch

from sera.session_state import model_identity
from workbench.storage import Store

from .data import PREPARED, ROOT, RowStream, read, sha, write
from .model import apply, delta
from .runtime import RequestSession
from .study import update


def independent_spans(tags):
    result = set()
    position = 0
    while position < len(tags):
        tag = tags[position]
        if tag == "O":
            position += 1
            continue
        label, start = tag[2:], position
        position += 1
        while position < len(tags) and tags[position] == "I-" + label:
            position += 1
        result.add((label, start, position))
    return result


def count(records):
    hits = exact = tp = fp = fn = tokens = correct_tokens = 0
    for row in records:
        predicted, target = independent_spans(row["predicted_tags"]), independent_spans(row["target_tags"])
        hits += row["predicted_intent"] == row["target_intent"]
        exact += row["predicted_intent"] == row["target_intent"] and predicted == target
        tp += len(predicted & target)
        fp += len(predicted - target)
        fn += len(target - predicted)
        correct_tokens += sum(a == b for a, b in zip(row["predicted_tags"], row["target_tags"]))
        tokens += len(row["target_tags"])
    return {"examples": len(records), "intent_correct": hits, "intent_accuracy": hits / len(records),
            "exact_frames": exact, "frame_accuracy": exact / len(records),
            "slot_true_positive": tp, "slot_false_positive": fp, "slot_false_negative": fn,
            "slot_span_f1": 2 * tp / max(1, 2 * tp + fp + fn), "token_accuracy": correct_tokens / tokens}


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default="runs/SC-pilot-001")
    parser.add_argument("--output", default="runs/SC-audit-001")
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit, output = ROOT / args.fit, ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    checked = 0
    for stage in ("before", "after"):
        results = read(fit / f"{stage}.json")
        for value in results.values():
            assert count(value["records"]) == {k: v for k, v in value.items() if k != "records"}
            checked += len(value["records"])
    result = read(fit / "summary.json")
    checkpoint = ROOT / result["checkpoint"]
    session = RequestSession({"path": result["checkpoint"], "sha256": sha(checkpoint)})
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    frames = {}
    # Fixed illustrations, recorded together regardless of the prediction.
    for i, text in enumerate(("set an alarm for nine am", "what is the weather in london", "add milk to my shopping list")):
        frames[str(i)] = session.ask(str(i), text)
    value = session.snapshot()
    restored = RequestSession(session.checkpoint, value)
    assert restored.snapshot() == value
    learner = restored.base.base.session.learner
    assert learner.legacy.snapshot()["payload"]
    assert restored.owner is learner.session.owner is learner.solver.neural.owner
    assert restored.owner is learner.solver.components["typed"].owner is restored.base.base.session.owner
    legacy = restored.base.ask("stream-retention", "can orbit reach beacon")
    assert legacy["frame"]["actor"] == "orbit" and legacy["frame"]["target"] == "beacon"
    assert "model" in legacy
    config = saved["config"]
    left, right = copy.deepcopy(session.owner), copy.deepcopy(session.owner)
    opts, streams = [], []
    for owner in (left, right):
        opt = torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=config["learning_rate"])
        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
        opts.append(opt)
        streams.append(RowStream(PREPARED / "train.jsonl", config["seed"], limit=config["limit"],
                                 buffer_size=config["buffer_size"], state=saved["stream"]))
    for _ in range(2):
        update(left, opts[0], streams[0].take(config["batch_size"]), config["vocabulary"])
    update(right, opts[1], streams[1].take(config["batch_size"]), config["vocabulary"])
    torch.save({"delta": delta(right), "optimizer": opts[1].state_dict(), "stream": streams[1].snapshot(),
                "rng": torch.get_rng_state()}, output / "interrupted.pt")
    loaded = torch.load(output / "interrupted.pt", weights_only=True, map_location="cpu")
    right = copy.deepcopy(session.owner)
    apply(right, loaded["delta"])
    optimizer = torch.optim.Adam([p for p in right.parameters() if p.requires_grad], lr=config["learning_rate"])
    optimizer.load_state_dict(loaded["optimizer"])
    stream = RowStream(PREPARED / "train.jsonl", config["seed"], limit=config["limit"],
                       buffer_size=config["buffer_size"], state=loaded["stream"])
    torch.set_rng_state(loaded["rng"])
    update(right, optimizer, stream.take(config["batch_size"]), config["vocabulary"])
    assert same(delta(left), delta(right))
    assert same(opts[0].state_dict(), optimizer.state_dict())
    assert streams[0].snapshot() == stream.snapshot()
    torch.save({"delta": delta(right), "optimizer": optimizer.state_dict(), "stream": stream.snapshot()}, output / "resumed.pt")
    store = Store(ROOT / "runs/sera-requests")
    if store.read() is not None:
        raise ValueError("Request store already exists; do not overwrite")
    store.commit(value, None)
    assert store.verify_history() == 1
    summary = {"status": "PASS", "independently_counted_records": checked,
               "source_split_stream_tests": "research-continuation/26_stream_curriculum/data-tests.xml",
               "optimizer_resume": "two uninterrupted updates equal one/save/restore/one, including Adam state and buffered stream",
               "audit_only_updates": 4, "audit_only_presentations": 4 * config["batch_size"],
               "request_session_roundtrip_exact": True, "actual_owner_aliases": True,
               "older_constraint_request": legacy, "illustrations": frames,
               "saved_owner": model_identity(session.owner), "store": "runs/sera-requests",
               "primary_model_updated_by_audit": False, "final_test_opened": False}
    write(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("illustrations", "older_constraint_request")}))


if __name__ == "__main__":
    main()
