"""Install the development-selected learner and verify the actual shared owner."""

import json

import torch

from sera.session_state import model_identity
from workbench.storage import Store

from .batch import annotate
from .data import ROOT, read, sha, write
from .runtime import RequestSession

OUT = ROOT / "research-continuation/26_stream_curriculum"


def main():
    torch.set_num_threads(1)
    development = read(ROOT / "runs/SC-dev-002/summary.json")
    final = read(ROOT / "runs/SC-final-001/summary.json")
    assert final["selected"] == development["selected"]
    selected = development["selected"]
    if not selected["qualified_annotation_aid"]:
        raise ValueError("Development admission gate was not met")
    directory = ROOT / "runs/sera-requests-live"
    store = Store(directory)
    if store.read() is not None:
        raise ValueError("Preserve the existing live learner")
    parent_pointer = ROOT / "runs/sera-constraints/current.json"
    parent_before = sha(parent_pointer)
    session = RequestSession({key: selected[key] for key in ("path", "sha256")})
    learner = session.base.base.session.learner
    assert session.owner is learner.session.owner is learner.solver.neural.owner
    assert session.owner is learner.solver.components["typed"].owner is session.base.base.session.owner
    before = model_identity(session.owner)
    examples = ("set an alarm for nine am", "what is the weather in london", "add milk to my shopping list")
    frames = [session.ask(f"example-{i + 1}", text) for i, text in enumerate(examples)]
    old_task = session.base.ask("stream-retention-full", "can orbit reach beacon")
    assert old_task["frame"]["actor"] == "orbit" and old_task["frame"]["target"] == "beacon"
    assert "model" in old_task
    assert model_identity(session.owner) == before
    value = session.snapshot()
    restored = RequestSession(session.checkpoint, value)
    assert restored.snapshot() == value
    store.commit(value, None)
    assert store.verify_history() == 1
    output = ROOT / "runs/SC-annotation-demo"
    output.mkdir(parents=True, exist_ok=False)
    source = output / "requests.txt"
    source.write_text("\n".join(examples) + "\n", encoding="utf-8")
    batch = annotate(restored, source, output / "annotations.jsonl")
    assert batch["predictions"] == 3 and batch["input_errors"] == 0
    assert restored.snapshot() == value and sha(parent_pointer) == parent_before
    result = {"status": "PASS", "selected": selected, "owner": before,
              "shared_owner_aliases": True, "exact_session_restore": True,
              "parent_store_pointer_unchanged": parent_before, "old_task": old_task,
              "fixed_illustrations": frames, "batch": batch,
              "saved_store": directory.relative_to(ROOT).as_posix(),
              "development_admission": True, "selection_changed_after_test": False,
              "actions_executed": 0, "integration_source_sha256": sha(__file__)}
    write(OUT / "integration.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("old_task", "fixed_illustrations")}))


if __name__ == "__main__":
    main()
