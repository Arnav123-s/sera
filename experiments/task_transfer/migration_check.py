"""Preserve the deployed descendant while repairing calibration provenance."""

import copy
import json
import zipfile

import numpy as np
import torch

from experiments.grounded_language.data import examples
from experiments.sparse_mechanisms.owner_weights import capture
from sera.session_state import model_identity
from workbench.model import Learner

from .runtime import load, read_csv, save, source
from .study import RELEASE, ROOT, freeze, sha, write


def main():
    torch.set_num_threads(1)
    directory = ROOT/"runs/sera-task-transfer"
    folder = RELEASE/"TT-MIGRATION-001"
    freeze(folder, {"scope": "Repair cross-task calibration reuse without retraining or changing learned numerical tensors",
                    "source_before": "de3b98e839afec3d8e139141abaf1e58dd43fd16567e1a6413d3d93fb94095f4",
                    "source_after": source(), "old_revision": json.loads((directory/"current.json").read_text())})
    old_pointer = json.loads((directory/"current.json").read_text())
    old_revision = json.loads((directory/"revisions"/old_pointer["revision"]).read_text())
    original_files = {p.name: sha(p) for p in (directory/"revisions").glob("*.json")}
    strict_restore_rejected = False
    try:
        load(directory)
    except ValueError as error:
        strict_restore_rejected = "explicitly migrate" in str(error)
    assert strict_restore_rejected
    parent = Learner(json.loads((directory/"parent.json").read_text()))
    parent_state = {k: v.detach().clone() for k, v in parent.session.owner.state_dict().items()}
    language_rows = examples(983017, 64, "final")
    _, original_logits = capture(parent.session.owner, language_rows)
    session = load(directory, migrate=True)
    assert session.tasks == old_revision["tasks"]
    assert all(torch.equal(v, session.owner.state_dict()[k]) for k, v in parent_state.items())
    for name, task in session.tasks.items():
        assert np.array_equal(session.owner.task_readouts[name].weight.detach().numpy(), task["acquisition"]["selected"]["coefficients"])
    assert any(e["operation"] == "observe" and e["used_inputs"] for e in session.events)
    query = [[-.5, .2, .8], [0., 0., 0.], [.8, -.3, .1], [-.2, -.4, -.7]]
    expected = json.loads((RELEASE/"TT-OWNER-001/result.json").read_text())["after_change"]
    assert session.predict("calibration_estimator", query)["outputs"] == expected
    before = session.snapshot()
    batch = read_csv(RELEASE/"TT-OWNER-001/examples/new_task.csv", "teach")
    renamed_rejected = False
    try:
        session.teach("renamed_task", "calibration", batch)
    except ValueError as error:
        renamed_rejected = "whole session" in str(error)
    assert renamed_rejected and before == session.snapshot()
    observation = next(e["used_inputs"][0] for e in session.events if e["operation"] == "observe")
    batch = copy.deepcopy(batch)
    batch["calibration"]["x"][0] = observation
    observation_rejected = False
    try:
        session.teach("observation_reuse", "calibration", batch)
    except ValueError as error:
        observation_rejected = "whole session" in str(error)
    assert observation_rejected and before == session.snapshot()
    owner = model_identity(session.owner)
    revision_name = save(directory, session)
    session = load(directory)
    assert model_identity(session.owner) == owner
    assert session.predict("calibration_estimator", query)["outputs"] == expected
    _, new_logits = capture(session.owner, language_rows)
    assert all(np.array_equal(a, b) for a, b in zip(original_logits, new_logits, strict=True))
    assert all(sha(directory/"revisions"/name) == digest for name, digest in original_files.items())
    assert session.learner.solve(3, 2, 4)["solutions"] == [2]
    with zipfile.ZipFile(folder/"migration-state.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(directory/"current.json", "current.json")
        z.write(directory/"revisions"/revision_name, "revisions/"+revision_name)
    output = {"status": "PASS", "old_owner": old_revision["owner"], "owner": owner,
              "old_revision": old_pointer["revision"], "new_revision": revision_name,
              "old_revisions_preserved": len(original_files), "learned_tensors_unchanged": True,
              "strict_restore_rejected": strict_restore_rejected, "renamed_reuse_rejected": renamed_rejected,
              "historical_observation_reuse_rejected": observation_rejected,
              "rejected_attempts_do_not_mutate_state": True, "language_logits_bitwise_equal": True,
              "outputs_bitwise_equal": True, "formal_route_passed": True, "source": source(),
              "pointer_sha256": sha(directory/"current.json")}
    write(folder/"result.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
