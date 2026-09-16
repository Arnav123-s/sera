import copy
import uuid

import pytest
import torch

from experiments.grounded_language.model import LanguageR1
from experiments.grounded_language.study import BASE, ROOT, read
from workbench.language_runtime import MIGRATABLE_RUNTIMES, execute_acquired, runtime_source
from workbench.model import Learner, transact
from workbench.storage import Store

pytestmark = pytest.mark.skipif(not (ROOT/"research-continuation/21_grounded_language/qualification.json").exists(),
                                reason="Requires the local prospectively qualified interface")


def test_language_parameters_persist_with_live_state_and_reproved_skills():
    torch.set_num_threads(1)
    directory = ROOT/"runs/math-runtime-tests"/uuid.uuid4().hex
    store = Store(directory)
    original = read(BASE)
    store.commit(copy.deepcopy(original), None)
    result = transact(directory, {"operation": "learn_language", "request_id": "grounded-1",
                                 "text": "subtract three from x then multiply by two to get four modulo eleven"})
    assert result["result"]["status"] == "ACCEPTED" and result["result"]["solutions"] == [5]
    assert result["result"]["execution"]["method"].startswith("preserved guarded program")
    assert result["result"]["execution"]["equation"] == "2 × x − 6 ≡ 4 (mod 11)"
    first = Learner(store.read())
    assert isinstance(first.session.owner, LanguageR1)
    assert first.session.owner is first.solver.neural.owner is first.solver.components["typed"].owner
    assert len(first.session.events) == len(original["interaction"]["events"])
    language_before = {n: t.clone() for n, t in first.session.owner.state_dict().items() if n.startswith("language_")}
    transact(directory, {"operation": "learn_csv", "request_id": "new-data-1", "name": "verification_series",
                         "csv": "timestamp,value\n"+"\n".join(f"{i},{2*i+1}" for i in range(16)), "horizon": 3})
    transact(directory, {"operation": "advance", "request_id": "new-motion-1", "steps": 1, "degrees": 45})
    finite = transact(directory, {"operation": "solve", "request_id": "finite-1", "a": 4, "b": 5, "c": 6})
    assert finite["result"]["solutions"] == [0]
    restored = Learner(store.read())
    assert len(restored.session.events) == len(original["interaction"]["events"])+1
    assert "verification_series" in restored.streams
    assert all(torch.equal(t, restored.session.owner.state_dict()[n]) for n, t in language_before.items())
    assert restored.session.owner.interpret(["subtract three from x then multiply by two to get four modulo eleven"])[0]["labels"] == [1, 2, 3, 4]
    assert store.verify_history() == 5
    damaged = copy.deepcopy(store.read())
    damaged["language"]["checkpoint"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="checkpoint identity"):
        Learner(damaged)
    old = copy.deepcopy(store.read())
    old["language"]["runtime_source"] = next(iter(MIGRATABLE_RUNTIMES))
    with pytest.raises(ValueError, match="interpreter changed"):
        Learner(old)
    migrated = Learner(old, migrate=True)
    assert migrated.snapshot()["owner_sha256"] == old["owner_sha256"]
    assert migrated.language["checkpoint"] == old["language"]["checkpoint"]
    assert migrated.language["runtime_source"] == runtime_source()
    assert old["language"]["runtime_source"] in MIGRATABLE_RUNTIMES
    old["language"]["runtime_source"] = "f"*64
    with pytest.raises(ValueError, match="interpreter changed"):
        Learner(old, migrate=True)
    # Zero coefficients must use the same guarded skill/fallback and preserve
    # the distinction between the two operation orders.
    for labels, solutions in (([0, 0, 2, 0], []), ([1, 0, 2, 0], list(range(11)))):
        execution = execute_acquired(restored, {"labels": labels, "solutions": solutions})
        assert execution["solutions"] == solutions
    with pytest.raises(ValueError, match="disagrees"):
        execute_acquired(restored, {"labels": [1, 2, 3, 4], "solutions": [6]})
