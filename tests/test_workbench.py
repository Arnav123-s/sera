import copy
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from experiments.guarded_consolidation.core import Library, Work
from sera.world_graph import SharedOwnerRef
from workbench.model import Learner, transact
from workbench.server import make_handler
from workbench.storage import Store
from workbench.streams import analyze, parse_csv

LOCAL_PARENT = Path(__file__).resolve().parents[1]/"runs/A06-integrated-001/solver/versions/v0.pt"
local_learner = pytest.mark.skipif(not LOCAL_PARENT.exists(), reason="Requires the preserved local trained parent")


def sample(count=40):
    return "timestamp,value\n"+"\n".join(f"{i},{10+.15*i+np.sin(i/5)}" for i in range(count))


def test_prequential_predictions_cannot_see_future_values():
    rows = parse_csv(sample())
    prefix = analyze(rows[:25])
    full = analyze(rows)
    assert full["prequential"][:len(prefix["prequential"])] == prefix["prequential"]
    changed = copy.deepcopy(rows)
    changed[-1]["value"] += 500
    assert analyze(changed)["prequential"][:-1] == full["prequential"][:-1]
    assert analyze(changed)["prequential"][-1]["flagged"]


@pytest.mark.parametrize("content", [sample().replace("0,10.0", "0,nan"), sample().replace("1,", "0,", 1),
                                      sample().replace("2,", "2.5,", 1), "wrong,header\n1,2", sample(4)])
def test_invalid_observations_are_not_admitted(content):
    with pytest.raises(ValueError):
        parse_csv(content)


@local_learner
def test_real_owner_restores_after_data_learning_and_continuing_actions():
    learner = Learner()
    before = {n: t.clone() for n, t in learner.session.owner.state_dict().items() if not n.startswith(("interaction_", "live_"))}
    old_library = learner.library.record()
    result = learner.learn_csv("sensor", sample(), 6)
    assert result["report"]["rows"] == 40
    assert len(learner.session.owner.contexts()) == 1
    context = learner.streams["sensor"]["context"]
    values = [r["value"] for r in learner.streams["sensor"]["observations"]]
    assert learner.session.owner.forecast_context(context, values, 6) == result["report"]["forecast"]
    with pytest.raises(ValueError, match="Stale"):
        Library.restore(old_library, SharedOwnerRef.from_solver(learner.solver), Work(), proof_backend="indexed")
    learner.advance(3, 120.)
    assert learner.solve(4, 5, 6)["answer"] == 0
    snapshot = learner.snapshot()
    restored = Learner(snapshot)
    assert restored.solver.identity() == learner.solver.identity()
    assert restored.streams == learner.streams
    assert restored.solve(0, 0, 0)["solutions"] == list(range(11))
    assert restored.solve(0, 0, 1)["status"] == "no_solution"
    a, b = learner.advance(2, -30.), restored.advance(2, -30.)
    assert a == b
    assert all(torch.equal(t, learner.session.owner.state_dict()[n]) for n, t in before.items())


@local_learner
def test_versions_corrections_and_duplicate_requests_are_preserved(tmp_path):
    first = {"operation": "learn_csv", "request_id": "one", "name": "sensor", "csv": sample(16)}
    transact(tmp_path, first)
    repeated = transact(tmp_path, first)
    assert repeated["result"]["reused_transaction"]
    transact(tmp_path, {**first, "request_id": "two", "csv": sample(20)})
    revised = sample(20).replace("0,10.0", "0,11.0")
    result = transact(tmp_path, {**first, "request_id": "three", "csv": revised})
    assert result["result"]["update_kind"] == "correction"
    assert Store(tmp_path).verify_history() == 3
    assert len(list((tmp_path/"revisions").glob("*.json"))) == 3
    current = json.loads((tmp_path/"current.json").read_text())
    path = tmp_path/"revisions"/current["revision"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        Store(tmp_path).read()


def test_local_http_rejects_foreign_origins_and_preserves_static_ui():
    class Fake:
        token = "test-token"
        def state(self):
            return {"status": "ready"}
        def execute(self, payload):
            return {"received": payload}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Fake()))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"Live data tasks" in urlopen(url).read()
        assert json.load(urlopen(url+"/api/state")) == {"status": "ready"}
        request = Request(url+"/api/execute", data=b'{"operation":"solve"}',
                          headers={"X-Sera-Session": "test-token", "Origin": "https://example.com"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
