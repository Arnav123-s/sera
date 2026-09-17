import copy
import json

import pytest

from experiments.self_study.algebra import check, independent, vector
from experiments.self_study.lessons import expression, knuth
from experiments.self_study.model import FastMap
from experiments.self_study.sources import ROOT, Arxiv, valid_url


@pytest.mark.parametrize("domain,p,q", [("sum", [0, 1], [0, "-1/2", "1/2"]),
                                      ("integral", [1, -2, 3], [0, 1, -1, 1])])
def test_certificate_and_independent_root_bound(domain, p, q):
    assert check(domain, p, q)["accepted"] and independent(domain, p, q)
    q[0] = 1
    assert not check(domain, p, q)["accepted"] and not independent(domain, p, q)


@pytest.mark.parametrize("bad", [True, 1.5, "nan", "1/0", 2**200])
def test_reject_unsafe_rational(bad):
    with pytest.raises((ValueError, ZeroDivisionError)):
        vector([bad])


@pytest.mark.parametrize("bad", ["__import__('os')", "N.__class__", "N**100000", "1/0", "N[0]"])
def test_math_reader_never_executes_source(bad):
    with pytest.raises((ValueError, SyntaxError)):
        expression(bad, [0, 1])


@pytest.mark.parametrize("url", ["http://arxiv.org/abs/1", "https://evil.test/abs/1",
                                  "https://arxiv.org@evil.test/abs/1", "https://arxiv.org:444/abs/1",
                                  "https://arxiv.org/login", "file:///D:/private"])
def test_source_scope(url):
    with pytest.raises(ValueError):
        valid_url(url)


def test_real_source_equations_and_independent_screen():
    client = Arxiv(ROOT / "runs/SS-sources")
    meta, raw = client.paper("math/9207222v1")
    lessons = knuth(meta, raw)
    assert len(lessons) == 6
    for lesson in lessons:
        assert check("sum", lesson["input"], lesson["output"])["accepted"] == independent("sum", lesson["input"], lesson["output"])
    assert all(check("sum", lesson["input"], lesson["output"])["accepted"] for lesson in lessons[:5])
    assert not check("sum", lessons[5]["input"], lessons[5]["output"])["accepted"]


def test_learned_map_and_atomic_invalid_update():
    learner = FastMap()
    assert not independent("sum", [0, 3], learner.propose([0, 3]))
    learner.update([0, 1], [0, "-1/2", "1/2"])
    assert independent("sum", [0, 3], learner.propose([0, 3]))
    previous = copy.deepcopy(learner.state_dict())
    with pytest.raises(ValueError):
        learner.update([0, 1], [float("nan")])
    assert all((previous[n] == v).all() for n, v in learner.state_dict().items())


def test_owned_source_lock_closes_on_failure(tmp_path, monkeypatch):
    import experiments.self_study.sources as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    class Failed:
        def open(self, *args, **kwargs):
            raise OSError("simulated network interruption")
    monkeypatch.setattr(module, "build_opener", lambda *_: Failed())
    client = module.Arxiv(tmp_path / "runs/cache")
    with pytest.raises(OSError):
        client.paper("math/9207222v1")
    assert not (client.directory / "request.lock").exists()
    event = json.loads((client.directory / "acquisitions.jsonl").read_text())
    assert event["status"] == "failed"
