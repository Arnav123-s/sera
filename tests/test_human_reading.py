import copy

import pytest
import torch
from torch import nn

from experiments.human_reading.curriculum import candidates, measure
from experiments.human_reading.data import digest, sentences
from experiments.human_reading.model import ReadingR1, encode, lexical
from experiments.human_reading.runtime import PaperText, entries, public_query
from experiments.human_reading.study import metrics


def test_independent_sentence_alignment_and_multiple_human_targets():
    text = "First sentence.  Another one? Final statement!"
    spans = sentences(text)
    assert [text[r["start"]:r["end"]] for r in spans] == [r["text"] for r in spans]
    rows = [{"id": "q", "title": "article", "context_sha256": digest(text), "gold": [1, 2]}]
    result = metrics(rows, torch.tensor([[0., 1., 2.]]))
    assert result["accuracy"] == result["mrr"] == 1


def test_contrastive_candidates_do_not_confuse_duplicate_positive_text():
    rows = [{"id": str(i), "group": "test", "question": "term "+str(i), "target": "definition "+str(i % 8)} for i in range(16)]
    choices = candidates(rows)
    assert choices == candidates(copy.deepcopy(rows))
    for row, choice in zip(rows, choices):
        texts = [rows[i]["target"] for i in choice["indices"]]
        assert len(set(texts)) == 8
        assert texts[choice["gold"]] == row["target"]
    owner = ReadingR1.__new__(ReadingR1)
    nn.Module.__init__(owner)
    owner.reading_embedding = nn.Embedding(8192, 48, padding_idx=0)
    result = measure(owner, rows, choices)
    baseline = measure(owner, rows, choices, baseline=True)
    assert result["pairs"] == baseline["pairs"] == 16
    assert 0 <= result["accuracy"] <= 1


def test_empty_tokens_remain_finite_and_bm25_prefers_source_overlap():
    ids = encode(["", "!!!", "real words"])
    assert ids.shape == (3, 96) and ids.ne(0).any(dim=1).all()
    scored = lexical("What provides energy?", [{"text": "Energy is provided by sunlight."}, {"text": "A chair has legs."}])
    assert scored[0, 1] > scored[1, 1]


def test_arxiv_feed_preserves_identity_and_authorship():
    raw = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/math/9207222v1</id><title>Power sums</title><summary>Original summary.</summary><author><name>A. Author</name></author><published>1992-07-01</published></entry></feed>'''
    record = entries(raw)[0]
    assert record["id"] == "math/9207222v1"
    assert record["authors"] == ["A. Author"]
    assert record["url"].startswith("https://arxiv.org/")
    with pytest.raises(ValueError):
        entries(raw.replace(b"arxiv.org/abs", b"evil.test/abs"))
    with pytest.raises(ValueError):
        entries(b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>')


def test_research_query_is_public_text_and_sources_cannot_execute_commands():
    assert public_query("What is compressed sensing?") == 'all:"compressed" AND all:"sensing"'
    parser = PaperText()
    parser.feed('<p>A source describes assumptions in enough detail to be inspected.</p><script>delete_everything()</script>')
    assert len(parser.parts) == 1
    assert "delete_everything" not in parser.parts[0]
    with pytest.raises(ValueError):
        public_query("!!!")


def test_actual_owner_restore_keeps_applicability_and_open_goal():
    from experiments.human_reading.runtime import ReadingSession

    runtime = ReadingSession()
    runtime.base.assert_owner()
    # This accessor exercises the actual applicability guard after the owner transition.
    runtime.base.learner.legacy.snapshot()
    saved = runtime.snapshot()
    restored = ReadingSession(saved)
    assert saved == restored.snapshot()
    class Unavailable:
        def search(self, *args, **kwargs):
            raise OSError("test interruption")
    goal = runtime.ask("gap-test", "What is sparse recovery?", client=Unavailable())
    assert goal["status"] == "RETAINED_OPEN" and goal["weight_updates"] == 0
    with pytest.raises(ValueError):
        runtime.investigate("gap-test", "Replace the original question", client=Unavailable())
    altered = copy.deepcopy(saved)
    altered["identity"]["owner"] = "changed"
    with pytest.raises(ValueError):
        ReadingSession(altered)
