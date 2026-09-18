import pytest
import torch

from experiments.book_learning.data import body, dictionary_entries, split
from experiments.book_learning.model import BookR1
from experiments.book_learning.study import metric
from experiments.human_reading.runtime import ReadingSession


def test_complete_dictionary_entries_keep_distinct_senses_and_attribution():
    text = '''*** START OF THE PROJECT GUTENBERG EBOOK TEST ***
FORCE
Force, noun. A physical push or pull; a complete example sentence remains here.

FORCE
Force, noun. The persuasive strength of an argument; another complete sense remains here.

MASS
Mass, noun. An amount of matter; retain the entry and its grammatical metadata.
*** END OF THE PROJECT GUTENBERG EBOOK TEST ***'''
    rows = list(dictionary_entries(text))
    assert len(rows) == 3 and rows[0]["group"] == rows[1]["group"]
    assert rows[0]["text"] != rows[1]["text"]
    assert all(text[r["start"]:r["end"]].strip() == r["text"] for r in rows)
    assert split(rows[0]["group"]) == split(rows[1]["group"])
    with pytest.raises(ValueError):
        body("Catalogue summaries are not original book bodies.")


def test_unknown_words_do_not_inflate_covered_accuracy():
    result = metric(torch.tensor([[0., 5.], [5., 0.]]), torch.tensor([1, 0]))
    assert result["covered_accuracy"] == 1 and result["unknown_rate"] == .5


def test_existing_memory_uses_order_without_changing_predecessor_weights():
    session = ReadingSession()
    before = {k: v.clone() for k, v in session.owner.state_dict().items()}
    owner = BookR1.attach(session.owner, ["unknown", "word"], torch.zeros(2))
    contexts = [["alpha", "beta", "gamma"], ["beta", "alpha", "gamma"]]
    features = owner.book_features(contexts)
    assert torch.allclose(features[0, 1], features[1, 1], atol=1e-6)
    assert not torch.allclose(features[0, 0], features[1, 0], atol=1e-7)
    assert all(torch.equal(v, owner.state_dict()[k]) for k, v in before.items())
    assert all(not p.requires_grad for n, p in owner.named_parameters() if not n.startswith("book_"))
