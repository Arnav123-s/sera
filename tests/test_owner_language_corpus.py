"""Tests for the human corpus, its partitions and the mechanically cut tasks."""

import json
import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from experiments.owner_language import corpus, tasks  # noqa: E402

BUILT = (corpus.CORPUS / "items.jsonl").is_file()
needs_corpus = pytest.mark.skipif(not BUILT, reason="Corpus has not been built in this checkout")


def test_splits_are_a_deterministic_function_of_the_group():
    first = [corpus.split_of("webster", f"word-{index}") for index in range(200)]
    second = [corpus.split_of("webster", f"word-{index}") for index in range(200)]
    assert first == second
    assert set(first) == {"train", "dev", "final"}


def test_split_weights_are_respected_in_the_large():
    counts = {"train": 0, "dev": 0, "final": 0}
    for index in range(20000):
        counts[corpus.split_of("plato", f"section-{index:05d}")] += 1
    assert 0.78 < counts["train"] / 20000 < 0.82
    assert 0.08 < counts["dev"] / 20000 < 0.12
    assert 0.08 < counts["final"] / 20000 < 0.12


def test_gutenberg_licence_block_is_not_taught():
    text = ("front matter and licence noise\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\nreal body here\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\nback matter")
    body, start, end = corpus.body(text)
    assert body.strip() == "real body here"
    assert "licence noise" not in body and "back matter" not in body
    assert 0 < start < end <= len(text)


def test_dictionary_entries_follow_websters_own_structure():
    sample = ("ABATE\nA*bate\", v.t.\n\nDefn: To beat down; to lessen; to diminish the force of a thing.\n"
              "More text on the same paragraph.\n\nNot part of the definition.\n"
              "ABATEMENT\nA*bate\"ment, n.\n\nDefn: The act of abating, or the state of being abated in law.\n")
    entries = list(corpus.dictionary_entries(sample))
    assert [entry["headword"] for entry in entries] == ["ABATE", "ABATEMENT"]
    assert entries[0]["definition"].startswith("To beat down")
    assert "Not part of the definition" not in entries[0]["definition"]


def test_multi_headword_lines_keep_their_variants():
    sample = ("ABASSI; ABASSIS\nA*bas\"si, n.\n\nDefn: A silver coin of Persia worth about twenty cents.\n")
    entry = list(corpus.dictionary_entries(sample))[0]
    assert entry["variants"] == ["ABASSI", "ABASSIS"]
    assert entry["group"] == "ABASSI"


def test_tokeniser_is_deterministic_and_lowercasing():
    assert corpus.tokenise("The Quick, brown fox-trot!") == ["the", "quick", ",", "brown", "fox-trot", "!"]
    assert corpus.tokenise("naïve") == corpus.tokenise("naïve")


@needs_corpus
def test_no_group_appears_in_two_splits():
    seen = {}
    for item in corpus.load_items():
        key = (item["source"], item["group"])
        if key in seen:
            assert seen[key] == item["split"], key
        seen[key] = item["split"]


@needs_corpus
def test_the_transfer_source_is_never_in_a_trained_split():
    for item in corpus.load_items(sources="descartes"):
        assert item["split"] == "transfer"
    assert not corpus.load_items(split="train", sources="descartes")


@needs_corpus
def test_quarantined_sources_produce_no_items():
    sources = {item["source"] for item in corpus.load_items()}
    assert "dailydialog" not in sources


@needs_corpus
def test_vocabulary_is_built_from_training_material_only():
    record = corpus.load_vocabulary()
    counted = sum(1 for item in corpus.load_items() if item["split"] == "train")
    assert record["counted_items"] == counted
    assert record["words"][:5] == ["<pad>", "<unk>", "<eos>", "<def>", "<is>"]


@needs_corpus
def test_evaluation_items_never_reuse_a_training_group():
    training = {(item["source"], item["group"]) for item in corpus.load_items(split="train")}
    for split in ("dev", "final", "transfer"):
        payload = tasks.load_evaluation(split)
        for family, rows in payload["families"].items():
            for row in rows:
                assert (row["source"], row["group"]) not in training, (split, family, row["id"])


@needs_corpus
def test_forced_choice_items_contain_their_own_answer_at_declared_chance():
    payload = tasks.load_evaluation("dev")
    for family in ("cloze", "definition"):
        for row in payload["families"][family][:200]:
            assert row["answer"] in row["candidates"]
            assert len(row["candidates"]) == 8
            assert len(set(row["candidates"])) == 8
            assert row["chance"] == pytest.approx(0.125)


@needs_corpus
def test_evaluation_sets_are_reproducible_from_the_frozen_inputs():
    tokens = tasks.Tokens()
    items = corpus.load_items(split="dev")
    first, _ = tasks.cloze_items(items[:200], tokens)
    second, _ = tasks.cloze_items(items[:200], tokens)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
