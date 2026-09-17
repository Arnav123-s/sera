"""Numerical score replay may never change executable interpretation semantics."""

import copy
import json
import runpy
import shutil
from pathlib import Path

import pytest

from experiments.language_inquiry.graph import SCORE_ATOL, SCORE_RTOL, same_interpretation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def interpretation():
    record = json.loads(
        (ROOT / "research-continuation/25_constraint_inquiry/parent.json").read_text()
    )
    return record["graph"]["payload"]["jobs"]["ready"]["interpretation"]


def test_measured_linux_differences_preserve_decisions_and_inputs(interpretation):
    path = ROOT / "research-continuation/28_concept_refinement/publication-hardening"
    diagnostic = json.loads((path / "linux-replay-diagnostic.json").read_text())
    replay = copy.deepcopy(interpretation)
    for row in diagnostic["differences"]:
        if row["path"].startswith("ready/"):
            index = int(row["path"].split("/")[3])
            replay["alternatives"][index]["proposal_score"] = row["observed"]
    original, new = copy.deepcopy(interpretation), copy.deepcopy(replay)
    assert interpretation != replay
    assert same_interpretation(replay, interpretation)
    assert interpretation == original and replay == new


@pytest.mark.parametrize("change", ["action", "alternative_order", "source", "assumptions"])
def test_semantic_changes_are_rejected(interpretation, change):
    altered = copy.deepcopy(interpretation)
    if change == "action":
        altered["frame"]["actions"][0] = -altered["frame"]["actions"][0]
    elif change == "alternative_order":
        altered["alternatives"].reverse()
    elif change == "source":
        altered["parser"] = "different learned weights"
    else:
        altered["assumptions"].append("unverified new premise")
    assert not same_interpretation(interpretation, altered)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01, True, "0.5"])
def test_invalid_scores_are_rejected(interpretation, score):
    altered = copy.deepcopy(interpretation)
    altered["alternatives"][0]["proposal_score"] = score
    assert not same_interpretation(interpretation, altered)


def test_large_score_change_is_rejected(interpretation):
    altered = copy.deepcopy(interpretation)
    score = altered["alternatives"][1]["proposal_score"]
    altered["alternatives"][1]["proposal_score"] += 4 * (SCORE_ATOL + SCORE_RTOL * score)
    assert not same_interpretation(interpretation, altered)


def test_unreviewed_graph_cannot_admit_historical_scores(tmp_path):
    source = ROOT / "experiments/language_inquiry"
    for name in ("compatibility.py", "graph.py"):
        shutil.copyfile(source / name, tmp_path / name)
    approved = runpy.run_path(str(tmp_path / "compatibility.py"))
    assert approved["GRAPH_SCORE_REPLAY_SOURCES"] == {
        "103f05f6fb895f49d7704022f220feadfca9d8fa5593ee67e79f3e1ef2c5cb15"
    }
    with (tmp_path / "graph.py").open("a") as stream:
        stream.write("\n# An unreviewed future change\n")
    assert not runpy.run_path(str(tmp_path / "compatibility.py"))["GRAPH_SCORE_REPLAY_SOURCES"]
