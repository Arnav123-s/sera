"""Retention statistics count learning and later improvement without clipping."""

import pytest

from experiments.stream_curriculum.report import retention_matrix


def row(domain, correct):
    return {"scenario": domain, "predicted_intent": "right" if correct else "wrong",
            "target_intent": "right", "predicted_tags": ["O"], "target_tags": ["O"]}


def test_block_retention_includes_later_improvement_and_equal_block_weight():
    stages = [[row("a", True), row("a", False), row("b", False), row("c", False)],
              [row("a", True), row("a", False), row("b", True), row("c", False)],
              [row("a", True), row("a", True), row("b", False), row("c", True)]]
    result = retention_matrix(stages, [{"domains": [name]} for name in ("a", "b", "c")])
    assert result["immediate_acquisition"] == [.5, 1, 1]
    assert result["forgetting_first_five_blocks"] == [-.5, 1.]
    assert result["mean_forgetting"] == .25
    assert result["final_block_macro"] == pytest.approx(2 / 3)
