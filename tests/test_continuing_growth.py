import copy

import pytest

from experiments.continuing_growth.common import SKILLS, digest
from experiments.continuing_growth.data import partition
from experiments.continuing_growth.model import credit
from experiments.continuing_growth.runtime import validate_events


def scores(value):
    return {"scores": [value]*len(SKILLS), "accuracy": [value]*len(SKILLS)}


def test_forget_relearn_does_not_reissue_discovery_points():
    before, after = scores(.4), scores(.5)
    fresh = credit(before, after, [.4]*len(SKILLS), 0)
    repeated = credit(before, after, fresh["highwater"], 0)
    assert fresh["new_highwater"] > 0 and fresh["cross_strand_highwater"] > 0
    assert repeated["new_highwater"] == repeated["cross_strand_highwater"] == 0
    assert credit(after, before, fresh["highwater"], 0)["reward"] < 0


def test_translation_groups_do_not_cross_partitions():
    english = partition(["request:"+str(i) for i in range(200)])
    spanish = partition(["request:"+str(i) for i in range(0, 200, 3)])
    assert all(english[g] == name for g, name in spanish.items())
    assert set(english.values()) == {"train", "probe", "final", "future_train", "future_final"}


def event():
    e = {"sequence": 0, "before": scores(.4), "after": scores(.5), "skill": SKILLS[0],
         "knowledge_before": "old", "knowledge_after": "new", "accepted": True}
    e["credit"] = credit(e["before"], e["after"], [.4]*len(SKILLS), 0)
    e["id"] = digest(e)
    return e


def test_stale_or_duplicate_credit_is_rejected():
    first = event()
    validate_events([first], first["credit"]["highwater"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_events([first, first], first["credit"]["highwater"])
    second = copy.deepcopy(first)
    second.pop("id")
    second["sequence"] = 1
    second["id"] = digest(second)
    with pytest.raises(ValueError, match="Stale"):
        validate_events([first, second], first["credit"]["highwater"])


def test_signed_credit_and_highwater_cannot_be_forged():
    e = event()
    e["credit"]["reward"] += 1
    e.pop("id")
    e["id"] = digest(e)
    with pytest.raises(ValueError, match="reward"):
        validate_events([e], e["credit"]["highwater"])


def test_nonfinite_or_missing_strand_scores_are_rejected():
    with pytest.raises(ValueError, match="Finite"):
        credit(scores(.4), scores(float("nan")), [.4]*len(SKILLS), 0)
