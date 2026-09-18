from experiments.continuing_growth.common import SKILLS
from experiments.sustained_refinement import admissible, bucket


def record(score, correct):
    return {"scores": [score]*len(SKILLS), "skills": {s: {"correct": correct} for s in SKILLS}}


def test_original_accuracy_floor_prevents_cumulative_drift():
    anchor = record(.4, 20)
    before = record(.41, 19)
    assert admissible(before, record(.42, 19), anchor)
    assert not admissible(before, record(.45, 18), anchor)


def test_macro_gain_cannot_purchase_a_large_strand_regression():
    before = record(.4, 20)
    trial = record(.5, 20)
    trial["scores"][0] = .35
    assert not admissible(before, trial, before)


def test_request_translation_partition_is_deterministic():
    ids = ["request:"+str(i) for i in range(100)]
    assert [bucket(i) for i in ids] == [bucket(i) for i in ids]
    assert set(map(bucket, ids)) == {"train", "final", "future_train", "future_final"}
