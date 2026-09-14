"""Evaluate ordinary world continuation with explicit retained output capabilities."""

from sera.data import TASKS as LEGACY_TASKS
from sera.evaluation import evaluate
from sera.storage import digest
from sera.typed_learning import TASKS, score_typed
from sera.typed_protocol import typed_suite


def required_capabilities(solver, specs):
    names = {f"world/{spec.identifier}/{facet}" for spec in specs for facet in ("prediction", "control")}
    names.update(f"legacy/{task}" for task in LEGACY_TASKS)
    if "typed" in solver.components:
        names.update(f"typed/{partition}/{task}" for partition in ("test-id", "test-extent", "test-composition") for task in TASKS)
    return names


def evaluate_capabilities(solver, specs, *, seed, samples=1024, typed_samples=128, work=None):
    from sera.connected import evaluate_worlds
    report, scores = evaluate_worlds(solver, specs, seed=seed, samples=samples,
                                     work=work, separate_retention=True)
    legacy, values = evaluate(solver, seed=seed, split="promotion-retention-v2", samples=samples,
                               tasks=range(len(LEGACY_TASKS)))
    report["legacy_retention"] = legacy
    for task, vector in values.items():
        name = f"legacy/{task}"
        scores.capabilities[name] = vector
        scores.dataset_ids[name] = digest([legacy["dataset_id"], task])
    if work is not None:
        work.add("legacy_retention_predictions", samples * len(LEGACY_TASKS))
    if "typed" in solver.components:
        suite, manifest = typed_suite(seed=seed, support_count=0, validation_count=0, test_count=typed_samples)
        report["typed_retention_protocol"] = manifest
        report["typed_retention"] = {}
        for partition, rows in suite.items():
            if not rows:
                continue
            typed, values = score_typed(solver.components["typed"], rows, work=work,
                                        use_programs=True, return_scores=True)
            report["typed_retention"][partition] = typed
            for task, vector in values.items():
                name = f"typed/{partition}/{task}"
                scores.capabilities[name] = vector
                scores.dataset_ids[name] = digest([row.identifier for row in rows if row.task == task])
    report["dataset_id"] = digest([report["dataset_id"], scores.dataset_ids])
    report["admission_contract"] = "separate-retention-v2"
    report["retention_scope"] = (
        "Per-world prediction accuracy and deterministic control success; five legacy outputs (four trained tasks and earliest-binding probe); "
        "seven typed tasks in three partitions when installed. Motion uses exp(-MSE). "
        "Empirical 0.02 score-loss gates; gain bound uses only world objective samples. "
        "This is output retention, not a test of improver quality or every internal component."
    )
    if set(scores.capabilities) != required_capabilities(solver, specs):
        raise ValueError("Retention evaluation coverage is incomplete")
    return report, scores
