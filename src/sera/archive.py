"""Measured specialist retrieval from immutable lineages using development evidence only."""

from __future__ import annotations

from sera.storage import digest, write_json


def useful_archive(store, evaluator, *, dataset_id, split="development", work=None):
    if split not in {"development", "validation"} or not dataset_id:
        raise ValueError("Archive selection requires declared development evidence")
    records = []
    for path in sorted((store.root / "versions").glob("v*.json")):
        solver = store.load(path.stem)
        scores = evaluator(solver)
        if not scores or any(not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("Archive competence scores must be finite and bounded")
        size = (path.with_suffix(".pt")).stat().st_size
        records.append({"version": path.stem, "identity": solver.identity(), "scores": scores,
                        "checkpoint_bytes": size})
        if work is not None:
            work.add("archive_checkpoint_bytes_read", size)
            work.add("archive_versions_evaluated")
    tasks = set(records[0]["scores"])
    if any(set(row["scores"]) != tasks for row in records):
        raise ValueError("Archive versions must share a comparison task set")
    useful = []
    for row in records:
        dominated = any(other["checkpoint_bytes"] <= row["checkpoint_bytes"] and
                        all(other["scores"][task] >= row["scores"][task] for task in tasks) and
                        (other["checkpoint_bytes"] < row["checkpoint_bytes"] or
                         any(other["scores"][task] > row["scores"][task] for task in tasks)) for other in records)
        if not dominated:
            useful.append(row)
    selected = {task: max(useful, key=lambda row: (row["scores"][task], -row["checkpoint_bytes"]))["version"]
                for task in sorted(tasks)}
    result = {"schema_version": 1, "split": split, "dataset_id": dataset_id,
              "versions": records, "useful_versions": [row["version"] for row in useful],
              "specialists": selected, "current": store.current_record()["version"],
              "storage_bytes": sum(row["checkpoint_bytes"] for row in records)}
    result["selection_id"] = digest(result)
    write_json(store.root / "useful-archive.json", result)
    return result


def retrieve_specialist(store, archive, task):
    body = {key: value for key, value in archive.items() if key != "selection_id"}
    if digest(body) != archive["selection_id"] or archive["split"] not in {"development", "validation"}:
        raise ValueError("Archive selection integrity failure")
    version = archive["specialists"][task]
    solver = store.load(version)
    record = next(row for row in archive["versions"] if row["version"] == version)
    if solver.identity() != record["identity"]:
        raise ValueError("Archive specialist identity changed")
    return solver
