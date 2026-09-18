"""Independent recount, leakage/retention checks and persisted application verification."""

import copy
import json
from collections import defaultdict

import numpy as np
import torch

from experiments.concept_refinement.audit import languages
from sera.session_state import model_identity
from workbench.storage import Store

from .corpus import CORPUS, DOMAINS
from .data import OUT, ROOT, RUN, normalized, read, sha, write
from .model import ReadingR1, delta, load_curriculum, parent
from .runtime import ReadingSession
from .study import fit


def interval(first, second, seed=2921):
    # Independent article/group bootstrap, preserving all questions within a group.
    groups = defaultdict(list)
    for a, b in zip(first, second, strict=True):
        if a["id"] != b["id"] or a["gold"] != b["gold"]:
            raise AssertionError("Comparison examples changed")
        key = a.get("article", a.get("group"))
        groups[key].append(int(a["correct"])-int(b["correct"]))
    arrays = list(groups.values())
    total, count = np.array([sum(r) for r in arrays]), np.array([len(r) for r in arrays])
    if len(arrays) < 2:
        return {"difference": float(total.sum()/count.sum()), "group_bootstrap_95": None,
                "groups": len(groups), "reason": "One held-out group cannot estimate between-group uncertainty"}
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(arrays), (2000, len(arrays)))
    values = total[index].sum(1) / count[index].sum(1)
    return {"difference": float(total.sum()/count.sum()),
            "group_bootstrap_95": np.quantile(values, [.025, .975]).tolist(), "groups": len(groups)}


def recount(result):
    records = result["predictions"]
    if not records or abs(sum(r["correct"] for r in records)/len(records)-result["accuracy"]) > 1e-12:
        raise AssertionError("Independent accuracy recount failed")
    for row in records:
        gold = row["gold"] if isinstance(row["gold"], list) else [row["gold"]]
        if row["correct"] != (row["prediction"] in gold):
            raise AssertionError("Recorded correctness disagrees with the human target")


def main():
    torch.set_num_threads(1)
    destination = OUT / "integration-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed audit")
    final = read(OUT / "final.json")
    for result in final["results"].values():
        recount(result)
    curriculum = read(OUT / "curriculum-final.json")
    for results in curriculum["results"].values():
        for result in results.values():
            recount(result)
    leakage = {}
    for domain in DOMAINS:
        group_sets, text_sets = [], []
        for split in ("train", "dev", "final"):
            rows = read(CORPUS / f"{domain}-{split}.json")
            group_sets.append({r["group"] for r in rows})
            text_sets.append({normalized(r[k]) for r in rows for k in ("question", "target")})
        for i in range(3):
            for j in range(i):
                assert not group_sets[i] & group_sets[j]
                assert not text_sets[i] & text_sets[j]
        leakage[domain] = {"groups_disjoint": True, "exact_texts_disjoint": True}
    predecessor = parent()
    old = {n: v.clone() for n, v in predecessor.owner.state_dict().items()}
    before_language, ids = languages(predecessor.owner)
    before_math = predecessor.exact_motion([2, 3, 1], "3", "5", "-1")
    subject = next(iter(predecessor.subjects))
    before_empirical = predecessor.predict(subject, [.8, .8, .8, .8])
    runtime = ReadingSession()
    assert all(torch.equal(v, runtime.owner.state_dict()[n]) for n, v in old.items())
    after_language, after_ids = languages(runtime.owner)
    assert ids == after_ids
    for locale, tensors in before_language.items():
        assert all(torch.equal(a, b) for a, b in zip(tensors, after_language[locale], strict=True))
    assert runtime.base.exact_motion([2, 3, 1], "3", "5", "-1") == before_math
    after_empirical = runtime.base.predict(subject, [.8, .8, .8, .8])
    assert before_empirical["parent_owner"] == model_identity(predecessor.owner)
    assert after_empirical["parent_owner"] == model_identity(runtime.owner)
    assert {k: v for k, v in after_empirical.items() if k != "parent_owner"} == {
        k: v for k, v in before_empirical.items() if k != "parent_owner"
    }
    runtime.base.assert_owner()
    restored = ReadingSession(runtime.snapshot())
    assert model_identity(restored.owner) == model_identity(runtime.owner)
    tampered = copy.deepcopy(runtime.snapshot())
    tampered["identity"]["owner"] = "0"*64
    try:
        ReadingSession(tampered)
    except ValueError:
        tamper_rejected = True
    else:
        raise AssertionError("Changed reader owner accepted")
    example = read(RUN / "dev.json")[0]
    answer = runtime.read_passage(example["question"], example["context"], example["source"]+"#"+example["id"])
    assert restored.read_passage(example["question"], example["context"], answer["source"]) == answer
    assert not answer["verified_claim"]
    # A network interruption must preserve the task and all weights.
    class Unavailable:
        def search(self, *args, **kwargs):
            raise OSError("independent audit: source temporarily unavailable")
    before = model_identity(runtime.owner)
    result = runtime.investigate("interruption-check", "What is compressed sensing?", client=Unavailable())
    assert result["status"] == "RETAINED_OPEN" and result["original_goal_preserved"]
    assert before == model_identity(runtime.owner)
    store = Store(ROOT / "runs/HR-audit-session")
    if store.read() is not None:
        raise FileExistsError("Preserve prior audit session")
    store.commit(runtime.snapshot(), None)
    assert store.verify_history() == 1
    assert ReadingSession(store.read()).snapshot() == runtime.snapshot()
    # Independent continuation of a mid-training checkpoint must reproduce final weights.
    selected = read(OUT / "selection.json")["best_trained"]
    saved = torch.load(ROOT / selected["checkpoint"], weights_only=True, map_location="cpu")
    owner = load_curriculum(ReadingR1.attach(parent().owner))
    cached = torch.load(RUN / "features-train.pt", weights_only=True, map_location="cpu")
    labels = torch.tensor([r["gold"][0] for r in read(RUN / "train.json")])
    replay = ROOT / "runs/HR-resume-audit"
    if replay.exists():
        raise FileExistsError("Preserve prior resume audit")
    fit(owner, (cached["x"], cached["mask"]), labels, replay,
        saved["config"]["kind"], saved["config"]["seed"],
        resume=(ROOT / selected["checkpoint"]).with_name("step-0280.pt"))
    assert all(torch.equal(v, delta(owner)[n]) for n, v in saved["delta"].items())
    chosen = final["results"][final["selected"]]
    selected_stage = read(OUT / "curriculum-selection.json")["stage"]
    result = {"status": "PASS", "actual_shared_owner": True, "owner": model_identity(runtime.owner),
              "predecessor_tensors_exact": len(old), "four_language_probes_exact": {k: len(v) for k, v in ids.items()},
              "retained_algebra": before_math, "empirical_forecast_exact": True,
              "checkpoint_resume_exact": True, "restore_exact": True, "tamper_rejected": tamper_rejected,
              "interrupted_goal_preserved": True, "cohort_isolation": leakage,
              "reading_vs_bm25": interval(chosen["predictions"], final["results"]["bm25"]["predictions"]),
              "curriculum_vs_initial": {d: interval(curriculum["results"][selected_stage][d]["predictions"],
                                                   curriculum["results"]["initial"][d]["predictions"]) for d in DOMAINS},
              "example": answer, "final_identity": sha(OUT / "final.json"),
              "curriculum_final_identity": sha(OUT / "curriculum-final.json")}
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"example", "retained_algebra"}}, indent=2))


if __name__ == "__main__":
    main()
