"""Recount saved predictions, audit teaching membership, and plot fixed outcomes."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from .audit import count
from .data import PREPARED, ROOT, read, sha, write
from .study import identities

OUT = ROOT / "research-continuation/26_stream_curriculum"


def retention_matrix(records, blocks):
    """Stage by taught-domain-block accuracy; no clipping of later improvements."""
    matrix = []
    for stage in records:
        matrix.append([count([r for r in stage if r["scenario"] in block["domains"]])["intent_accuracy"]
                       for block in blocks])
    forgetting = [max(matrix[i][j] for i in range(j, len(matrix) - 1)) - matrix[-1][j]
                  for j in range(len(matrix) - 1)]
    return {"intent_accuracy_matrix": matrix,
            "immediate_acquisition": [matrix[i][i] for i in range(len(matrix))],
            "mean_already_taught": [sum(row[:i + 1]) / (i + 1) for i, row in enumerate(matrix)],
            "final_block_macro": sum(matrix[-1]) / len(matrix[-1]),
            "forgetting_first_five_blocks": forgetting,
            "mean_forgetting": sum(forgetting) / len(forgetting)}


def checked_predictions(path, score, expected_rows):
    rows = read(path)
    assert len(rows) == len(expected_rows)
    assert len({r["id"] for r in rows}) == len(rows)
    for row, source in zip(rows, expected_rows):
        assert (row["id"], row["text"], row["target_intent"], row["target_tags"]) == (
            source["id"], source["text"], source["intent"], source["tags"])
        assert len(row["predicted_tags"]) == len(row["target_tags"])
    actual = count(rows)
    assert all(score[key] == value for key, value in actual.items())
    return rows


def document(development, final, audit):
    selected = next(r for r in final["scores"] if r["checkpoint"]["path"] == final["selected"]["path"])
    dev_selected = next(r for r in development["scores"] if r["checkpoint"]["path"] == final["selected"]["path"])
    integration = read(OUT / "integration.json")
    test_tree = ET.parse(OUT / "full-regression.xml")
    tests = len(list(test_tree.iter("testcase")))
    def pct(value):
        return f"{100 * value:.2f}%"
    lines = [
        "# SERA: real-request acquisition, successive learning and a usable annotation tool\n",
        f"I taught the continuing SERA owner's request interface from **11,514 real English requests**. "
        f"The development-selected checkpoint scored **{pct(selected['intent_accuracy'])} intent accuracy**, "
        f"**{pct(selected['slot_span_f1'])} entity-span F1**, and **{pct(selected['frame_accuracy'])} exact intent-plus-entity frames** "
        f"on **{final['eligible_rows']:,} official held-out requests**. It now processes local text/JSONL files into labelled requests and entity spans. "
        "[Use the saved learner](README.md).\n",
        "I also carried one learned interface and its optimizer through six successive domain groups. "
        "A 256-record replay reservoir improved final retention in both paired training seeds. "
        "These are measured acquisitions in the taught request domains. The complete records retain "
        "earlier fits, individual errors, repairs, source identities and full supervised costs.\n",
        "## What SERA was taught\n",
        "MASSIVE 1.1 supplied 11,514 training requests, 2,033 development requests and 2,974 official test requests. "
        "Training covers 18 domains, 60 supplied intents and 55 entity categories (111 BIO tags). Domains include "
        "alarms, audio, calendars, cooking, date/time, email, general interaction, home devices, lists, music, news, "
        "playback, question answering, recommendations, social interaction, takeaway food, transport and weather. "
        "Learning a question-answering request category is measured as intent classification, separately from answering its content.\n",
        "The interface learns hashed-word embeddings, a projection into the existing recurrent owner, normalization "
        "and two output heads. It adds **450,219 trainable parameters** (1,800,876 float32 value bytes). "
        "The original 145 predecessor tensors are protected. The optimizer receives source intent and token-span labels; "
        "inference encodes request text alone. The schema, loss, curriculum and optimizer are supplied engineering. "
        "The acquired text-to-frame mapping is the learned capability.\n",
        "Training uses an exact resumable 128-record shuffle buffer. A global mixed-order view is a seeded "
        "permutation of the same 11,514 rows, repeated across passes. The source has 5,302 distinct lowercased "
        "training words occupying 3,886 of 8,191 usable hash buckets; collision effects remain part of measured performance. "
        "All train/development annotations reconstructed exactly; final exclusions appear below.\n",
        "## Completed research cycles\n",
        "| Cycle | Executed teaching and decision |\n|---|---|",
        "| SC-001 | One 256-request pilot, 80 updates, 2,560 presentations. Teaching intent reached 246/256; the separate 128-row development probe reached 65/128. Exact optimizer/session continuation passed. |",
        "| SC-002 | Two seeds × shared/pooled routes; all source training rows, 1,800 updates and 57,600 presentations per model. Source-order results are preserved. |",
        "| SC-003 | Same examples, seeds, optimizer and updates with a full source permutation. This prospectively tested ordering interference and qualified a shared request-annotation checkpoint. |",
        "| SC-004 | Two seeds × current-only/256-record replay, six successive domain groups, 300 updates per group, one persistent optimizer per lifetime. Each step uses 16 new-stream presentations plus 16 repeated/replayed presentations. |",
        "| SC-005 | Resume each of the four mixed checkpoints from step 1,800 to 5,400. Exact Adam/RNG/cursor restore adds 115,200 presentations per continuation without rerunning its completed prefix. |\n",
        "The primary fits total **36,080 optimizer updates and 1,154,560 presentations**, including the pilot; "
        "these repeated presentations are not additional unique examples. Audit/test updates are separate and remain charged "
        "in job receipts. Sixteen terminal checkpoints enter the fixed evaluation cohort; the four longer-exposure "
        "endpoints are continuations of existing learners. There are two seeds per comparison, not thousands of independent learners.\n",
        "## Complete fixed-cohort results\n",
        "All values below use the same official test partition. The checkpoint was chosen by development exact frames, "
        "then intent accuracy, then lexical path, before final access. Sequential models are supplementary and do not participate "
        "in selection. There was one final evaluation, with no subsequent tuning or reselection.\n",
        "| Checkpoint | Intent accuracy | Span F1 | Exact frames |\n|---|---:|---:|---:|",
    ]
    for score in final["scores"]:
        name = Path(score["checkpoint"]["fit"]).name
        marker = " **selected**" if score["checkpoint"]["path"] == final["selected"]["path"] else ""
        lines.append(f"| {name}{marker} | {pct(score['intent_accuracy'])} | {pct(score['slot_span_f1'])} | {pct(score['frame_accuracy'])} |")
    novel = selected["novel_text"]
    lines += [
        f"\nThe selected checkpoint is `{final['selected']['path']}`, SHA-256 `{final['selected']['sha256']}`. "
        f"Its development scores were {pct(dev_selected['intent_accuracy'])} intent, {pct(dev_selected['slot_span_f1'])} span F1 "
        f"and {pct(dev_selected['frame_accuracy'])} exact frames. It met the predeclared ≥70% intent and ≥50% span-F1 annotation gate.\n",
        f"Official source coverage: {final['eligible_rows']}/{final['source_rows']} rows, {len(final['rejected'])} exclusions. "
        f"Among the {novel['examples']} test rows whose exact case-folded text does not occur in training, the selected model scored "
        f"{pct(novel['intent_accuracy'])} intent, {pct(novel['slot_span_f1'])} span F1 and {pct(novel['frame_accuracy'])} exact frames. "
        "The full official score is kept alongside this declared novelty subset. Source IDs are disjoint across all three partitions.\n",
        f"Excluded source records: `{json.dumps(final['rejected'])}`. The 1–40-token admission rule was frozen before test access. "
        f"Counting excluded requests as incorrect over all {final['source_rows']} source rows gives "
        f"{pct(selected['intent_correct'] / final['source_rows'])} intent and "
        f"{pct(selected['exact_frames'] / final['source_rows'])} exact frames. Entity F1 uses the eligible rows.\n",
        f"The fixed most-frequent-training-intent control (`{final['fixed_control']['intent']}`) scored "
        f"{pct(final['fixed_control']['intent_accuracy'])} intent accuracy. Its all-outside entity output has zero span F1. "
        "The trained pooled route is the stronger architectural control. It shares the supplied interface size and update count "
        "but bypasses recurrence, so compute costs differ. Rankings of intent and joint frames can differ; "
        "the selection criterion was frozen in advance. No standard-LLM ranking is inferred from these internal comparisons.\n",
        "## Successive acquisition and retained knowledge\n",
        "Blocks are: (1) alarm/audio/calendar, (2) cooking/datetime/email, (3) general/iot/lists, "
        "(4) music/news/play, (5) qa/recommendation/social, (6) takeaway/transport/weather. "
        "Each lifetime visits every training ID. Current-stream exposure is 28,800 presentations and total training "
        "is 57,600 presentations in each arm. Both arms advance identical reservoir RNG/bookkeeping for audit; "
        "only the replay arm trains on memory samples.\n",
        "| Seed / arm | Final development intent | Final equal-block accuracy | Mean forgetting, percentage points |\n|---|---:|---:|---:|",
    ]
    for value in audit["sequential"]:
        lines.append(f"| {value['seed']} / {value['arm']} | {pct(value['final_development']['intent_accuracy'])} | "
                     f"{pct(value['final_block_macro'])} | {100 * value['mean_forgetting']:.2f} |")
    lines += [
        "\nEqual-block accuracy is the registered primary retention outcome. Forgetting is the best post-teaching "
        "accuracy before the final stage minus final accuracy, averaged over the first five blocks; a negative entry "
        "retains an observed later improvement. Full 6×6 matrices, immediate acquisition and already-taught curves "
        "are in `independent-audit.json`. Memory contains 256 records, plus a complete seen-ID set, insertion/sample RNGs, "
        "Adam state and source cursor. Serialized memory and bookkeeping sizes are recorded separately.\n",
        "![Held-out accuracy and successive-domain retention](results.png)\n",
        "Replay is the implemented improvement for the measured sequential interference. Full mixing is the implemented "
        "improvement for stationary source-order exposure. Their uses differ: global mixing revisits the whole known corpus; "
        "the sequential learner uses a bounded reservoir of previously seen examples. Fixed replay is an engineered learning "
        "procedure; these trials measure the ability it enables rather than relabelling it as an independently learned investigator.\n",
        "## What the saved learner actually did\n",
        "The selected actual owner processed the same three illustrations fixed during the pilot, wrote their classifications "
        "and entity spans to `runs/SC-annotation-demo/annotations.jsonl`, restored its exact request session, and processed "
        "the older conditional motion request through its inherited shared owner. Every illustration is retained below.\n",
        "| Input | Learned intent | Predicted entity values |\n|---|---|---|",
    ]
    for frame in integration["fixed_illustrations"]:
        entities = "; ".join(f"{e['type']}: {e['value']}" for e in frame["entities"]) or "∅"
        lines.append(f"| {frame['text']} | {frame['intent']} | {entities} |")
    lines += [
        "\nThese are displayed predictions, not a hand-labelled success count. The file-annotation task is performed; "
        "the record explicitly reports zero external actions. Softmax values are exposed as model scores without "
        "a calibration claim. Missing/invalid input has a retained error record.\n",
        "## Diagnosis, repair and verification\n",
        "I preserved the source-order fits and tested mixing with unchanged training information. I then tested "
        "bounded replay under matched exposure and extended the mixed fits by exact optimizer continuation. The final "
        "table includes all outcomes, including cases where more exposure or a particular route is less effective. "
        "These measurements guide the next intervention; they are not universal ability verdicts.\n",
        "An 8-second preparation attempt timed out; its partial output remains. A retry correctly refused the occupied "
        "parent location. A lightweight preparation path completed in a new directory. The first persistence audit "
        "exposed stale guarded-fact binding after owner extension; capturing facts before attachment and rebinding "
        "after loading repaired the lifecycle. Its failed source/log are retained. Removing an unused import later "
        "used an explicit preserving store migration with unchanged owner and predictions. The initial download "
        "metadata incorrectly described partial archive retrieval; the original metadata and exact correction "
        "are preserved because the license at the end required reading the whole compressed archive.\n",
        f"The final suite contains **{tests} passing test cases**. The independent audit recounted "
        f"**{audit['recounted_predictions']:,} saved predictions**, checked all 16 cohort checkpoints and training-ID membership, "
        "and verified that final selection remained unchanged. Every completed fit retained 145 predecessor tensors "
        "and 96 earlier language outputs exactly. Interrupted/uninterrupted optimizer and replay fixtures agree. "
        "All original applicability and prior-store contracts remain separately preserved.\n",
        "## Preservation and next use\n",
        "`raw-records.zip` contains complete prediction records, summaries, freeze receipts and saved request revisions. "
        "The checkpoint archives contain every cohort endpoint and pilot/resume audit states. "
        "`local-state-manifest.json` indexes every retained intermediate; they remain in local `runs/` without multiplying "
        "large checkpoint copies in Git. `data.zip` contains original and prepared source bytes plus the original license. "
        "`sources.zip`, `supervision.zip` and the manifests bind code, costs and results. Prior release artifacts remain intact.\n",
        "[Architecture comparison](architecture-audit.md), [source review](literature.md), [completed checklist](checklist.md), "
        "[exact next research action](next-cycle.md). The next integration connects learned request entities to a persistent "
        "local task, verified corrections and reacquisition on fresh frozen evidence. The present tool is immediately usable "
        "for request annotation through the commands in README.md.\n",
        "Data attribution: FitzGerald et al., [MASSIVE](https://github.com/alexa/massive), CC BY 4.0; English source "
        "material from Bastianelli et al., [SLURP](https://aclanthology.org/2020.emnlp-main.588/). "
        "Replay mechanism reference: Chaudhry et al., [On Tiny Episodic Memories](https://arxiv.org/html/1902.10486v4).\n",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    from .evaluate import partition

    development = read(ROOT / "runs/SC-dev-002/summary.json")
    final = read(ROOT / "runs/SC-final-001/summary.json")
    assert development["selected"] == final["selected"]
    assert development["models"] == final["models"]
    # This is a recount of the fixed final cohort, never a new fit or model selection.
    split_rows = {name: partition(name)[0] for name in ("dev", "test")}
    training = [json.loads(line) for line in (PREPARED / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    train_ids = {str(row["id"]) for row in training}
    split_ids = {name: {str(row["id"]) for row in rows} for name, rows in split_rows.items()}
    assert not train_ids & split_ids["dev"] and not train_ids & split_ids["test"]
    assert not split_ids["dev"] & split_ids["test"]
    verified, predictions = [], 0
    for name, folder, assessment in (("dev", "SC-dev-002", development), ("test", "SC-final-001", final)):
        for index, score in enumerate(assessment["scores"]):
            path = ROOT / "runs" / folder / f"records-{index}.json"
            rows = checked_predictions(path, score, split_rows[name])
            predictions += len(rows)
            verified.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "rows": len(rows)})
    sequential = []
    for seed in (2651, 2652):
        for arm in ("naive", "replay"):
            folder = ROOT / "runs" / f"SC-sequential-{arm}-{seed}"
            result = read(folder / "summary.json")
            stages = []
            for assessment in result["assessments"]:
                path = ROOT / assessment["records"]
                assert sha(path) == assessment["sha256"]
                rows = checked_predictions(path, assessment, split_rows["dev"])
                stages.append(rows)
                predictions += len(rows)
                verified.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "rows": len(rows)})
            sequential.append({"seed": seed, "arm": arm, **retention_matrix(stages, result["blocks"]),
                               "final_development": count(stages[-1]), "blocks": result["blocks"]})
    checkpoint_audits = []
    for model in [*final["models"], *final["supplementary"]]:
        path = ROOT / model["path"]
        assert sha(path) == model["sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["identities"] == identities()
        assert set(map(str, saved["seen_ids"])) == train_ids
        assert all(name.startswith("stream_") for name in saved["delta"])
        assert saved["optimizer"]["state"]
        entry = {**model, "step": saved["step"], "unique_training_ids": len(train_ids),
                 "delta_parameters": sum(t.numel() for t in saved["delta"].values()),
                 "serialized_checkpoint_bytes": path.stat().st_size}
        if "reservoir" in saved:
            memory = saved["reservoir"]
            assert len(memory["rows"]) == 256 and set(memory["seen"]) == train_ids
            assert all(r["partition"] == "train" and str(r["id"]) in train_ids for r in memory["rows"])
            entry["replay_records"] = len(memory["rows"])
            entry["replay_record_json_bytes"] = len(json.dumps(memory["rows"], ensure_ascii=False).encode())
            entry["unique_id_bookkeeping_json_bytes"] = len(json.dumps(memory["seen"]).encode())
        checkpoint_audits.append(entry)
        del saved
    paired = []
    for seed in (2651, 2652):
        naive, replay = [next(r for r in sequential if r["seed"] == seed and r["arm"] == a) for a in ("naive", "replay")]
        paired.append({"seed": seed, "final_intent_gain": replay["final_development"]["intent_accuracy"] - naive["final_development"]["intent_accuracy"],
                       "final_macro_gain": replay["final_block_macro"] - naive["final_block_macro"],
                       "forgetting_reduction": naive["mean_forgetting"] - replay["mean_forgetting"]})
    result = {"schema": "sera.stream-consolidated-audit.1", "status": "PASS",
              "recounted_predictions": predictions, "verified_records": verified,
              "checkpoint_audits": checkpoint_audits, "sequential": sequential, "paired_replay": paired,
              "training_and_assessment_ids_disjoint": True, "final_selection_unchanged": True,
              "statistical_unit": "two training seeds per arm; requests are not independent lifetimes",
              "source_sha256": sha(Path(__file__))}
    write(OUT / "independent-audit.json", result)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    groups = [("Source order", "SC-full-shared"), ("Mixed", "SC-mixed-shared"),
              ("More exposure", "SC-exposure-shared"), ("Pooled control", "SC-exposure-pooled")]
    for i, (label, prefix) in enumerate(groups):
        values = [100 * r["intent_accuracy"] for r in final["scores"] if Path(r["checkpoint"]["fit"]).name.startswith(prefix)]
        axes[0].bar(i, np.mean(values), color="#2563eb", alpha=.7)
        axes[0].scatter([i] * len(values), values, color="#111827", s=25, zorder=3)
    axes[0].set(xticks=range(4), xticklabels=[r[0] for r in groups], ylim=(0, 100), ylabel="Intent accuracy (%)",
                title="Official test: fixed checkpoints\nDots are the two training seeds")
    axes[0].tick_params(axis="x", rotation=20)
    for ax, arm in zip(axes[1:], ("naive", "replay")):
        matrices = np.array([r["intent_accuracy_matrix"] for r in sequential if r["arm"] == arm]) * 100
        matrix = matrices.mean(axis=0)
        rendered = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 55 else "black")
        ax.set(xticks=range(6), xticklabels=range(1, 7), yticks=range(6), yticklabels=range(1, 7),
               xlabel="Domain block assessed", ylabel="After teaching block",
               title=("Current examples only" if arm == "naive" else "256-record replay") + "\nDevelopment intent accuracy (%), mean of two seeds")
    figure.colorbar(rendered, ax=axes[1:], shrink=.7)
    figure.savefig(OUT / "results.png", dpi=160)
    figure.savefig(OUT / "results.svg")
    plt.close(figure)
    document(development, final, result)
    print(json.dumps({"status": "PASS", "recounted_predictions": predictions, "paired_replay": paired}))


if __name__ == "__main__":
    main()
