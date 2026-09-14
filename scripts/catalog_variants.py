"""Label preserved research variants and point to their existing evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from sera.storage import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def stats(values):
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "seeds": len(values)}


def artifact(path, **fields):
    file = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size, **fields}


def main():
    data = read("reports/stage-three-data.json")
    runs = data["runs"]
    entries = []
    source = data["manifest"]["environment"]["source_sha256"]

    def add(identity, name, role, results, paths, scope, source_id=source):
        entries.append({"id": identity, "name": name, "role": role, "status": "preserved-research-variant",
                        "source_sha256": source_id, "results": results,
                        "artifacts": [artifact(path) for path in paths], "scope": scope})

    legacy = read("runs/baseline-v1/summary.json")
    for row in legacy["aggregate"]:
        kind = row["kind"]
        add(f"sequence-{kind}-v1", f"SERA Sequence / {kind}", "sequence-core alternative", row,
            [f"runs/baseline-v1/{kind}-{seed}/checkpoint.pt" for seed in range(3)],
            "Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core.",
            legacy["manifest"]["environment"]["source_sha256"])
    add("delta-world-v1", "SERA Delta World", "compact R1 world model",
        {"long_masked_prediction_accuracy": stats([r["world"]["prediction"]["accuracy"] for r in runs]),
         "memory_reset_accuracy": stats([r["world"]["memory_reset"]["accuracy"] for r in runs]),
         "real_parameters": 17530},
        [f"runs/stage-three-complete/{seed}/solver/versions/v0.pt" for seed in range(3)],
        "Select component r1 from the v0 solver bundle. Four identifying sensors and supplied world identity; deterministic control, not an aliased stochastic belief planner.")
    for routing in ("all", "top1"):
        add(f"reference-world-{routing}-v1", f"SERA Reference / {routing}", "handbook-size R1 experiment",
            {"rank4_accuracy": stats([next(x["prediction"]["accuracy"] for x in r["world"]["reference"] if x["routing"] == routing and x["rank"] == 4) for r in runs]),
             "core_state_bytes": 35840},
            [(f"runs/stage-three-audit/{seed}/reference-all.pt" if routing == "all" else f"runs/stage-three-complete/{seed}/reference-top1.pt") for seed in range(3)],
            "Full rotor/delta/density state. All-routing checkpoint was independently reconstructed in the audit. Trained with fewer updates and smaller batches than Delta World; this is not a matched ranking.")
    for kind, title in (("projective", "Projective Instrument"), ("complex", "Complex Instrument"),
                         ("real", "Real Instrument"), ("hmm", "HMM-22"), ("gru", "GRU-21")):
        add(f"aliased-{kind}-v1", f"SERA Control / {title}", "aliased-event predictor comparison",
            {label: stats([r["predictors"]["models"][kind]["tests"][label]["accuracy"] for r in runs])
             for label in ("ordinary", "repeated", "long-random")},
            [f"runs/stage-three-complete/{seed}/predictor-{kind}.pt" for seed in range(3)],
            "Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited.")
    for procedural in (False, True):
        field = "with_programs" if procedural else "tests"
        add("typed-procedural-v1" if procedural else "typed-neural-v1",
            "SERA Typed + Rules / v1" if procedural else "SERA Typed / v1", "typed capability route",
            {task: stats([r["typed"][field]["ordinary"]["tasks"][task]["score"] for r in runs]) for task in runs[0]["typed"][field]["ordinary"]["tasks"]},
            [f"runs/stage-three-complete/{seed}/solver/versions/v0.pt" for seed in range(3)],
            "Select component typed. Same neural weights with/without two selected arithmetic procedures. Original extended/structure sets duplicate three task families; use evaluation-v2 for corrected transfer evidence.")
    cases = [case for run in runs for case in run["programs"]["cases"]]
    add("program-reuse-v1", "SERA Program Reuse", "world-specific guide and macro-library experiment",
        {f"{guide}/library={library}": {"solved": sum(c["success"] for c in cases if c["guide"] == guide and c["library"] == library),
                                       "cases": sum(c["guide"] == guide and c["library"] == library for c in cases)}
         for guide in ("fixed", "general", "classical") for library in (False, True)},
        [f"runs/stage-three-complete/{seed}/acquired-programs.json" for seed in range(3)],
        "Verified finite transformations. Macro reuse expands primitive action length under the same token budget. The classical program-study guide was not saved as a standalone checkpoint; its results, seed and training recipe remain preserved.")
    add("intervention-selector-v1", "SERA Intervention Selector", "learned eight-method improvement selector",
        {str(i): {"anchor_utility": stats([r["outer"]["generations"][i]["anchor"]["mean_utility"] for r in runs]),
                  "frontier_utility": stats([r["outer"]["generations"][i]["frontier"]["mean_utility"] for r in runs])} for i in range(3)},
        [f"runs/stage-three-complete/{seed}/outer-policy-{generation}.pt" for seed in range(3) for generation in range(3)],
        "Nine saved policies, three seeds. No sustained advantage over the strongest fixed procedure. Policies trained against a fixed base; retained as a finite selector experiment, not evidence of recursive improvement.")
    current = read("runs/evaluation-v2-complete/summary.json")
    for procedural in (False, True):
        route = "typed-v2-procedural" if procedural else "typed-v2-neural"
        add("typed-procedural-v2" if procedural else "typed-neural-v2",
            "SERA Typed + Rules / v2" if procedural else "SERA Typed / v2", "typed capability route",
            {task: stats([r["results"][route]["test-composition"]["tasks"][task]["score"] for r in current["typed"]])
             for task in current["typed"][0]["results"][route]["test-composition"]["tasks"]},
            [f"runs/evaluation-v2-complete/{seed}/typed-v2.pt" for seed in range(3)],
            "Fresh training on semantic-v2 support/validation; withheld composition cases. Same architecture as v1. This cohort remains separate from the admitted world solver.", current["environment"]["source_sha256"])
    catalog = {"schema_version": 1, "purpose": "Stable research variants, not competing AGI claims or a leaderboard across different tasks.",
               "preservation": "Artifacts stay at existing paths. Routes may reference the same bundle; these are not extra independent runs. Nothing is deleted, renamed or copied to make this catalog.",
               "historical_commit": "9072d3a0cfae8ffbfefdadbd1f6cad41e85c56a7", "variants": entries}
    catalog["new_workspaces"] = {
        "runs/evaluation-v2-complete": "Canonical three-seed comparison, semantic datasets, checkpoints, score vectors and all 15 retrospective proposal checks.",
        "runs/evaluation-v2-review": "Preservation baseline, independent replay verification and fresh live-continuation report.",
        "runs/evaluation-v2-smoke": "Four-update development plumbing check with frozen variants; excluded from performance evidence.",
        "runs/evaluation-v2-fresh-smoke": "Four-update from-scratch entry-point check; excluded from performance evidence.",
        "runs/sera-0.4-current": "Active continuation fork with the prior ledger and one new rejected candidate; old 0.3 parent unchanged."
    }
    write_json(ROOT / "research/variants.json", catalog)
    lines = ["# Preserved SERA research variants", "",
             "I retain useful alternatives as named research variants with their original checkpoints, results and limits. A variant can be a component or an evaluation route; it is not automatically a complete architecture. No experiment is discarded because it diverged from the handbook.", "",
             "The [machine-readable registry](variants.json) records source identities, exact artifact hashes and measured results. The [performance report](../reports/evaluation-v2-study.md) compares the corrected typed tests. The [earlier workspace index](workspace-index.md) remains the frozen catalog of the previous audit, including failed and interrupted trials.", "",
             "| Stable variant | Role | Preservation boundary |", "|---|---|---|"]
    lines += [f"| `{r['id']}` — {r['name']} | {r['role']} | {r['scope']} |" for r in entries]
    lines += ["", "The five aliased predictors share one comparison protocol. The eight early sequence cores share another. Reference-size versus compact R1 uses different training budgets. Neural and procedure-assisted results are separate routes. I do not pool these into an overall accuracy ranking.", "",
              "The next integration task uses the compact Delta World direction and one typed stream. The HMM and hybrid remain strong controls to revisit at the appropriate bottleneck. The standalone v2 typed checkpoint is a new research variant; it has not replaced an admitted solver.", ""]
    lines += ["## New workspace roles", "", "| Path | Role |", "|---|---|"]
    lines += [f"| `{path}` | {role} |" for path, role in catalog["new_workspaces"].items()]
    lines += ["", "Checkpoint and raw-data paths in the registry refer to preserved local artifacts. They are not implied to be included in a fresh Git checkout. The public repository contains code, compact evidence, hashes and reproducible recipes.", ""]
    (ROOT / "research/variants.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Cataloged {len(entries)} variants/routes; existing artifacts preserved.")


if __name__ == "__main__":
    main()
