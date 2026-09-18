"""Readable findings and exact evidence links for the broader discovery cycle."""

import json

import torch

from experiments.discovery_frontier import OUT, RUN, STEPS, read, write
from scripts.self_chosen_report import formula


def main():
    torch.set_num_threads(1)
    selection, final, audit = (read(OUT / (name+".json")) for name in ("selection", "final", "audit"))
    saved = torch.load(RUN / f"{selection['selected']}-{STEPS:04d}.pt", weights_only=True)
    records = saved["inner"]["records"][saved["parent_count"]:]
    evaluated = {r["id"]: r["result"] for r in final["results"][final["selected"]]["checks"]}
    catalog = [{**r, "readable": formula(r["proposal"]) if "coefficients" in r["proposal"] else
                f"{r['question']['domain']}: {' '.join(r['proposal']['pattern'])!r} predicts {r['proposal']['target']}",
                "reserved_evaluation": evaluated[r["id"]]} for r in records]
    write(OUT / "discoveries.json", catalog)
    selected = final["results"][final["selected"]]
    compositions = [r for r in catalog if r["proposal"]["method"] == "compose"]
    text = ["# Broader self-chosen discovery: CF-001", "",
            f"The selected successor added **{selected['exact_routes']} exact conditional routes**, checked on **{selected['exact_cases']:,} fresh executions**, and **{selected['language_confirmed']} human-language associations** meeting the reserved threshold out of {selected['language_attempted']} retained candidates. Its {saved['parent_count']} earlier discovery records remain intact.", "",
            "## Why the search expanded", "",
            "The original laboratory goal needed acceleration from position and time. The first portfolio retained no eligible route with exactly those available measurements, so the observation attempt stayed open and downloaded no data. This cycle expanded automatically generated questions to pairs of missing inputs and added composition of already learned rules. The learner's proposals still received no supplied target equations.", "",
            "An additional audit identified one inherited route that duplicated an older constant-force calculation. Its valid executable weights remain preserved; the new reward rule gives that native coverage zero novelty points. Historical SD scores are retained with this explicit reconciliation.", "",
            "## Matched investigations", "",
            "Three runs each completed 384 choices with three proposed approaches per choice: **1,152 investigations and 3,456 proposals**. Two learned-investigator seeds were compared with one deterministic balanced control. Each candidate could fit sparse coefficients, fit dense coefficients, or combine two previously learned owned rules. Every candidate was saved before independent assessment.", "",
            "| Investigator | New retained records | Exact final routes/cases | Human final confirmations | Checked development points |",
            "|---|---:|---:|---:|---:|"]
    for name, c in selection["candidates"].items():
        r = final["results"][name]
        text.append(f"| {name} | {c['new_records']} | {r['exact_routes']}/{r['exact_cases']} | {r['language_confirmed']}/{r['language_attempted']} | {c['reward']:.2f} |")
    text += ["", "Selection was fixed from development new-route count, then checked reward, before final access. The final keeps that selection. Human association results retain their specific corpus and scope; they do not grant a source or interpretation override.", "",
             "## Equations the learner proposed", ""]
    for domain, target in (("motion_0", "a0"), ("motion_1", "a1"), ("motion_2", "a2"), ("polynomials", "c0")):
        choices = [r for r in catalog if r["question"]["domain"] == domain and r["proposal"]["target"] == target]
        choices.sort(key=lambda r: (set(r["proposal"]["requires"]) != {"x", "x0", "v0", "t"}, r["proposal"]["execution_cost"], r["id"]))
        for row in choices[:2]:
            text.append(f"- **{domain}**: `{row['readable']}`; inputs {', '.join(row['proposal']['requires'])}; nonzero {', '.join(row['proposal']['nonzero']) or 'none'}; method {row['proposal']['method']}; record `{row['id'][:12]}`.")
    text += ["", f"{len(compositions)} newly retained records came from explicit two-rule composition. Their dependency IDs point to previously retained coefficient weights. Alternative fitted programs remain in the same portfolio. The supplied variable names and polynomial families describe the modeled world; the learned output consists of selected investigations, coefficients, compositions, scoped readouts and updated investigation weights.", "",
             "## Language and retained abilities", "",
             "All request IDs used anywhere in SD-001 were excluded across translations. This left fresh English groups; the retained Spanish, French and German source slices were already fully covered by that exclusion. They received no new teaching in CF-001. The earlier four-language model and the original source gates remain preserved. [Exact source counts and exclusions](freeze.json) accompany the raw source manifest in the archive.", "",
             f"Independent audit reassessed all {audit['proposal_checks']:,} submissions, checked composition dependencies, replayed 64 decisions exactly, and executed {audit['fresh_owned_route_tasks']} fresh tasks through the owned coefficients. All {audit['protected_tensors']} inherited tensors and all twelve-strand probe predictions were retained exactly.", "",
             "## Original goal, costs and reuse", "",
             "The preserved laboratory goal resumes with the qualified successor when its portfolio contains the required relation. Its original calibration/held-out split and numerical thresholds remain fixed. [Observation receipt](../41_self_chosen_discovery/observation.json) · [Observed owner and independent replay](../41_self_chosen_discovery/observation-integration.json).", "",
             "[Run the saved learner](README.md) · [All new discoveries](discoveries.json) · [Final evaluation](final.json) · [Audit](audit.json) · [Full local costs](costs.json) · [Earlier credit reconciliation](credit-reconciliation.json). All unsuccessful candidates and source identities are preserved alongside the selected state."]
    (OUT / "report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps({"selected": final["selected"], "new_records": len(records), "compositions": len(compositions)}))


if __name__ == "__main__":
    main()
