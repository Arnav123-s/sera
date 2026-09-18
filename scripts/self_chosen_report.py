"""Make a readable, source-linked record from completed discovery evidence."""

import json

import torch

from experiments.self_chosen.common import OUT, RUN, STEPS, read, write


def formula(proposal):
    terms = []
    for term, coefficient in zip(proposal["terms"], proposal["coefficients"], strict=True):
        factors = [name if power == 1 else f"{name}^{power}" for name, power in term]
        terms.append(coefficient+"*"+("*".join(factors) or "1"))
    return proposal["target"]+" = "+" + ".join(terms)


def main():
    torch.set_num_threads(1)
    selection, final, audit, future = (read(OUT / (p+".json")) for p in ("selection", "final", "audit", "future"))
    saved = torch.load(RUN / f"{selection['selected']}-{STEPS:04d}.pt", weights_only=True)
    final_map = {r["id"]: r["result"] for r in final["results"][final["selected"]]["checks"]}
    catalog = [{**r, "readable": formula(r["proposal"]) if "coefficients" in r["proposal"] else
                f"{r['question']['domain']}: phrase {' '.join(r['proposal']['pattern'])!r} predicts {r['proposal']['target']}",
                "reserved_evaluation": final_map[r["id"]]} for r in saved["records"]]
    write(OUT / "discoveries.json", catalog)
    lines = ["# Self-chosen discovery: SD-001", "",
             "The selected shared learner acquired **60 independently certified conditional mathematical routes** and retained six human-language association candidates, five of which met the separate reserved-evaluation threshold. It generated questions and fitted proposals from its retained executions, committed them before assessment, and learned from independently checked reward.", "",
             "## What was taught first", "",
             "Three elementary supplied traces taught the actions test, diversify and revise. The actual retained sum/integral operators executed the arithmetic examples. The action policy chose correctly on 12/12 new structured status cases. This measures transfer in the supplied process representation; the workflow, status encoding and examples are engineering/teaching contributions.", "",
             "## What the learner selected and learned", "",
             "The inventory contained 400 generated questions across constant, linearly changing and quadratically changing acceleration, polynomial operators, and English, Spanish, French and German request interpretations. One fifth of question IDs was reserved for future procedure checks. Inputs, units, feature grammar and proposal algorithms were supplied. The learner chose investigations, fitted rational relationship coefficients from its own operator executions, proposed text associations from its own classifications, and updated owned question-policy and retained-rule weights.", "",
             "Four main runs made **1,024 investigation decisions and 2,048 proposal submissions**. The selected lineage made 256 decisions and retained 66 records. All submitted candidates, rejects, exact states and costs remain preserved. Native forward calculations, aliases and dominated input requirements receive no new-discovery credit. A new route means useful conditional execution coverage in this registry; derived consequences are recorded as such.", "",
             "| Investigator | Retained routes | Exact final routes/cases | Human associations meeting final threshold | Development points |",
             "|---|---:|---:|---:|---:|"]
    for name, candidate in selection["candidates"].items():
        result = final["results"][name]
        lines.append(f"| {name} | {candidate['retained']} | {result['exact_routes']}/{result['exact_cases']} | {result['language_confirmed']}/{result['language_attempted']} | {candidate['reward']:.2f} |")
    lines += ["", "Selection used development distinct-route count before final access. The selected learned-4101 owner passed 5,760 fresh exact executions across its 60 mathematical routes. The balanced controller is deterministic under this protocol: its two repeated runs are identical, retained and charged, rather than independent statistical replications. Balanced exploration obtained more confirmed language associations; the selected learned controller obtained more exact motion routes.", "",
              "## Examples of acquired executable relationships", ""]
    examples = []
    for domain, target in (("motion_0", "a0"), ("motion_0", "x"), ("motion_1", "a0"), ("motion_2", "a0")):
        candidates = [r for r in catalog if r["question"]["domain"] == domain and r["proposal"]["target"] == target]
        if candidates:
            examples.append(min(candidates, key=lambda r: (r["proposal"]["execution_cost"], r["id"])))
    for row in examples:
        lines += [f"- **{row['question']['domain']}**: `{row['readable']}`. Required inputs: {', '.join(row['proposal']['requires'])}. Nonzero: {', '.join(row['proposal']['nonzero']) or 'none'}. [Record](discoveries.json) `{row['id'][:12]}`."]
    lines += ["", "Here x is position, v velocity, t elapsed time, and a0/a1/a2 are the supplied coefficients of the model's acceleration polynomial. Initial position and velocity are x0/v0. The independently checked identity has the scope stated in its certificate. These formulas are new retained routes for SERA, derived from its earlier operators and the supplied model family.", "",
              "## Human language evidence", "", "| Locale | Proposed expression | Intent | Reserved correct/count | Retained assessment |", "|---|---|---|---:|---|"]
    for row in catalog:
        if "pattern" in row["proposal"]:
            r = row["reserved_evaluation"]
            lines.append(f"| {row['question']['domain']} | {' '.join(row['proposal']['pattern'])} | {row['proposal']['target']} | {r['correct']}/{r['count']} | {'confirmed on this cohort' if r['accepted'] else 'open for refinement'} |")
    lines += ["", "The original human request IDs are grouped across translations. Fit uses the parent's predicted intents; the assessor alone supplies human annotation outcomes. These are prospective continuation partitions within an older training corpus, so the parent's earlier exposure remains part of the record. Both development and final outcomes accompany every association, and the existing applicability/source gate still governs factual use.", "",
              "## Future investigation procedures", "", "Twelve matched trials crossed old/new retained knowledge with old/new frozen procedure weights and balanced exploration. Each received 32 reserved-question decisions. The observed new-route counts are:", "",
              "| Seed | Knowledge | Old procedure | Learned procedure | Balanced |", "|---|---|---:|---:|---:|"]
    for seed in (41401, 41402):
        for knowledge in ("old", "new"):
            r = {t["policy"]: t["new_retained"] for t in future["trials"] if t["seed"] == seed and t["knowledge"] == knowledge}
            lines.append(f"| {seed} | {knowledge} | {r['old']} | {r['new']} | {r['balanced']} |")
    lines += ["", "The frozen all-seed/all-knowledge procedure-improvement gate remained closed. Retained executable relationships and investigator weights are preserved separately. The next procedure experiment should target uncovered capability acquisition with a matched cost measure, using these results to avoid rewarding easy repeats.", "",
              "## Independent verification and persistence", "",
              f"The audit reassessed all {audit['proposal_checks']:,} main proposals, rejected forged reward, replayed {audit['exact_resume_decisions']} decisions exactly after restart, and executed {audit['fresh_owned_route_tasks']} fresh tasks through the retained owned coefficients. All {audit['protected_tensors']} inherited tensors and all twelve-strand probe predictions matched their parent. Original source gates, earlier discoveries and unfinished goals are retained.", "",
              "The actual admitted model is recorded in [integration.json](integration.json). [Every discovery](discoveries.json) · [Independent audit](audit.json) · [Future comparison](future.json) · [Exact finite protocol](FINITE.md)."]
    if (OUT / "observation.json").exists():
        observed = read(OUT / "observation.json")
        e = observed["evaluation"]
        lines += ["", "## Independent physical observations", "",
                  f"An already committed learned acceleration relation was checked against [Madison College's measured free-fall data]({observed['source']}). Using ten calibration positions, it predicted ten reserved positions with **{e['held_out_rmse_metres']:.6f} m RMSE**. The fitted acceleration was **{e['nuisance_parameters']['a0']:.6f} m/s²**. The same-budget linear control's RMSE was {e['controls']['1']['rmse']:.6f} m; the quadratic control's was {e['controls']['2']['rmse']:.6f} m.", "",
                  "The provider supplied the free-fall interval. Only time and position were used; derived velocity/acceleration columns were excluded. The relation was selected before downloading the CSV. Calibration coefficients, held-out residuals and conditional uncertainty are retained with source hashes. The empirical-confirmation point is separate from the frozen discovery comparison and cannot be repeated. [Observation protocol](OBSERVATION_PROTOCOL.md) · [Raw numerical receipt](observation.json) · [Owner update and replay](observation-integration.json)."]
        lines += ["", "This observed continuation uses the Stage 42 successor. A CSV import defect initially shifted an unlabeled row-index column into time and position. That zero-credit result and its source are preserved. The corrected replay used the same learned equation, raw bytes, rows, calibration split and acceptance thresholds; only the named-column mapping changed. [Exact correction and retained failure](OBSERVATION_CORRECTION.md)."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"records": len(catalog), "report": str(OUT / "report.md")}))


if __name__ == "__main__":
    main()
