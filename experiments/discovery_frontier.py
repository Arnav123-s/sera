"""Broaden self-chosen gaps and compose already learned owned relationships."""

import argparse
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.self_chosen import equations as eq
from experiments.self_chosen import language, teaching
from experiments.self_chosen.common import DOMAINS, LANGUAGES, ROOT, digest, read, sha, write
from experiments.self_chosen.runtime import Session, reward_for
from experiments.verified_completion.credit import decode, encode
from workbench.storage import Store

OUT = ROOT / "research-continuation/42_composed_discovery"
RUN = ROOT / "runs/CF-study-001"
STEPS = 384
RUNS = (("learned", 4201), ("learned", 4202), ("balanced", 4201))


def contracts():
    return {"source": sha(Path(__file__)), "protocol": sha(OUT / "PROTOCOL.md")}


def native(question, proposal):
    domain, target, required = question["domain"], proposal.get("target"), set(proposal.get("requires", []))
    if domain == "motion_0":
        alternatives = {"a0": [{"f", "m"}], "v": [{"f", "m", "t", "v0"}],
                        "x": [{"f", "m", "t", "v0", "x0"}]}
        return any(needs <= required for needs in alternatives.get(target, []))
    return False


def credit(question, proposal, receipt, records):
    result = reward_for(question, proposal, receipt, records)
    if receipt["accepted"] and native(question, proposal):
        result.update(total=0., new_connection=0., new_route=0., new_coverage=0., cheaper_execution=0., retain=False,
                      reconciliation="already executable by the retained constant-force route")
    return result


def fit(question, rows, method):
    missing = question["missing"] or []
    missing = [missing] if isinstance(missing, str) else missing
    terms = eq.features({**question, "missing": missing[0] if missing else None})
    terms = [t for t in terms if not any(n in missing for n, _ in t)
             and all(all(row[n] or p >= 0 for n, p in t) for row in rows)]
    if not terms:
        return {"status": "OPEN", "method": method, "reason": "No defined typed feature"}
    matrix = np.array([[float(eq.value(t, row)) for t in terms] for row in rows])
    y = np.array([float(row[question["target"]]) for row in rows])
    scale = np.linalg.norm(matrix, axis=0)
    valid = scale > 1e-12
    terms, matrix, scale = [t for t, good in zip(terms, valid, strict=True) if good], matrix[:, valid], scale[valid]
    if not terms:
        return {"status": "OPEN", "method": method, "reason": "No informative imagined feature"}
    z, calls = matrix/scale, 0
    if method == "dense":
        coef = np.linalg.lstsq(z, y, rcond=1e-10)[0]/scale
        calls = 1
    else:
        active, residue, coef = [], y.copy(), np.zeros(len(terms))
        for _ in range(min(10, len(terms))):
            correlation = abs(z.T @ residue)
            correlation[active] = -1
            chosen = int(correlation.argmax())
            if chosen in active:
                break
            active.append(chosen)
            fitted = np.linalg.lstsq(z[:, active], y, rcond=1e-10)[0]
            calls += 1
            residue = y-z[:, active] @ fitted
            coef[active] = fitted/scale[active]
            if abs(residue).max() < 1e-8:
                break
    rational = [Q(float(c)).limit_denominator(720) if abs(c) > 1e-8 else Q(0) for c in coef]
    chosen = [(t, c) for t, c in zip(terms, rational, strict=True) if c]
    error = max(abs(sum(c*eq.value(t, row) for t, c in chosen)-row[question["target"]]) for row in rows)
    if not chosen or error > Q(1, 1000000):
        return {"status": "OPEN", "method": method, "reason": "No compact exact imagined fit", "imagined_error": str(error), "linear_solves": calls}
    terms, coefficients = zip(*chosen, strict=True)
    return equation_record(question, terms, coefficients, method) | {"imagined_error": str(error), "linear_solves": calls}


def equation_record(question, terms, coefficients, method):
    if any(Q(float(c)).limit_denominator(720) != c for c in coefficients):
        return {"status": "OPEN", "method": method, "reason": "Composed coefficient needs a wider retained decoder",
                "candidate_coefficients": list(map(str, coefficients))}
    return {"status": "CONJECTURE", "target": question["target"], "method": method,
            "terms": [list(map(list, t)) for t in terms], "coefficients": list(map(str, coefficients)),
            "requires": sorted({n for t in terms for n, _ in t}),
            "nonzero": sorted({n for t in terms for n, p in t if p < 0}),
            "canonical": eq.canonical(question["target"], terms, coefficients),
            "execution_cost": sum(1+sum(abs(p) for _, p in t) for t in terms)}


def compose(question, records, owner):
    import sympy as sp
    missing = question["missing"] or []
    missing = [missing] if isinstance(missing, str) else missing
    available = set(eq.layout(question["domain"]))-{question["target"], *missing}
    rules = [r for r in records if r["question"]["domain"] == question["domain"] and "coefficients" in r["proposal"]]
    symbols = {n: sp.Symbol(n) for n in eq.layout(question["domain"])}
    def expression(record):
        values = [Q(float(v)).limit_denominator(720) for v in owner.self_discoveries[record["id"]]]
        if list(map(str, values)) != record["proposal"]["coefficients"]:
            raise ValueError("Composition source weights changed")
        return sum(sp.Rational(c.numerator, c.denominator)*sp.prod(symbols[n]**p for n, p in t)
                   for t, c in zip(record["proposal"]["terms"], values, strict=True))
    for outer in sorted(rules, key=lambda r: (r["proposal"]["execution_cost"], r["id"])):
        if outer["proposal"]["target"] != question["target"]:
            continue
        absent = set(outer["proposal"]["requires"])-available
        if len(absent) != 1:
            continue
        middle = next(iter(absent))
        for inner in sorted(rules, key=lambda r: (r["proposal"]["execution_cost"], r["id"])):
            if inner["proposal"]["target"] != middle or not set(inner["proposal"]["requires"]) <= available:
                continue
            expression_value = sp.expand(sp.cancel(expression(outer).subs(symbols[middle], expression(inner))))
            terms, coefficients, represented = [], [], True
            for item in sp.Add.make_args(expression_value):
                coefficient, factor = item.as_coeff_Mul()
                powers = factor.as_powers_dict()
                if not coefficient.is_Rational or any(s not in symbols.values() or not power.is_Integer for s, power in powers.items() if s != 1):
                    represented = False
                    break
                terms.append(tuple(sorted((str(s), int(power)) for s, power in powers.items() if s != 1 and power)))
                coefficients.append(Q(str(coefficient)))
            if represented and terms:
                return equation_record(question, terms, coefficients, "compose") | {"dependencies": [outer["id"], inner["id"]]}
    return {"status": "OPEN", "method": "compose", "reason": "No applicable two-rule composition in the retained registry"}


def prepare():
    if (RUN / "freeze.json").exists():
        raise FileExistsError("Preserve the frozen frontier")
    RUN.mkdir(parents=True, exist_ok=True)
    parent = Store(ROOT / "runs/sera-self-discovery-live").read()
    write(RUN / "parent.json", parent)
    prior_private = read(ROOT / "runs/SD-study-001/language-private.json")
    excluded = {r["id"] for groups in prior_private.values() for rows in groups.values() for r in rows}
    session = Session(parent["parent"], saved=decode(parent["state"]))
    public, private, sources = {}, {}, []
    vocabulary = session.owner.stream_config["vocabulary"]["intents"]
    from experiments.stream_curriculum.model import encode as tokens
    for locale in LANGUAGES:
        path = ROOT / ("runs/SC-data-002/train.jsonl" if locale == "en-US" else f"runs/SS-language-data/{locale}-train.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups = {"fit": [], "check": [], "final": []}
        for row in sorted(rows, key=lambda r: digest(str(r["id"]))):
            if str(row["id"]) in excluded:
                continue
            bucket = int(digest(str(row["id"]))[:8], 16) % 5
            group = "fit" if bucket < 3 else "check" if bucket == 3 else "final"
            if len(groups[group]) < 768:
                groups[group].append({"id": str(row["id"]), "text": row["text"], "intent": row["intent"]})
        imagined = []
        with torch.no_grad():
            for start in range(0, len(groups["fit"]), 64):
                batch = groups["fit"][start:start+64]
                logits, _ = session.owner.request_logits(tokens([r["text"] for r in batch]))
                imagined.extend({"id": r["id"], "text": r["text"], "prediction": vocabulary[int(v.argmax())], "confidence": float(v.max())}
                                for r, v in zip(batch, logits.softmax(-1), strict=True))
        public[locale], private[locale] = imagined, groups
        sources.append({"locale": locale, "source": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "counts": {k: len(v) for k, v in groups.items()}})
    write(RUN / "language-public.json", public)
    write(RUN / "language-private.json", private)
    write(RUN / "sources.json", {"sources": sources, "excluded_sd_ids": sorted(excluded), "exclusion_includes_translations": True})
    corrected = [r["id"] for r in session.records if native(r["question"], r["proposal"])]
    write(OUT / "credit-reconciliation.json", {"preserved_parent": parent["owner"], "existing_forward_records": corrected,
                                               "action": "preserve valid executable weights; exclude native forward coverage from fresh novelty credit",
                                               "historical_sd_scores_unchanged": True})
    write(RUN / "freeze.json", {"contracts": contracts(), "parent": sha(RUN / "parent.json"), "runs": RUNS, "steps": STEPS,
                               "source_manifest": sha(RUN / "sources.json"), "final_opened": False})
    write(OUT / "freeze.json", read(RUN / "freeze.json"))
    print(json.dumps({"native_records_reconciled": len(corrected), "sources": sources}, indent=2))


def assess(question, proposal, partition="check"):
    if question["domain"] in DOMAINS[:4]:
        return eq.certify(question["domain"], proposal)
    if proposal["status"] != "CONJECTURE":
        return {"accepted": False, "kind": "OPEN_SEARCH", "reason": proposal["reason"]}
    rows = read(RUN / "language-private.json")[question["domain"]][partition]
    covered = [r for r in rows if language.matches(proposal["pattern"], r["text"])]
    good = [r for r in covered if r["intent"] == proposal["target"]]
    return {"accepted": len(covered) >= 10 and len(good)/len(covered) >= .9,
            "kind": "HUMAN_ANNOTATED_ASSOCIATION", "partition": partition, "correct": len(good), "count": len(covered),
            "accuracy": len(good)/len(covered) if covered else None, "verified_ids": [r["id"] for r in good],
            "covered_ids": [r["id"] for r in covered], "counterexamples": [r for r in covered if r not in good],
            "source": sha(RUN / "sources.json"), "scope": "new request groups, specified locale and source corpus",
            "applicability_permission": False}


class FrontierSession(Session):
    def __init__(self, parent, seed=4201, saved=None):
        super().__init__(parent["parent"], saved=decode(parent["state"]))
        self.discovery_parent = parent
        self.parent_count = len(self.records)
        self.public = read(RUN / "language-public.json")
        self.inventory = []
        for domain in DOMAINS[:4]:
            names = list(eq.layout(domain))
            for target in names:
                others = [n for n in names if n != target]
                for count in (1, 2):
                    for omitted in itertools.combinations(others, count):
                        self.inventory.append({"domain": domain, "target": target, "missing": list(omitted),
                                               "id": digest([domain, target, omitted, "CF"]),
                                               "question": f"Predict {target} from retained {domain} knowledge without {', '.join(omitted)}"})
        self.inventory += language.questions(self.public)
        self.events, self.visits = [], {}
        self.recent = dict.fromkeys(DOMAINS, 0.)
        self.seed, self.rng = seed, np.random.default_rng(seed)
        self.owner.self_discovery_config.update(seed=seed, frontier=contracts())
        self.optimizer = torch.optim.Adam(self.owner.self_question_policy.parameters(), lr=.005)
        if saved:
            self.restore(saved)

    def choose(self, arm, phase="main"):
        available = []
        for q in self.inventory:
            if q["domain"] in DOMAINS[:4]:
                known = set(eq.layout(q["domain"]))-{q["target"], *(q["missing"] or [])}
                covered = any(r["question"]["domain"] == q["domain"] and r["proposal"]["target"] == q["target"]
                              and set(r["proposal"]["requires"]) <= known
                              and set(r["proposal"].get("nonzero", [])) <= {"t", "m"} for r in self.records)
                if covered:
                    continue
            available.append(q)
        if not available:
            raise ValueError("Declared distinct question frontier exhausted")
        minimum = min(self.visits.get(q["id"], 0) for q in available)
        available = [q for q in available if self.visits.get(q["id"], 0) == minimum]
        x = self.features(available)
        with torch.no_grad():
            if arm == "balanced" or self.rng.random() < .2:
                counts = {d: sum(e["question"]["domain"] == d for e in self.events) for d in DOMAINS}
                i = min(range(len(available)), key=lambda k: (counts[available[k]["domain"]], available[k]["id"]))
            else:
                scores = self.owner.self_question_policy(x).squeeze(-1).numpy()
                i = int(self.rng.choice(np.flatnonzero(scores >= scores.max()-1e-12)))
        return available[i], x[i]

    def step(self, arm, commit):
        q, x = self.choose(arm)
        seed = int(q["id"][:8], 16)+100003*self.visits.get(q["id"], 0)
        if q["domain"] in DOMAINS[:4]:
            rows = eq.imagine(self.owner, q["domain"], seed)
            proposals = [fit(q, rows, m) for m in ("sparse", "dense")]+[compose(q, self.records, self.owner)]
            imagined = [{k: str(v) for k, v in row.items()} for row in rows]
        else:
            excluded = {tuple(p["pattern"]) for e in self.events if e["question"]["id"] == q["id"] for p in e["proposals"] if "pattern" in p}
            proposals = [language.propose(q, self.public[q["domain"]], m, excluded) for m in ("single", "pair")]
            proposals.append({"status": "OPEN", "method": "compose", "reason": "Empirical associations require their own independent examples"})
            imagined = {"source": sha(RUN / "language-public.json"), "locale": q["domain"]}
        action = teaching.action(self.owner, "unverified")
        pending = {"sequence": len(self.events), "question": q, "proposals": proposals, "imagined": imagined,
                   "goal": self.goal, "parent_owner": self.discovery_parent["owner"], "seed": seed, "action": action,
                   "investigator": digest({k: v.detach().tolist() for k, v in self.owner.self_question_policy.state_dict().items()})}
        pending["commitment"] = digest(pending)
        commit(pending)
        if action != "test":
            raise ValueError("The learned process policy deferred assessment; preserve its committed proposal")
        receipts, total = [], 0.
        for p in proposals:
            receipt = assess(q, p)
            points = credit(q, p, receipt, self.records)
            if points["retain"]:
                identifier = digest([pending["commitment"], p, receipt])
                record = {"id": identifier, "question": q, "proposal": p, "receipt": receipt,
                          "relation": points["relation"], "route": points["route"], "commitment": pending["commitment"],
                          "sequence": len(self.events), "origin": "own execution fit or composition of owned learned rules"}
                weights = list(map(lambda v: float(Q(v)), p["coefficients"])) if "coefficients" in p else [receipt["accuracy"], receipt["count"]]
                self.owner.self_discoveries[identifier] = nn.Parameter(torch.tensor(weights, dtype=torch.float64), requires_grad=False)
                self.records.append(record)
            receipts.append({"receipt": receipt, "credit": points})
            total += points["total"]
        if arm == "learned":
            self.optimizer.zero_grad(set_to_none=True)
            (self.owner.self_question_policy(x).squeeze()-total).square().backward()
            torch.nn.utils.clip_grad_norm_(self.owner.self_question_policy.parameters(), 2.)
            self.optimizer.step()
        event = {**pending, "receipts": receipts, "reward": total, "arm": arm, "features": x.tolist()}
        event["id"] = digest(event)
        self.events.append(event)
        self.visits[q["id"]] = self.visits.get(q["id"], 0)+1
        self.recent[q["domain"]] = .8*self.recent[q["domain"]]+.2*total
        return event

    def state(self):
        return {"contracts": contracts(), "parent_count": self.parent_count, "parent_owner": self.discovery_parent["owner"],
                "inner": super().state()}

    def restore(self, saved):
        # Session.__init__ first restores its own original snapshot.
        if "inner" not in saved:
            return super().restore(saved)
        if saved["contracts"] != contracts() or saved["parent_owner"] != self.discovery_parent["owner"]:
            raise ValueError("Changed frontier source or owner")
        super().restore(saved["inner"])
        self.owner.self_discovery_config.update(frontier=contracts())
        self.parent_count = saved["parent_count"]


def train(arm, seed):
    if (OUT / "selection.json").exists() or read(RUN / "freeze.json")["contracts"] != contracts():
        raise ValueError("Preserve frozen frontier work")
    paths = sorted(RUN.glob(f"{arm}-{seed}-*.pt"))
    session = FrontierSession(read(RUN / "parent.json"), seed=seed, saved=torch.load(paths[-1], weights_only=True) if paths else None)
    if len(session.events) >= STEPS:
        raise FileExistsError("This frontier run is complete")
    def commit(pending):
        path = RUN / "proposals" / f"{arm}-{seed}-{pending['sequence']:04d}.json"
        if path.exists() and read(path) != pending:
            raise ValueError("Resumed candidate differs after evidence access")
        if not path.exists():
            write(path, pending)
    while len(session.events) < STEPS:
        session.step(arm, commit)
        if len(session.events) % 64 == 0:
            torch.save(session.state(), RUN / f"{arm}-{seed}-{len(session.events):04d}.pt")
            print(json.dumps({"arm": arm, "seed": seed, "decisions": len(session.events),
                              "new_records": len(session.records)-session.parent_count}), flush=True)


def select():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Preserve frontier selection")
    candidates = {}
    for arm, seed in RUNS:
        name = f"{arm}-{seed}"
        saved = torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True)
        inner = saved["inner"]
        candidates[name] = {"new_records": len(inner["records"])-saved["parent_count"], "reward": sum(e["reward"] for e in inner["events"]),
                            "checkpoint": sha(RUN / f"{name}-{STEPS:04d}.pt")}
    selected = max(candidates, key=lambda k: (candidates[k]["new_records"], candidates[k]["reward"], k))
    write(OUT / "selection.json", {"selected": selected, "candidates": candidates, "final_opened": False, "contracts": contracts()})


def final():
    if (OUT / "final.json").exists():
        raise FileExistsError("Preserve the frontier final")
    selection = read(OUT / "selection.json")
    results = {}
    for name in selection["candidates"]:
        saved = torch.load(RUN / f"{name}-{STEPS:04d}.pt", weights_only=True)
        records = saved["inner"]["records"][saved["parent_count"]:]
        checks = []
        for r in records:
            domain = r["question"]["domain"]
            result = eq.check_execution(r["proposal"], eq.reference_rows(domain, 42981, 96)) if domain in DOMAINS[:4] else assess(r["question"], r["proposal"], "final")
            checks.append({"id": r["id"], "domain": domain, "result": result})
        exact = [c for c in checks if c["domain"] in DOMAINS[:4]]
        empirical = [c for c in checks if c["domain"] not in DOMAINS[:4]]
        results[name] = {"checks": checks, "exact_pass": all(c["result"]["accepted"] for c in exact),
                         "exact_routes": len(exact), "exact_cases": sum(c["result"]["checked"] for c in exact),
                         "language_confirmed": sum(c["result"]["accepted"] for c in empirical), "language_attempted": len(empirical)}
    write(OUT / "final.json", {"selected": selection["selected"], "results": results,
                               "admitted": results[selection["selected"]]["exact_pass"], "prior_final_reused": False})
    print(json.dumps({k: {a: b for a, b in v.items() if a != "checks"} for k, v in results.items()}, indent=2))


def integrate():
    final = read(OUT / "final.json")
    if not final["admitted"] or not read(OUT / "audit.json")["passed"]:
        raise ValueError("Independent frontier admission is required")
    state = torch.load(RUN / f"{final['selected']}-{STEPS:04d}.pt", weights_only=True)
    session = FrontierSession(read(RUN / "parent.json"), saved=state)
    saved = {"schema": "sera.composed-discovery.1", "parent": session.discovery_parent, "state": encode(state),
             "owner": session.identity(), "audit": sha(OUT / "audit.json"), "final": sha(OUT / "final.json")}
    store = Store(ROOT / "runs/sera-composed-discovery-live")
    if store.read() is not None:
        raise FileExistsError("Preserve the continuing frontier owner")
    store.commit(saved, None)
    write(OUT / "integration.json", {"owner": saved["owner"], "store": "runs/sera-composed-discovery-live",
                                     "records": len(session.records), "same_actual_owner": session.owner is session.base.owner})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "select", "final", "integrate"))
    p.add_argument("--arm", choices=("learned", "balanced"))
    p.add_argument("--seed", type=int, choices=(4201, 4202))
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.action == "train":
        if (args.arm, args.seed) not in RUNS:
            raise ValueError("Use a declared investigator")
        train(args.arm, args.seed)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
