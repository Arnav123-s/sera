"""Sealed simulated apparatus; independent shot-level phase paths and actuation.

Only this assessor knows the teacher parameters. Learner acquisition sees
command/monitor/outcome records. No learner prediction generates its labels.
"""

import itertools

import numpy as np

from experiments.gap_inquiry import digest, sha
from experiments.intervention_model import ROOT, program

FAMILIES = ("static", "markov", "mixed", "pulse_loss", "unreliable_pulses", "omitted")


def rng_for(*parts):
    return np.random.default_rng(int(digest(list(parts))[:16], 16))


def pulse_program(ticks, count, offset=0):
    if count == 0:
        return {"ticks": ticks, "pulses": []}
    indices = sorted({max(1, min(ticks-1, round(ticks*(i+1)/(count+1))+offset)) for i in range(count)})
    return program({"ticks": ticks, "pulses": indices})


def bank(purpose):
    if purpose == "initial":
        return [pulse_program(n, 0) for n in (2, 4, 6, 8, 10, 12, 14, 16)]
    if purpose == "candidates":
        return [pulse_program(ticks, count) for count in (1, 3, 0, 2, 4) for ticks in (8, 12, 16)]
    if purpose == "selection":
        return [pulse_program(t, c, 1) for t, c in itertools.product((7, 11, 15), (0, 1, 2, 4))]
    if purpose == "adequacy":
        return [pulse_program(t, c, -1) for t, c in itertools.product((6, 9, 13, 16), (0, 1, 2, 3, 4))]
    if purpose == "final":
        # Development exposed overlap between the initial evaluation grid and
        # assessed controls. Freeze genuinely unseen, deduplicated histories.
        known = {digest(p) for name in ("initial", "candidates", "selection", "adequacy") for p in bank(name)}
        result = []
        for ticks, count in itertools.product((5, 7, 9, 11, 13, 15), (0, 1, 2, 3, 4)):
            candidates = [{"ticks": ticks, "pulses": list(indices)} for indices in itertools.combinations(range(1, ticks), count)]
            unseen = [p for p in candidates if digest(p) not in known]
            result.extend(sorted(unseen, key=digest)[:3])
        return result
    if purpose == "wide":
        return [pulse_program(t, c) for t, c in itertools.product((18, 20, 24), (0, 1, 2, 4, 6))]
    raise ValueError("Unknown prospective control bank")


class Source:
    def __init__(self, family, seed, subject):
        if family not in FAMILIES:
            raise ValueError("Unknown sealed source family")
        self.family, self.seed, self.subject = family, seed, subject
        rng = rng_for("world", family, seed)
        self.a, self.b = rng.uniform(.15, .3), rng.uniform(.2, .4)
        self.c = self.d = self.failure = 0.
        if family == "static":
            self.a, self.b = 0., rng.uniform(.3, .6)
        elif family == "markov":
            self.a, self.b = rng.uniform(.3, .6), 0.
        elif family == "pulse_loss":
            self.c = rng.uniform(.07, .14)
        elif family == "unreliable_pulses":
            self.failure = rng.uniform(.4, .7)
        elif family == "omitted":
            self.d = rng.uniform(.06, .12)
        self.identity = digest({"schema": "sera.phase-apparatus.1", "family": family, "seed": seed,
                                "source_code": sha(ROOT / "experiments/intervention_source.py")})

    def observe(self, decision, purpose="acquisition", shots=1024):
        requested = program(decision["program"])
        if decision["source"] != self.identity or decision["subject"] != self.subject:
            raise ValueError("Wrong independent apparatus")
        rng = rng_for("outcome", self.identity, decision["id"], purpose)
        pulses = requested["pulses"]
        applied = rng.random((shots, len(pulses))) >= self.failure
        groups = []
        patterns, counts = np.unique(applied, axis=0, return_counts=True) if pulses else (np.zeros((1, 0), dtype=bool), [shots])
        for pattern, count in zip(patterns, counts, strict=True):
            actual = {"ticks": requested["ticks"], "pulses": [p for p, flag in zip(pulses, pattern, strict=True) if flag]}
            n = int(count)
            sign, signed_area = 1, 0.
            for tick in range(actual["ticks"]):
                if tick in actual["pulses"]:
                    sign = -sign
                signed_area += sign * .25
            total_time, pulse_count = actual["ticks"] * .25, len(actual["pulses"])
            # Integrate independently drawn static detuning; Poisson phase flips
            # realize the irreversible and pulse-induced channels shot by shot.
            detuning = rng.standard_cauchy(n) * self.b
            jumps = rng.poisson(self.a * total_time / 2, n)
            pulse_jumps = rng.poisson((self.c*pulse_count + self.d*pulse_count**2)/2, n)
            coherence = np.cos(detuning * signed_area) * np.where((jumps+pulse_jumps) % 2 == 0, 1., -1.)
            plus = int(np.sum(rng.random(n) < (1+coherence)/2))
            groups.append({"applied": actual, "shots": n, "plus": plus})
        row = {"id": digest([self.identity, decision["id"], purpose, shots]), "subject": self.subject,
               "source": self.identity, "kind": "SIMULATED_OBSERVATION", "purpose": purpose,
               "decision": decision["id"], "requested": requested, "groups": groups}
        row["monitor"] = {"schema": "sera.actuator-monitor.1", "source": "independent-control-log:"+self.identity,
                          "shots": shots, "requested_pulse_shots": shots*len(pulses),
                          "applied_pulse_shots": sum(g["shots"]*len(g["applied"]["pulses"]) for g in groups),
                          "receipt": digest(row)}
        return row

    def truth(self, controls):
        """Independent characteristic-function expectation, actual history fixed."""
        result = []
        for p in controls:
            cuts = np.array([0, *p["pulses"], p["ticks"]], dtype=float) * .25
            intervals = np.diff(cuts)
            signed = intervals @ ((-1.)**np.arange(len(intervals)))
            attenuation = self.a*sum(intervals) + self.b*abs(signed) + self.c*len(p["pulses"]) + self.d*len(p["pulses"])**2
            result.append((1+np.exp(-attenuation))/2)
        return np.asarray(result)

    def assessment(self, purpose):
        rows = []
        for i, control in enumerate(bank(purpose)):
            decision = {"id": digest([self.identity, self.subject, purpose, i]), "program": control,
                        "source": self.identity, "subject": self.subject}
            rows.append(self.observe(decision, purpose, 1024 if purpose == "selection" else 2048))
        return rows
