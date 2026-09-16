"""Sequential supervised arms; no parallel numerical workers or completed restarts."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/21_grounded_language"
SEEDS = (151001, 151009, 151027)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "assess"), required=True)
    args = parser.parse_args()
    path = RELEASE/"cohort-protocol.json"
    sources = [*ROOT.glob("experiments/grounded_language/*.py"), *ROOT.glob("scripts/*language*py"),
               *ROOT.glob("tests/test*language*.py")]
    protocol = {"schema": "sera.language-cohort.1", "seeds": SEEDS,
                "arms": ["interface", "shared", "scratch"], "steps": 600, "batch": 32,
                "teaching": 4096, "development": 256, "checkpoint_every": 100,
                "lessons": {"steps": 240, "new_word_examples": 48, "new_wording_examples": 192},
                "primary": "Exact translation and answer correctness on held semantic triples; novel sentence structure is a separate challenge.",
                "intended_release_candidate": "L10-FINAL-151001-interface",
                "candidate_rule": "Fix the first unchanged-core candidate before final outcomes. Require >=.95 known exact translation, >=.99 accepted exact translation, >=.5 coverage, and <=.02 maximum old-skill drop. Reject unrestricted English regardless of these scores.",
                "controls": "Shared-core fine-tuning measures interference; reset shared parameters measure pretrained initialization effects at equal update counts. Reset-history inference probes recurrent-state dependence. Actual time and parameter counts are unequal and reported.",
                "eta": "Fixed supplied update/curriculum. No learning-to-learn improvement claim.",
                "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)}}
    if not path.exists():
        if args.phase != "train":
            raise ValueError("Training protocol must already be frozen")
        path.write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8", newline="\n")
    else:
        saved = json.loads(path.read_text())
        if json.loads(json.dumps(protocol)) != saved:
            raise ValueError("Prospective cohort configuration or sources changed")
    for seed in SEEDS:
        for arm in ("interface", "shared", "scratch"):
            name = f"L10-FINAL-{seed}-{arm}"
            if args.phase == "train":
                destination = RELEASE/name
                arguments = ["--name", name, "--seed", str(seed), "--scope", "interface" if arm == "interface" else "shared",
                             "--steps", "600", "--batch", "32", "--development", "256", "--checkpoint-every", "100", "--final"]
                if arm == "scratch":
                    arguments.append("--scratch")
                script, module = "run_language_bounded.py", "experiments.grounded_language.study"
            else:
                # Forgetting is measured for all arms. Acquisition is reported for all,
                # preventing selective claims about only the best initialization.
                destination = RELEASE/(name+"-assessment")
                arguments = ["--source", name, "--name", destination.name, "--seed", str(seed+10000), "--lesson-steps", "240"]
                script, module = "run_language_assessment_bounded.py", "experiments.grounded_language.assessment"
            if (destination/"result.json").exists():
                print(json.dumps({"preserved_complete": destination.name}), flush=True)
                continue
            if destination.exists():
                raise ValueError(f"Unfinished {destination.name}: explicitly reconcile and resume its checkpoint")
            output = ROOT/"runs"/(destination.name.lower()+"-owned")
            command = [sys.executable, "-X", "utf8", str(ROOT/"scripts"/script), "--seconds", "240",
                       "--output", str(output), "--module", module, "--", *arguments]
            print(json.dumps({"starting": destination.name}), flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
