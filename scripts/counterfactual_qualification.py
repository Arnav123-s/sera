"""Run the explicitly amended completion protocol; preserve every predecessor."""

import argparse

import torch

from experiments.counterfactual_challenge import QUALIFICATION, ChallengedSession, install
from experiments.counterfactual_owner import InquirySession
from experiments.gap_inquiry import ROOT, read, sha, write
from scripts import counterfactual_study as predecessor
from scripts.solution_use import restore as restore_parent
from workbench.storage import Store

PARENT = ROOT / "runs/sera-counterfactual-live"
STORE = ROOT / "runs/sera-counterfactual-qualified-live"
RUN = ROOT / "runs/CI-qualified-001"


def freeze():
    if (QUALIFICATION / "freeze.json").exists():
        raise FileExistsError("The challenge protocol is already frozen")
    predecessor.verify_freeze()
    if (predecessor.OUT / "selection.json").exists() or (predecessor.OUT / "final.json").exists():
        raise ValueError("The predecessor did not have this validation failure")
    paths = [ROOT / "experiments/counterfactual_challenge.py", ROOT / "scripts/counterfactual_qualification.py",
             QUALIFICATION / "PROTOCOL.md"]
    write(QUALIFICATION / "freeze.json", {"files": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
          "parent_current": sha(ROOT / "runs/sera-solution-progress-live/current.json"),
          "inquiry_parent_current": sha(PARENT / "current.json"), "original_freeze": sha(predecessor.OUT / "freeze.json"),
          "reason": "All five narrow completion selectors made wrong exact-consensus claims on development validation; final questions remain unopened"})
    write(RUN / "frontier.json", read(predecessor.RUN / "frontier.json"))


def verify():
    record = read(QUALIFICATION / "freeze.json")
    original = ROOT / "research-continuation/45_counterfactual_inquiry/freeze.json"
    if sha(original) != record["original_freeze"] or sha(PARENT / "current.json") != record["inquiry_parent_current"]:
        raise ValueError("The preserved predecessor changed")
    if sha(ROOT / "runs/sera-solution-progress-live/current.json") != record["parent_current"]:
        raise ValueError("The original solution owner advanced")
    for path, expected in (read(original)["files"] | record["files"]).items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen source changed: " + path)


def restore(store=STORE):
    verify()
    base, growth = restore_parent()
    parent = Store(PARENT).read()
    original = InquirySession(base, parent["state"])
    if original.identity() != parent["owner"]:
        raise ValueError("Original experimental owner changed")
    saved = Store(store).read()
    if saved and saved["freeze"] != sha(QUALIFICATION / "freeze.json"):
        raise ValueError("Changed qualification protocol")
    session = ChallengedSession(original, saved["state"] if saved else None)
    return session, growth


def activate():
    install()
    predecessor.OUT, predecessor.RUN, predecessor.STORE = QUALIFICATION, RUN, STORE
    predecessor.restore_inquiry, predecessor.verify_freeze = restore, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--phase", choices=("train", "validation", "final"))
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.freeze:
        freeze()
    elif args.phase:
        activate()
        predecessor.run_phase(args.phase)
    else:
        raise ValueError("Choose freeze or a declared study phase")


if __name__ == "__main__":
    main()
