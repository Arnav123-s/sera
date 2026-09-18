"""Read-only replay of published historical receipts without private issuer keys.

The archive digest authenticates the published bytes. Replaying their original
cases checks the numerical outcomes. This adapter cannot issue assessments or
grant new authority; live HMAC validation remains in the original runtime.
"""

import hashlib
import json
import math
import zipfile
from contextlib import contextmanager

from experiments.quest_portfolio.assessor import challenge, lower_bound
from experiments.quest_portfolio.common import ROOT, digest

ARCHIVE_SHA256 = "a49c9c8c31167553b317f0ee1c76e2af88bb4b4777d69b54e22165092764eee0"
BRIDGE_WITNESS_SHA256 = "3d7cfd3d0c611fb8dbf3e034174eab4b0d0ccb1fe6dcd9a2bccfe15db1a40802"


def replay(record):
    receipt = record["receipt"]
    rows, truth = challenge(record["seed"], receipt["n"])
    predictions = record["predictions"]
    if (rows != record["cases"] or truth != record["truth"]
            or digest(rows) != receipt["inputs_sha256"]
            or digest(predictions) != receipt["prediction_sha256"]
            or len(predictions) != len(truth)):
        raise ValueError("Published assessment inputs or predictions changed")
    good, wrong, returned = [], [], []
    for index, (actual, expected) in enumerate(zip(predictions, truth, strict=True)):
        if actual is None:
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError("Invalid historical prediction")
        returned.append(index)
        (good if abs(actual - expected) <= 1e-8 else wrong).append(index)
    scopes = sorted({rows[i]["scope"] for i in good}) if not wrong else []
    lower = None if receipt["mode"] == "practice" else lower_bound(len(good), len(truth), receipt["alpha"])
    expected = {"correct": len(good), "wrong": len(wrong), "returned": len(returned),
                "valid_scopes": scopes, "lower_bound": lower,
                "qualified": lower is not None and lower >= .95}
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError("Historical assessment result failed replay")
    return receipt


class PublishedVerifier:
    def __init__(self, root=ROOT):
        path = root / "research-continuation/33_capability_portfolio/research.zip"
        if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Published historical authority archive changed")
        self.receipts = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if "/completed/" in name and name.endswith(".json"):
                    receipt = replay(json.loads(archive.read(name)))
                    prior = self.receipts.setdefault(receipt["id"], receipt)
                    if prior != receipt:
                        raise ValueError("Conflicting published receipt")
        witness = root / "research-continuation/43_knowledge_gaps/publication/historical-bridge-witness.json"
        if hashlib.sha256(witness.read_bytes()).hexdigest() != BRIDGE_WITNESS_SHA256:
            raise ValueError("Published bridge witness changed")
        receipt = replay(json.loads(witness.read_bytes()))
        published = json.loads((root / "research-continuation/34_learning_progress/qualification.json").read_text())
        if receipt != published:
            raise ValueError("Bridge witness differs from the previously published qualification")
        self.receipts[receipt["id"]] = receipt
        self.verified = 0

    def __call__(self, path, request):
        if request.get("action") != "verify":
            raise ValueError("Published replay is read-only; fresh assessments require a live issuer")
        receipts = request["receipts"]
        for receipt in receipts:
            if self.receipts.get(receipt.get("id")) != receipt:
                raise ValueError("Unknown or changed published receipt")
        self.verified += len(receipts)
        return {"verified": len(receipts), "method": "published-byte-identity-and-independent-case-replay",
                "new_assessment_authority": False}


@contextmanager
def historical_replay():
    """Explicit restore-only adapter; never accesses an embedded host path."""
    from experiments.learning_progress import bridge
    from experiments.quest_portfolio import runtime

    verifier = PublishedVerifier()
    original = runtime.call_assessor
    bridge_original = bridge.call_assessor
    runtime.call_assessor = verifier
    bridge.call_assessor = verifier
    try:
        yield verifier
    finally:
        runtime.call_assessor = original
        bridge.call_assessor = bridge_original
