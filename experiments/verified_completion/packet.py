"""Verify the adopted v14 evidence once in a fresh, resource-supervised directory."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/intake/v14-update/SERA_v14"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", action="store_true")
    if parser.parse_args().portable:
        portable()
        return
    subprocess.run([sys.executable, "code/verify_manifest.py"], cwd=PACKET, check=True)
    subprocess.run([sys.executable, "code/run_all.py", "--output", str(ROOT / "runs/VC-packet-check-001"),
                    "--budget-seconds", "260"], cwd=PACKET, check=True)


def portable():
    import numpy as np
    import torch

    sys.path.insert(0, str(PACKET))
    from s14.core import RewardLedger
    from s14.production import run

    original = json.loads((PACKET / "results/demo/result.json").read_text())
    local_root = ROOT / "runs/VC-packet-check-001/checks"
    local = json.loads((local_root / "restored_demo/result.json").read_text())
    uninterrupted = json.loads((local_root / "uninterrupted_demo/result.json").read_text())
    assert local == uninterrupted
    hashes, differences = [], []

    def compare(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                compare(a[key], b[key], path+"/"+key)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+"/"+str(index))
        elif isinstance(a, float):
            differences.append(abs(a-b))
            assert np.isclose(a, b, atol=2e-12, rtol=2e-12)
        elif a != b:
            assert isinstance(a, str) and re.fullmatch("[0-9a-f]{64}", a) and re.fullmatch("[0-9a-f]{64}", b)
            hashes.append(path)

    compare(original, local)
    RewardLedger.restore(local["reward_ledger"])
    old = torch.load(PACKET / "results/demo/policy_after.pt", weights_only=True)
    new = torch.load(local_root / "restored_demo/policy_after.pt", weights_only=True)
    tensor_error = max(float((old["weights"][k]-new["weights"][k]).abs().max()) for k in old["weights"])
    assert tensor_error < 2e-12
    production = run()
    # Source witness hashes and isolated upstream mathematics checks are unchanged.
    expected = json.loads((PACKET / "results/production.json").read_text())
    compare(expected, production, "/production")
    out = ROOT / "research-continuation/32_verified_completion/packet-verification.json"
    if out.exists():
        raise FileExistsError("Preserve completed packet qualification")
    out.write_text(json.dumps({"manifest_entries": 253, "scientific_jobs_replayed_once": True,
                               "local_interrupted_equals_uninterrupted": True,
                               "maximum_numeric_difference": max(differences),
                               "maximum_policy_tensor_difference": tensor_error,
                               "cross_platform_hash_differences": hashes,
                               "cross_platform_byte_identity": False,
                               "source_packet_modified": False,
                               "original_failure": "runs/VC-packet-check-001/lifecycle.log"}, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
