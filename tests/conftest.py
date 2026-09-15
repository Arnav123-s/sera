from pathlib import Path

import pytest
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)


def pytest_collection_modifyitems(items):
    """Local artifact checks are additional to the portable source test suite."""
    root = Path(__file__).resolve().parents[1]
    parent = "runs/SHARED-GG-001/circle/corrected/versions/"
    common = [parent+"v0.pt", parent+"v0.json"]
    required = {
        "test_cross_route_transfer.py": common + [
            "runs/sera-0.5-current/evidence/e8ac1aebb3fc907ee/typed.json.gz",
            "runs/sera-0.5-current/evidence/e8ac1aebb3fc907ee/world.json"],
        "test_instance_transfer.py": common + ["runs/SHARED-GG-001/circle/pre-correction.pt"],
        "test_transfer_integration.py": common + ["runs/SHARED-GG-001/circle/situation.json"],
    }
    for item in items:
        names = required.get(Path(str(item.path)).name, [])
        if any(not (root/name).is_file() for name in names):
            item.add_marker(pytest.mark.skip(reason="Requires preserved local trained artifacts; portable checks remain enabled"))
