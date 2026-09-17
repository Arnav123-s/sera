"""Explicitly admitted, parameter-preserving interpreter migrations.

The original acquisition source remains the weight artifact's provenance.
The release retains every prior source byte and checks identical learned tensors
and derived inquiry results before saving a new implementation revision.
"""

import hashlib
from pathlib import Path


def reviewed(name, identity):
    path = Path(__file__).parent/name
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == identity


TRAINING_SOURCES = ({"1708da71bb6a7a5fe442691a0bd4f9f92e65c562e06afe37eb49b859dd682bc4"}
    if reviewed("model.py", "9f02ebd332e9b25ad6bc6dfbf86e4b04e621678ac9c67ecddba54e18fd25df1d")
    and reviewed("data.py", "0fe4b3d8e48511285b8def83c3843530b38392fe30c6f782d27e900fe9916983") else set())
RUNTIME_SOURCES = ({"1d44b9114bb829d5e3b47333dfc213c1213beba05503042e4128649ffeaf1961"}
    if reviewed("runtime.py", "80428f122a3eb64915910741b95369abef72e05d8e7edc2b3d88e2ba6587d702") else set())
GRAPH_SOURCES = ({"6b32a70d0768f0cc81971f88f0ff22c2d068155bb1c861baf8df73824ce64248"}
    if reviewed("graph.py", "103f05f6fb895f49d7704022f220feadfca9d8fa5593ee67e79f3e1ef2c5cb15") else set())
