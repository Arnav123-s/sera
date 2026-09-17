"""Admit the retained request record only through its reviewed successor reader."""

import hashlib
from pathlib import Path

REPLAY_SOURCES = (
    {"91a0c526e55fca8cbff8f37db07e12a84615a45d5d185af7a2f6df5fec3e51ed"}
    if hashlib.sha256(Path(__file__).with_name("runtime.py").read_bytes()).hexdigest()
    == "66a15259b6eb6ef7324e186d0143deeeb5c277c679abc74ec6813a66a996019d"
    else set()
)
