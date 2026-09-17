"""Admit only a reviewed numerical-replay reader for the retained constraint state."""

import hashlib
from pathlib import Path

REPLAY_SOURCES = (
    {"74abc339e58b074b32468270895a7f6dac66f35bdc63344df7cc6397b7bfe5d9",
     "fc370be85947f75da95bb9b4ba86592b932e187deedcc2b7b3097db2e4350e02"}
    if hashlib.sha256(Path(__file__).with_name("runtime.py").read_bytes()).hexdigest()
    == "3805b4693e4c918f5c8136a155d1be0455cd442419f708f427ac10f11a4abbda"
    else set()
)
