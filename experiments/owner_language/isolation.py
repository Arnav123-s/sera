"""Explicit laboratory import isolation and recorded process identity.

The reused interpreter carries an editable install that appends the production
``src`` directory to ``sys.path``.  An unqualified ``import sera`` therefore
resolves to production.  Review finding 5 requires that laboratory work prove,
inside a fresh process, that every imported module actually comes from this
checkout.  ``activate()`` makes that true and ``identity()`` records it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
LAB_SRC = LAB_ROOT / "src"
REQUIRED = ("sera", "experiments", "scripts", "workbench")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate():
    """Give this checkout precedence over any environment-installed package."""
    if not (LAB_SRC / "sera" / "__init__.py").exists():
        raise RuntimeError(f"Laboratory sources are missing under {LAB_SRC}")
    for entry in (str(LAB_SRC), str(LAB_ROOT)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(LAB_ROOT))
    sys.path.insert(0, str(LAB_SRC))
    stale = sorted(name for name in sys.modules
                   if name.split(".")[0] in REQUIRED and not _under_lab(sys.modules[name]))
    if stale:
        raise RuntimeError("Non-laboratory modules were imported before isolation: " + ", ".join(stale))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return str(LAB_ROOT)


def _under_lab(module):
    path = getattr(module, "__file__", None)
    if path is None:
        paths = list(getattr(module, "__path__", []) or [])
        if not paths:
            return True
        return all(Path(p).resolve().is_relative_to(LAB_ROOT) for p in paths)
    return Path(path).resolve().is_relative_to(LAB_ROOT)


def check(names=REQUIRED):
    """Import each package and fail unless it resolves inside the laboratory."""
    resolved = {}
    for name in names:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None) or list(getattr(module, "__path__", []))[0]
        origin = Path(origin).resolve()
        if not origin.is_relative_to(LAB_ROOT):
            raise RuntimeError(f"{name} resolved to {origin}, outside {LAB_ROOT}")
        resolved[name] = {"file": origin.as_posix(),
                          "sha256": digest(origin) if origin.is_file() else None}
    return resolved


def third_party():
    import numpy
    import torch
    return {"numpy": {"version": numpy.__version__, "file": Path(numpy.__file__).resolve().as_posix()},
            "torch": {"version": torch.__version__, "file": Path(torch.__file__).resolve().as_posix()}}


def identity(extra_modules=()):
    """A recordable description of the executing process and its code."""
    modules = check()
    for name in extra_modules:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(LAB_ROOT):
            raise RuntimeError(f"{name} resolved to {origin}, outside {LAB_ROOT}")
        modules[name] = {"file": origin.as_posix(), "sha256": digest(origin)}
    return {"schema": "sera.owner-language.process-identity.1",
            "lab_root": LAB_ROOT.as_posix(),
            "executable": sys.executable,
            "executable_sha256": digest(sys.executable),
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "sys_path_head": sys.path[:4],
            "modules": modules,
            "third_party": third_party()}


def main():
    activate()
    print(json.dumps(identity(("experiments.owner_language.isolation",)), indent=2))


if __name__ == "__main__":
    LAB_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(LAB_ROOT))
    from experiments.owner_language.isolation import main as entry
    entry()
