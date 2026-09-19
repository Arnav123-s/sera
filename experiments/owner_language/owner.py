"""Explicit laboratory restore of the current SERA owner, with a real inventory.

Review finding 2: the draft verifier called ``restore()`` with no laboratory
path and then merely checked that a file existed, so it reported production as
if it were the laboratory baseline.  Here the store is always named explicitly,
the restored identity is compared against the recorded owner, and the returned
record says which bytes were read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
BASELINE_STORE = LAB_ROOT / "runs/owner-learning-001/baseline"
OWNER_IDENTITY = "62ea82cff82b88d1eb3848f7246494979c63894d869db603007e3d06111063ec"
PARENT_OWNER_IDENTITY = "376b3d076294ac5b92f9360bb27553edbd96d1aa2b12ae22913b55c0b135fa94"


def _pointer(store):
    return json.loads((Path(store) / "current.json").read_text())


def restore_owner(store=BASELINE_STORE, *, expect=OWNER_IDENTITY):
    """Restore the qualified owner from an explicitly named laboratory store.

    Returns ``(session, growth, record)``.  ``session`` is the live
    ``InterventionSession``; ``session.owner`` is the one shared R1 object the
    language, mathematics, inquiry, field and intervention routes all hold.
    """
    store = Path(store).resolve()
    if not store.is_relative_to(LAB_ROOT / "runs"):
        raise ValueError(f"Restore from a laboratory store, not {store}")
    from experiments.intervention_portability import installed
    from scripts.intervention_study import restore as strict_restore

    with installed():
        session, growth = strict_restore(store=store)
    identity = session.identity()
    if expect is not None and identity != expect:
        raise ValueError(f"Restored owner {identity} is not the recorded owner {expect}")
    record = {"store": store.as_posix(), "pointer": _pointer(store), "owner": identity,
              "parent_owner": session.parent_owner, "subjects": len(session.subjects),
              "owner_class": type(session.owner).__name__,
              "owner_mro": [cls.__name__ for cls in type(session.owner).__mro__ if cls.__name__ != "object"]}
    return session, growth, record


def tensor_digest(tensor):
    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def parameter_inventory(module, *, values=True):
    """Every parameter and buffer of the shared owner, with trainability."""
    parameters = []
    for name, tensor in module.named_parameters():
        parameters.append({"name": name, "role": "parameter", "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "elements": tensor.numel(),
                           "requires_grad": bool(tensor.requires_grad),
                           "sha256": tensor_digest(tensor) if values else None})
    for name, tensor in module.named_buffers():
        parameters.append({"name": name, "role": "buffer", "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "elements": tensor.numel(),
                           "requires_grad": False,
                           "sha256": tensor_digest(tensor) if values else None})
    return parameters


def inventory_summary(entries):
    parameters = [e for e in entries if e["role"] == "parameter"]
    buffers = [e for e in entries if e["role"] == "buffer"]
    return {"tensors": len(entries), "parameters": len(parameters), "buffers": len(buffers),
            "parameter_elements": sum(e["elements"] for e in parameters),
            "buffer_elements": sum(e["elements"] for e in buffers),
            "trainable_parameters": sum(e["elements"] for e in parameters if e["requires_grad"]),
            "identity_digest": hashlib.sha256(
                json.dumps([{k: e[k] for k in ("name", "role", "shape", "dtype", "sha256")} for e in entries],
                           sort_keys=True).encode()).hexdigest()}
