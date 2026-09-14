"""Validated JSON state persistence without executable payloads or implicit ownership."""

from __future__ import annotations

import hashlib

import torch

from sera.storage import digest


def model_identity(model):
    state = model.state_dict()
    signature = (digest(model.export_config()),
                 tuple((name, value.data_ptr(), value._version, str(value.dtype), tuple(value.shape))
                       for name, value in state.items()))
    cached = getattr(model, "_session_identity_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    hasher = hashlib.sha256(signature[0].encode())
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str((value.dtype, tuple(value.shape))).encode())
        hasher.update(value.numpy().tobytes())
    identity = hasher.hexdigest()
    model._session_identity_cache = (signature, identity)
    return identity


def pack_tensors(state):
    if isinstance(state, dict):
        return {key: pack_tensors(value) for key, value in state.items()}
    if not isinstance(state, torch.Tensor) or not torch.isfinite(state).all():
        raise ValueError("Session state requires finite tensors")
    tensor = state.detach().cpu()
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "real": tensor.real.tolist()}
    if tensor.is_complex():
        result["imag"] = tensor.imag.tolist()
    return result


def unpack_tensors(payload, template):
    if isinstance(template, dict):
        if not isinstance(payload, dict) or set(payload) != set(template):
            raise ValueError("Session state keys do not match this model")
        return {key: unpack_tensors(payload[key], value) for key, value in template.items()}
    if not isinstance(payload, dict) or payload.get("shape") != list(template.shape):
        raise ValueError("Session tensor shape mismatch")
    if payload.get("dtype") != str(template.dtype):
        raise ValueError("Session tensor dtype mismatch")
    expected = {"shape", "dtype", "real", "imag"} if template.is_complex() else {"shape", "dtype", "real"}
    if set(payload) != expected:
        raise ValueError("Invalid session tensor encoding")
    try:
        real = torch.tensor(payload["real"], dtype=template.real.dtype, device=template.device)
        value = (torch.complex(real, torch.tensor(payload["imag"], dtype=real.dtype, device=real.device))
                 if template.is_complex() else real)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Malformed session tensor values") from error
    if value.shape != template.shape or not torch.isfinite(value).all():
        raise ValueError("Session tensor values violate shape or finiteness")
    return value
