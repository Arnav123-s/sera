import numpy as np
import pytest

from experiments.sparse_mechanisms.repair import decode, digest, encode, load, save


def test_exact_arbitrary_bytes_restore_after_unknown_corruption(tmp_path):
    payload = bytes(range(256))+b"odd length!"
    metadata, encoded = encode(payload, 431)
    damaged = encoded.copy()
    damaged[:, [1, 17, 39]] += [3.5, -2.4, 8.]
    archive = tmp_path/"coded.zip"
    save(archive, metadata, damaged)
    saved_metadata, saved_codes = load(archive)
    result, checks = decode(saved_metadata, saved_codes, digest(payload))
    assert result == payload and all(c["used_sparse_error_solver"] for c in checks)
    with pytest.raises(FileExistsError):
        save(archive, metadata, encoded)


def test_wrong_or_damaged_memory_never_returns_plausible_bytes():
    payload = bytes(range(61))
    metadata, encoded = encode(payload, 173)
    with pytest.raises(ValueError, match="digest"):
        decode(metadata, encoded, "0"*64)
    with pytest.raises(ValueError):
        decode(metadata, np.zeros_like(encoded), digest(payload))
    damaged = encoded.copy()
    damaged[0, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        decode(metadata, damaged, digest(payload))
    altered = dict(metadata, seed=metadata["seed"]+1)
    with pytest.raises(ValueError, match="Operator"):
        decode(altered, encoded, digest(payload))


def test_invalid_large_inputs_rejected():
    for payload in (b"", b"x"*8193):
        with pytest.raises(ValueError):
            encode(payload, 1)
