import io
import zipfile

import pytest
import torch

from scripts.refinement_artifacts import SplitFile, packed, unpacked
from scripts.refinement_audit import equal


def test_compact_checkpoint_preserves_optimizer_keys_rng_and_shared_values():
    original = {"state": {0: {"step": torch.tensor(12.), "moment": torch.tensor([.2, -.3])}},
                "rng": torch.tensor([4, 8, 2], dtype=torch.uint8), "tuple": (1, True, "retained")}
    buffer, names = io.BytesIO(), set()
    with zipfile.ZipFile(buffer, "w") as archive:
        first = packed(original, archive, names)
        second = packed(original, archive, names)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        assert len(archive.namelist()) == 3
        assert equal(original, unpacked(first, archive))
        assert equal(original, unpacked(second, archive))


def test_tensor_blob_content_hash_is_checked():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tensors/forged.npy", b"changed")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        with pytest.raises(ValueError, match="Changed"):
            unpacked({"kind": "tensor", "blob": "tensors/forged.npy"}, archive)


def test_segmented_archive_can_seek_across_part_boundaries(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("one", b"preserved one"*40)
        archive.writestr("two", b"preserved two"*60)
    raw = buffer.getvalue()
    paths = []
    for start in range(0, len(raw), 31):
        path = tmp_path / str(start)
        path.write_bytes(raw[start:start+31])
        paths.append(path)
    with zipfile.ZipFile(SplitFile(paths)) as archive:
        assert archive.read("one") == b"preserved one"*40
        assert archive.read("two") == b"preserved two"*60
