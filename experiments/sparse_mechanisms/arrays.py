"""Content-addressed, lossless numerical evidence without repeated array blobs."""

import hashlib
import io
import zipfile

import numpy as np


class ArrayWriter:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED)
        self.known = set()

    def put(self, values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values), allow_pickle=False)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in self.known:
            self.archive.writestr(digest+".npy", payload)
            self.known.add(digest)
        return {"array_sha256": digest}

    def close(self):
        self.archive.close()


class ArrayReader:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)

    def get(self, reference):
        digest = reference["array_sha256"]
        payload = self.archive.read(digest+".npy")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Array digest mismatch")
        return np.load(io.BytesIO(payload), allow_pickle=False)

    def close(self):
        self.archive.close()
