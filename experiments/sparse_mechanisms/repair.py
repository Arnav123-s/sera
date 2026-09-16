"""Experimental exact-byte recovery for small retained model artifacts.

Sparse errors are repaired with redundant linear measurements. A separately
retained SHA-256 is mandatory: plausible numerical reconstructions are rejected.
This is fault protection with expansion, not arbitrary-memory compression.
"""

import argparse
import hashlib
import io
import json
import time
import zipfile

import numpy as np

from .study import RELEASE, ROOT, source_files, write
from .variants import repair_dense, sketch_operator

BLOCK_WORDS = 32
CODE_VALUES = 64
MAX_BYTES = 8192


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(payload, seed):
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Input must contain 1..8192 bytes")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Seed must be a nonnegative 63-bit integer")
    padded = payload+b"\0"*((-len(payload)) % (2*BLOCK_WORDS))
    words = np.frombuffer(padded, dtype="<u2").reshape(-1, BLOCK_WORDS)
    a = sketch_operator(seed, CODE_VALUES, BLOCK_WORDS)
    encoded = (words.astype(float)/65535.)@a.T
    metadata = {"schema": "sera.sparse-repair.1", "length": len(payload), "seed": seed,
                "block_words": BLOCK_WORDS, "code_values": CODE_VALUES,
                "payload_sha256": digest(payload), "operator_sha256": digest(a.astype("<f8").tobytes()),
                "block_sha256": [digest(row.tobytes()) for row in words],
                "encoded_sha256": [digest(row.astype("<f8").tobytes()) for row in encoded],
                "numpy_version": np.__version__,
                "scope": "Finite float64 codeword corruption; trusted metadata and separately retained payload digest. Up to 8192 bytes. Numeric code storage expands data eightfold before metadata, and no fault-count guarantee is claimed."}
    return metadata, encoded


def decode(metadata, received, expected_sha256):
    if metadata.get("schema") != "sera.sparse-repair.1" or metadata.get("payload_sha256") != expected_sha256:
        raise ValueError("Trusted digest or schema mismatch")
    length = metadata.get("length")
    seed = metadata.get("seed")
    if not isinstance(length, int) or not 0 < length <= MAX_BYTES:
        raise ValueError("Invalid payload size")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Invalid matrix seed")
    blocks = (length+2*BLOCK_WORDS-1)//(2*BLOCK_WORDS)
    received = np.asarray(received)
    if received.shape != (blocks, CODE_VALUES) or received.dtype != np.float64 or not np.isfinite(received).all():
        raise ValueError("Invalid or nonfinite codeword array")
    if metadata.get("block_words") != BLOCK_WORDS or metadata.get("code_values") != CODE_VALUES:
        raise ValueError("Unsupported code dimensions")
    if len(metadata.get("block_sha256", [])) != blocks or len(metadata.get("encoded_sha256", [])) != blocks:
        raise ValueError("Missing block integrity metadata")
    a = sketch_operator(seed, CODE_VALUES, BLOCK_WORDS)
    if digest(a.astype("<f8").tobytes()) != metadata["operator_sha256"]:
        raise ValueError("Operator regeneration changed; use the archived environment")
    recovered, checks = [], []
    for index, row in enumerate(received):
        untouched = digest(row.astype("<f8").tobytes()) == metadata["encoded_sha256"][index]
        if untouched:
            value = np.linalg.lstsq(a, row, rcond=None)[0]
        else:
            value, _, _ = repair_dense(a, row)
        rounded = np.rint(value*65535.)
        if (rounded < 0).any() or (rounded > 65535).any():
            raise ValueError(f"Block {index} recovery outside word range")
        result = rounded.astype("<u2").tobytes()
        if digest(result) != metadata["block_sha256"][index]:
            raise ValueError(f"Block {index} failed independent payload integrity check")
        recovered.append(result)
        checks.append({"block": index, "used_sparse_error_solver": not untouched, "payload_hash_verified": True})
    payload = b"".join(recovered)[:length]
    if digest(payload) != expected_sha256:
        raise ValueError("Full payload integrity check failed")
    return payload, checks


def save(path, metadata, encoded):
    buffer = io.BytesIO()
    np.save(buffer, encoded, allow_pickle=False)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2)+"\n")
        archive.writestr("codewords.npy", buffer.getvalue())


def load(path):
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"metadata.json", "codewords.npy"}:
            raise ValueError("Unexpected archive members")
        if sum(item.file_size for item in archive.infolist()) > 200000:
            raise ValueError("Archive exceeds this small-artifact utility's limit")
        metadata = json.loads(archive.read("metadata.json"))
        words = np.load(io.BytesIO(archive.read("codewords.npy")), allow_pickle=False)
    return metadata, words


def repetition_recover(data, expected):
    # Eight byte copies match the 8x numeric-storage overhead of the code.
    rows = np.frombuffer(data, dtype=np.uint8).reshape(8, -1)
    answer = []
    for column in rows.T:
        counts = np.bincount(column, minlength=256)
        if counts.max() < 5:
            raise ValueError("No majority")
        answer.append(int(counts.argmax()))
    recovered = bytes(answer)
    if digest(recovered) != expected:
        raise ValueError("Repetition digest mismatch")
    return recovered


def demonstrate(name, seeds, start_seed):
    folder = RELEASE/name
    folder.mkdir(exist_ok=False)
    pointer_path = ROOT/"runs/sera-workbench/current.json"
    pointer_before = pointer_path.read_bytes()
    pointer = json.loads(pointer_before)
    live_path = ROOT/"runs/sera-workbench/revisions"/pointer["revision"]
    live_bytes = live_path.read_bytes()
    if digest(live_bytes) != pointer["sha256"]:
        raise ValueError("Live source revision integrity mismatch")
    live = json.loads(live_bytes)
    payload = (json.dumps({"schema": "sera.retained-numeric-contexts.1", "owner_sha256": live["owner_sha256"],
                           "contexts": live["contexts"]}, sort_keys=True, separators=(",", ":"))+"\n").encode()
    (folder/"retained-contexts.json").write_bytes(payload)
    protocol = {"name": name, "seeds": seeds, "start_seed": start_seed, "sources": source_files(),
                "payload_sha256": digest(payload), "payload_bytes": len(payload),
                "source_revision": str(live_path.relative_to(ROOT)), "source_revision_sha256": digest(live_bytes),
                "owner_sha256": live["owner_sha256"],
                "fault_model": "Replace 0/4/12/24 finite 64-bit cells per 64-cell block with N(0,5) values. Locations unknown to repair. Same replacement cells/bytes for eight-copy byte repetition at equal numeric storage. Metadata and trusted digest are intact.",
                "gate": "Return bytes only if each recovered block and the complete payload match pre-recorded hashes. No silent corrupt output. Report restore/reject rates for every fault count. No net compression or superiority over ordinary ECC/backups claim.",
                "boundary": "Protects a serialized copy of the actual retained numeric task contexts, not the entire neural checkpoint or the live owner; no live mutation."}
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in protocol["sources"]:
            archive.write(ROOT/path, path)
    rows = []
    for index in range(seeds):
        seed = start_seed+101*index
        metadata, encoded = encode(payload, seed)
        padded = payload+b"\0"*((-len(payload)) % 64)
        originals = [padded[i:i+64] for i in range(0, len(padded), 64)]
        clean_repetition = np.stack([np.frombuffer(block*8, dtype=np.uint8).copy() for block in originals])
        for damaged_count in (0, 4, 12, 24):
            damaged = encoded.copy()
            repetition = clean_repetition.copy()
            rng = np.random.default_rng(seed+damaged_count*100003)
            indices = []
            for block in range(len(encoded)):
                locations = rng.choice(64, damaged_count, replace=False)
                replacements = rng.normal(scale=5., size=damaged_count).astype("<f8")
                damaged[block, locations] = replacements
                repetition[block].reshape(64, 8)[locations] = replacements.view(np.uint8).reshape(-1, 8)
                indices.append(locations.tolist())
            for method in ("sparse_error", "eight_copies"):
                started = time.perf_counter()
                rejection = None
                try:
                    if method == "sparse_error":
                        restored, _ = decode(metadata, damaged, digest(payload))
                    else:
                        restored = b"".join(repetition_recover(block.tobytes(), digest(original)) for block, original in zip(repetition, originals, strict=True))[:len(payload)]
                    if restored != payload:
                        raise AssertionError("Silent wrong restore")
                    accepted = True
                except ValueError as exc:
                    accepted, rejection = False, str(exc)
                rows.append({"seed": seed, "method": method, "corrupt_cells_per_block": damaged_count,
                             "blocks": len(encoded), "corrupted_indices": indices,
                             "restored_exactly": accepted, "rejection": rejection,
                             "seconds": time.perf_counter()-started, "numeric_stored_bytes": int(encoded.nbytes),
                             "payload_bytes": len(payload), "metadata_bytes": len(json.dumps(metadata).encode()) if method == "sparse_error" else 32*len(encoded)})
            if index == 0:
                save(folder/f"example-{damaged_count}-damaged-cells.zip", metadata, damaged)
    assert pointer_path.read_bytes() == pointer_before and live_path.read_bytes() == live_bytes
    result = {"status": "PASS", "rows": rows, "silent_wrong_restores": 0,
              "live_source_unchanged": True, "payload_sha256": digest(payload), "payload_bytes": len(payload),
              "table": {f"{count}/{method}": {"trials": seeds, "exact_restores": sum(r["restored_exactly"] for r in rows if r["method"] == method and r["corrupt_cells_per_block"] == count)} for count in (0, 4, 12, 24) for method in ("sparse_error", "eight_copies")}}
    write(folder/"result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    protect = commands.add_parser("protect")
    protect.add_argument("--input", required=True)
    protect.add_argument("--output", required=True)
    protect.add_argument("--seed", type=int, default=43)
    restore = commands.add_parser("restore")
    restore.add_argument("--input", required=True)
    restore.add_argument("--output", required=True)
    restore.add_argument("--expected-sha256", required=True)
    demo = commands.add_parser("demonstrate")
    demo.add_argument("--name", required=True)
    demo.add_argument("--seeds", type=int, required=True)
    demo.add_argument("--start-seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "demonstrate":
        demonstrate(args.name, args.seeds, args.start_seed)
    elif args.command == "protect":
        from pathlib import Path
        path = Path(args.input)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Input exceeds 8192 bytes")
        metadata, encoded = encode(path.read_bytes(), args.seed)
        save(args.output, metadata, encoded)
        print(json.dumps({"status": "PROTECTED", "payload_sha256": metadata["payload_sha256"], "instruction": "Keep this digest separately. It is mandatory for restoration."}))
    else:
        metadata, encoded = load(args.input)
        payload, checks = decode(metadata, encoded, args.expected_sha256)
        # Exclusive creation occurs only after full verification, never on failure.
        with open(args.output, "xb") as stream:
            stream.write(payload)
        print(json.dumps({"status": "VERIFIED_EXACT_RESTORE", "bytes": len(payload), "checks": checks}))


if __name__ == "__main__":
    main()
