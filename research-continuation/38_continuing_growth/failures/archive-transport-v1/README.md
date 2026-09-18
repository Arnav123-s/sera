# Preserve the first archive and transport revision

The first portable archive passed exact checkpoint roundtrip but was 211,918,505 bytes, exceeding an ordinary Git blob's size limit. Its subsequent exhaustive verification ran past the first 160-second job allowance. The full job cost and original helpers, manifest and checkpoint index are preserved.

The revised transport splits those same archive bytes into 40 MiB release assets. The new manifest records the old manifest hash and the unchanged whole-archive hash. All checkpoints, failed candidates, optimizer states and raw evidence are retained. Original local archive: `runs/CG-package-001/oversized-v1/research.zip`. The corresponding public parts reconstruct it exactly without a second scientific run.
