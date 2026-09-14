"""Index existing work and verify preservation; never move, remove or deduplicate it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from sera.storage import write_json

ROLES = {
    "src/sera": ("active-code", "SERA 0.3 executable source; edit here for subsequent versioned research."),
    "scripts": ("active-verification", "Reproduction, reporting and audit entry points."),
    "tests": ("active-verification", "Numerical and lifecycle regression suite."),
    "examples": ("active-interface", "Declared typed request examples."),
    "research/reference-materials": ("source-reference", "Professionally named byte-identical references; original-package is a partial reading extraction."),
    "research/intake/Quantum_AGI_Research_Package": ("source-reference", "Complete original packet extraction; keep embedded names and source bytes."),
    "research/intake/sera-release-check": ("historical-package-check", "Early SERA renamed-release checkout; preserved source snapshot, not a reference architecture."),
    "research/intake/release-check": ("historical-package-check", "Early release verification files; not active code."),
    "research/intake/wheel-check": ("historical-package-check", "Early installed-wheel snapshot; not active code."),
    "research/intake/renders": ("source-visual-review", "Original source page rendering artifacts."),
    "research/intake/visual-review": ("source-visual-review", "Original visual review records."),
    "research/intake": ("historical-intake", "Early source reading, naming and repository setup records; nested source and installation trees have separate roles."),
    "research": ("research-record", "Current design/provenance documents and explicitly dated historical protocols. The source-packet audit supersedes completion claims."),
    "reports": ("research-evidence", "Versioned measurements and reviews. Current interpretation: source-packet-comparison.md; frozen 0.3 data: stage-three-data.json."),
    "runs/sera-0.3-current": ("active-solver", "Only designated continuation workspace; seed-zero 0.3 solver plus a separately recorded CLI learning attempt. Not a new study seed."),
    "runs/sera-current": ("historical-solver", "Preserved 0.2 continuation workspace despite its old current name; not the active solver."),
    "runs/sera-v3-current": ("superseded-solver", "Working copy from the obsolete 0.3 cohort. Do not pool its results with the repaired release."),
    "runs/stage-three-complete/worker-runs": ("preserved-worker-evidence", "Original worker outputs. Canonical 0/1/2 copies belong to these same runs, not additional independent seeds."),
    "runs/stage-three-complete": ("frozen-release-evidence", "Canonical repaired 0.3 study, seeds 0/1/2. Checkpoints, journals and measurements are immutable."),
    "runs/stage-three-audit": ("verification", "Independent reconstruction of trained reference memories and live sessions; not additional training-cohort seeds."),
    "runs/stage-three-development": ("development", "Tuning, pilots and probe results. Exposed development evidence; excluded from the final cohort."),
    "runs/stage-three-smoke-1": ("failed-smoke", "Short integration run rejected by source-freeze checking; not capability evidence."),
    "runs/stage-three-final": ("interrupted", "Stopped after discovering the untrained legacy decoder in the study setup. Incomplete cost records."),
    "runs/stage-three-release": ("failed-cohort", "Obsolete source ee62373c; seed one failed on all-missing likelihood targets. Preserve complete and partial outputs together."),
    "runs/stage-three-verified": ("interrupted", "Repaired serial cohort stopped before independent-worker orchestration. Not the final cohort."),
    "runs/stage-three-installed": ("superseded-package-check", "Installed pre-repair package labeled 0.3.0; source identity, not version string, distinguishes it."),
    "runs/stage-three-installed-final": ("release-package-check", "Installed repaired 0.3 package used for final installation checks."),
    "runs/original-reproduction-stage-three": ("source-reproduction", "Original 12 cores: numerical checks, 36 checkpoint evaluations and 36 fresh training/adaptation/scratch procedures. Separate from SERA models."),
    "runs/original-source-audit": ("historical-verification", "Earlier source audit, rendered reference pages and baseline source archive."),
    "runs/source-packet-review": ("current-audit", "Preservation baseline, supplementary source-model reproduction and integration-scope probes from this review."),
    "runs/baseline-v1": ("historical-evidence", "First SERA mechanism cohort; changed task encoding and models relative to the original packet."),
    "runs/integrated-v1": ("historical-evidence", "First integration and adaptation evidence."),
    "runs/matched-delta-v1": ("historical-control", "Early parameter-count control; use its own design and source identity."),
    "runs/instrument-complex-0": ("development", "Early single-instrument run; not an extra seed of the later suite."),
    "runs/instrument-suite-v1": ("historical-evidence", "Earlier separate real/complex EventInstrument suite."),
    "runs/connected-dev-r1": ("development", "Early connected R1 model probe."),
    "runs/connected-smoke-v2": ("smoke-check", "Short 0.2 integration check; not release capability evidence."),
    "runs/connected-v2": ("incomplete", "Early 0.2 cohort with a manifest but no complete summary; exact termination is not reconstructed."),
    "runs/connected-v2-final": ("historical-release-evidence", "Completed 0.2 study, preserved with its own source and verification."),
    "runs/cli-learning-check": ("historical-verification", "0.2 CLI continuation check; preserve its journal and preparation record."),
    "runs/package-check": ("historical-package-check", "Earlier package installation check; not source to edit."),
    "runs/release-check-v2": ("historical-package-check", "Preserved 0.2 checkout/install verification tree."),
    "build": ("build-snapshot", "Existing generated build tree; never the source of truth and not updated by this review."),
    "dist": ("release-artifacts", "Versioned wheels/source archives and historical packaged solver. Existing filenames are never overwritten by this review."),
    "src/sable_learning.egg-info": ("historical-package-metadata", "Preserved early package-name metadata; not an active second project."),
    "src/sera_learning.egg-info": ("package-metadata", "Existing editable-install metadata; generated, not model source."),
    ".github": ("active-verification", "Public continuous integration configuration."),
}

CORRECTED_DOCUMENTS = {
    "README.md", "research/architecture.md", "research/roadmap.md",
    "research/stage-three-checklist.md", "reports/stage-three-architecture-audit.md",
}


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(directory):
    hasher = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return hasher.hexdigest()


def classify(path):
    prefixes = [prefix for prefix in ROLES if path == prefix or path.startswith(prefix + "/")]
    if prefixes:
        prefix = max(prefixes, key=len)
        return prefix, *ROLES[prefix]
    if "/" not in path:
        return "repository-root", "project-entry", "Project description, dependencies and contribution settings."
    raise ValueError(f"Unclassified work: {path}")


def verify(root, baseline):
    unchanged, corrected, missing, unexpected = [], [], [], []
    for row in baseline["files"]:
        path = root / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        if sha_file(path) == row["sha256"]:
            unchanged.append(row["path"])
            continue
        if row["path"] not in CORRECTED_DOCUMENTS:
            unexpected.append(row["path"])
            continue
        historical = subprocess.check_output(["git", "show", f"{baseline['git_head']}:{row['path']}"], cwd=root)
        if hashlib.sha256(historical).hexdigest() != row["sha256"]:
            unexpected.append(row["path"])
            continue
        corrected.append({"path": row["path"], "original_sha256": row["sha256"],
                          "current_sha256": sha_file(path),
                          "preserved_at": f"{baseline['git_head']}:{row['path']}"})
    return {"baseline_files": len(baseline["files"]), "unchanged_files": len(unchanged),
            "corrected_documents_preserved_in_git": corrected, "missing_files": missing,
            "unexpected_changes": unexpected,
            "passed": not missing and not unexpected,
            "boundary": "Verifies work present at review start. Git metadata, third-party environment and runtime caches are excluded. This is local preservation, not an off-device backup claim."}


def catalog(root, baseline, baseline_path, preservation):
    groups, duplicates = {}, defaultdict(list)
    for row in baseline["files"]:
        prefix, role, note = classify(row["path"])
        group = groups.setdefault(prefix, {"path": prefix, "role": role, "purpose": note, "files": 0,
                                            "bytes": 0, "members": []})
        group["files"] += 1
        group["bytes"] += row["bytes"]
        group["members"].append(row)
        duplicates[row["sha256"]].append(row)
    for group in groups.values():
        rows = group.pop("members")
        identity = json.dumps(sorted(rows, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":"))
        group["baseline_inventory_sha256"] = hashlib.sha256(identity.encode()).hexdigest()
        path = root / group["path"]
        manifest = path / "manifest.json"
        if manifest.is_file():
            body = json.loads(manifest.read_text(encoding="utf-8"))
            group["recorded_source_sha256"] = body.get("environment", {}).get("source_sha256")
        pointer = path / "current.json"
        if pointer.is_file():
            body = json.loads(pointer.read_text(encoding="utf-8"))
            group["solver_at_review"] = {key: body[key] for key in ("version", "solver_sha256", "checkpoint_sha256") if key in body}
    substantial = [rows for rows in duplicates.values() if len(rows) > 1 and rows[0]["bytes"] >= 4096]
    repeated_bytes = sum((len(rows) - 1) * rows[0]["bytes"] for rows in duplicates.values())
    sources = []
    for row in baseline["files"]:
        if row["path"].endswith("sera/__init__.py"):
            parent = (root / row["path"]).parent
            sources.append({"path": parent.relative_to(root).as_posix(), "source_sha256": source_identity(parent)})
    result = {"schema_version": 1, "inventory_at": baseline["created_utc"],
              "baseline_commit": baseline["git_head"],
              "baseline_manifest": {"path": baseline_path.relative_to(root).as_posix(), "sha256": sha_file(baseline_path)},
              "inventory_scope": baseline["scope"],
              "total_files_at_review_start": len(baseline["files"]),
              "total_bytes_at_review_start": sum(row["bytes"] for row in baseline["files"]),
              "preservation": preservation,
              "active_solver": "runs/sera-0.3-current", "frozen_study": "runs/stage-three-complete",
              "current_audit": "reports/source-packet-comparison.md",
              "groups": sorted(groups.values(), key=lambda row: row["path"]),
              "new_review_work": {"path": "runs/source-packet-review", "role": "current-audit",
                                   "purpose": "New audit outputs are outside the pre-existing-work inventory."},
              "preserved_source_snapshots": sources,
              "duplicate_content": {"exact_duplicate_groups": sum(len(rows) > 1 for rows in duplicates.values()),
                                    "additional_copy_bytes": repeated_bytes,
                                    "meaning": "Logical file-byte duplication, not measured reclaimable disk space. No copies removed or hard-linked.",
                                    "largest_groups": [{"sha256": rows[0]["sha256"], "bytes_each": rows[0]["bytes"],
                                                        "paths": [row["path"] for row in rows]}
                                                       for rows in sorted(substantial, key=lambda rows: (len(rows)-1)*rows[0]["bytes"], reverse=True)[:12]]},
              "policy": ["Keep original sources and frozen evidence immutable.",
                         "Use this catalog for labels; do not infer status from final/current/verified in old directory names.",
                         "Keep original worker outputs and canonical copies as one experiment identity.",
                         "Start future experiments in fresh descriptive directories with parent, source hash, purpose and status.",
                         "Revise interpretations through Git; preserve the original result bytes and the earlier document revision.",
                         "Do not edit installed/build copies, rename immutable RNG tags, overwrite same-version artifacts or purge unsuccessful runs."]}
    return result


def index_markdown(result):
    lines = ["# SERA workspace index", "",
        "I preserve earlier work and give each location one explicit role. This index is the navigation layer; existing evidence paths stay stable so old scripts, journals and source references remain usable.", "",
        "- **Current source:** [src/sera](../src/sera).",
        "- **Active continuation solver:** [runs/sera-0.3-current](../runs/sera-0.3-current).",
        "- **Frozen 0.3 study:** [runs/stage-three-complete](../runs/stage-three-complete), canonical seeds 0/1/2.",
        "- **Current architecture judgment:** [source-packet comparison](../reports/source-packet-comparison.md).",
        "- **Structured catalog:** [workspace-catalog.json](workspace-catalog.json).", "",
        "Ignored local directories are available in this workspace; their links are not downloads from the public repository. Published measurements remain in reports. The catalog does not claim an off-device backup of local checkpoints.", "",
        "## Preservation record", "",
        f"The review began with {result['total_files_at_review_start']:,} work files totaling {result['total_bytes_at_review_start'] / 1048576:.2f} MiB. A compressed per-file SHA-256 inventory is preserved at `{result['baseline_manifest']['path']}`. Git history, the installed third-party environment and runtime caches are outside that work inventory.", "",
        f"Verification found {result['preservation']['unchanged_files']:,} unchanged files, {len(result['preservation']['corrected_documents_preserved_in_git'])} corrected documents whose original bytes remain in commit `{result['baseline_commit']}`, and {len(result['preservation']['missing_files'])} missing files. No existing work was moved, removed or deduplicated.", "",
        "## Location and role", "",
        "Counts below describe the starting inventory; the new audit has its own location. The worker subtree is counted separately from canonical study files.", "",
        "| Location | Role | Files | MiB | Purpose |", "|---|---|---:|---:|---|"]
    for row in result["groups"]:
        target = ".." if row["path"] == "repository-root" else "../" + row["path"]
        lines.append(f"| [{row['path']}]({target}) | {row['role']} | {row['files']} | {row['bytes'] / 1048576:.2f} | {row['purpose']} |")
    lines += ["| [runs/source-packet-review](../runs/source-packet-review) | current-audit | new | new | Baseline, supplementary reproduction and integration probes from this review. |", "",
        "## Avoiding duplicate work", "",
        f"The baseline contains {result['duplicate_content']['exact_duplicate_groups']} groups of byte-identical content, representing {result['duplicate_content']['additional_copy_bytes'] / 1048576:.2f} MiB of additional logical file copies. This includes canonical/worker outputs, reference extractions, packaged copies and repeated checkpoint components. It is not a measurement of reclaimable disk space.", "",
        "I retain these copies because they identify delivered sources, independent worker outputs, installations or historical workspaces. I do not count copies as independent evidence. New work uses one declared output directory; reports reference evidence instead of copying whole run trees. Installed packages and build snapshots are inspection artifacts, not parallel source branches.", "",
        "## Revisiting earlier work", "",
        "Read the directory role and its manifest first. Match the recorded source identity to the preserved source snapshots in the catalog. Several installed copies have the same package version and different source hashes; version 0.3.0 alone does not identify the repaired implementation. A run without recorded provenance remains labeled incomplete or development, rather than receiving an invented release identity.", "",
        "To continue learning, start from the active continuation directory or use `scripts/prepare_solver.py` with an explicit completed seed and a fresh output. Do not train into canonical study directories or overwrite an earlier wheel/source archive. The historical `sable/data-v1` RNG namespace remains unchanged because changing it would change recorded datasets.", "",
        "To verify preservation again:", "", "```text",
        "python scripts/catalog_workspace.py --baseline runs/source-packet-review/preservation-baseline.json.gz --check",
        "```", "",
        "The per-file inventory and hash checks establish preservation of the work present at this review's start. Earlier Git revisions preserve previous document claims. New audit outputs extend the record; they do not silently replace the 0.3 experiment.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_path = args.baseline.resolve()
    baseline = json.loads(gzip.decompress(baseline_path.read_bytes()))
    preservation = verify(root, baseline)
    if not preservation["passed"]:
        raise ValueError(json.dumps(preservation, indent=2))
    if not args.check:
        result = catalog(root, baseline, baseline_path, preservation)
        write_json(root / "research/workspace-catalog.json", result)
        (root / "research/workspace-index.md").write_text(index_markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(preservation, indent=2))


if __name__ == "__main__":
    main()
