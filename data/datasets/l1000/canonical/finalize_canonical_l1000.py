from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000")
DATASETS = {
    "l3_treated": {
        "stem": "l3_treated",
        "world": 32,
        "rows": 1_312_170,
        "manifest": "l3_treatment_manifest.parquet",
        "row_column": "treatment_row",
    },
    "l3_control": {
        "stem": "l3_control",
        "world": 4,
        "rows": 76_742,
        "manifest": "l3_control_manifest.parquet",
        "row_column": "control_row",
    },
    "l5": {
        "stem": "l5_signatures",
        "world": 16,
        "rows": 510_270,
        "manifest": "l5_signature_manifest.parquet",
        "row_column": "signature_row",
    },
}
KNOWN_SOURCE_SHA = {
    "l3_control": "45f528c978d48d8ac320f184a8d0cb21f22d54582d7af24b50701a91b6370d85",
    "l5": "fbb04a94d8aa8a4fa6daf01f7d94dcd938fb9b7937730f81bfba3d8d0f1472a8",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    unexpected = list(ROOT.rglob("*.part"))
    if unexpected:
        raise RuntimeError(f"Partial files remain: {unexpected[:20]}")

    summary = {}
    h5_receipt_sha = {}
    canonical_ranges = {}
    for dataset, cfg in DATASETS.items():
        ranges = []
        total_rows = 0
        total_bytes = 0
        for rank in range(cfg["world"]):
            stem = f"{cfg['stem']}_full_{rank:02d}-of-{cfg['world']:02d}"
            shard = ROOT / "data" / f"{stem}.h5"
            receipt_path = ROOT / "audit" / f"{stem}.json"
            verify_path = ROOT / "audit" / f"verify_{stem}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            verification = json.loads(verify_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "PASS" or verification.get("status") != "PASS":
                raise RuntimeError(f"Non-PASS receipt: {stem}")
            if receipt["sha256"] != verification["sha256"]:
                raise RuntimeError(f"Receipt hash disagreement: {stem}")
            if receipt["bytes"] != shard.stat().st_size:
                raise RuntimeError(f"Receipt byte disagreement: {stem}")
            with h5py.File(shard, "r") as handle:
                shape = tuple(handle["expression"].shape)
                start = int(handle.attrs["canonical_start"])
                stop = int(handle.attrs["canonical_stop"])
            if shape != (receipt["rows"], 978) or (start, stop) != (
                receipt["canonical_start"], receipt["canonical_stop"]
            ):
                raise RuntimeError(f"Shard shape/range disagreement: {stem}")
            ranges.append((start, stop))
            total_rows += int(receipt["rows"])
            total_bytes += int(receipt["bytes"])
            h5_receipt_sha[shard.relative_to(ROOT).as_posix()] = receipt["sha256"]
        if ranges[0][0] != 0 or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise RuntimeError(f"Non-contiguous ranges: {dataset}")
        if total_rows != cfg["rows"] or ranges[-1][1] != cfg["rows"]:
            raise RuntimeError(f"Wrong total rows: {dataset} {total_rows}")
        manifest = pq.read_table(ROOT / "data" / cfg["manifest"], columns=[cfg["row_column"]])
        if manifest.num_rows != cfg["rows"]:
            raise RuntimeError(f"Manifest total mismatch: {dataset}")
        if not np.array_equal(
            manifest[cfg["row_column"]].to_numpy(), np.arange(cfg["rows"], dtype=np.int64)
        ):
            raise RuntimeError(f"Manifest row axis mismatch: {dataset}")
        canonical_ranges[dataset] = ranges
        summary[dataset] = {
            "rows": total_rows,
            "genes": 978,
            "shards": cfg["world"],
            "compressed_expression_bytes": total_bytes,
        }

    source_hashes = {}
    for dataset in DATASETS:
        receipt = json.loads(
            (ROOT / "audit" / f"source_{dataset}_sha256.json").read_text(encoding="utf-8")
        )
        if receipt.get("status") != "PASS":
            raise RuntimeError(f"Source hash failed: {dataset}")
        if dataset in KNOWN_SOURCE_SHA and receipt["sha256"] != KNOWN_SOURCE_SHA[dataset]:
            raise RuntimeError(f"Source hash differs from prior independent receipt: {dataset}")
        source_hashes[dataset] = receipt

    selection = json.loads((ROOT / "audit/selection_audit.json").read_text(encoding="utf-8"))
    compact = {
        "status": "PASS",
        "l3_treated": {
            key: value
            for key, value in selection["l3_treated"].items()
            if not key.endswith("_values")
        },
        "l3_control": selection["l3_control"],
        "l5": {
            key: value for key, value in selection["l5"].items() if not key.endswith("_values")
        },
    }
    atomic_text(ROOT / "audit/compact_selection_summary.json", json.dumps(compact, indent=2, sort_keys=True) + "\n")

    card = f"""# Canonical L1000 all-QC-pass 978-gene dataset

## Frozen meaning

- **L3 treated:** {summary['l3_treated']['rows']:,} official QC-pass `trt_cp` Level 3 profiles. Each row is one original `sample_id`; replicates, cells, drugs, doses and durations are preserved and never pooled.
- **L3 controls:** {summary['l3_control']['rows']:,} official QC-pass `ctl_vehicle` Level 3 profiles. Each row is one original vehicle-control `sample_id`; controls are not averaged.
- **L5 signatures:** {summary['l5']['rows']:,} official QC-pass `trt_cp` Level 5 signatures. Each row is one original `sig_id`; Level 5 is already a moderated differential signature, so there is no separate L5 control expression matrix.
- **Genes:** exactly 978 directly measured L1000 landmark genes in one shared frozen order. No inferred genes are included.
- **Filters:** no dose, duration, cell-line, molecular-structure, scaffold or coverage filter beyond official `qc_pass` and perturbation type.

## L3 control binding

The treatment manifest maps each L3 treatment to a strict control context defined by cell line, project, detection plate, RNA plate and normalized treatment time. {compact['l3_treated']['rows_with_exact_qc_vehicle_control']:,} treatment profiles have such a QC-pass vehicle context; {compact['l3_treated']['rows_without_exact_qc_vehicle_control']:,} do not. The context manifest stores **all raw control rows** for each context. No fallback or implicit averaging is applied.

## Storage

Expression values are float32 HDF5 shards with gzip-1, shuffle and Fletcher32 checksums. Row metadata and mappings are Parquet. `scripts/canonical_l1000_loader.py` provides random row access. All output shards have extraction hashes and independent source-value spot checks; `READY.json` is written only after the final independent verification succeeds.

## Important interpretation

L3 is normalized expression, not treatment-minus-control delta. L5 is a differential signature, not raw expression. Any future L3 delta must be constructed explicitly from the treatment row and its listed raw vehicle controls, with the aggregation rule registered by the analysis.
"""
    atomic_text(ROOT / "DATASET_CARD.md", card)

    include = []
    include.extend(sorted((ROOT / "data").glob("*.h5")))
    include.extend(sorted((ROOT / "data").glob("*.parquet")))
    include.extend(sorted((ROOT / "audit").glob("*.json")))
    include.extend([ROOT / "DATASET_CARD.md", ROOT / "scripts/canonical_l1000_loader.py"])
    entries = []
    for path in include:
        relative = path.relative_to(ROOT).as_posix()
        digest = h5_receipt_sha[relative] if relative in h5_receipt_sha else sha256sum(path)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    canonical_manifest = {
        "format": "canonical-l1000-all-qc-pass-978-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "source_gctx": source_hashes,
        "files": entries,
    }
    canonical_manifest_path = ROOT / "CANONICAL_MANIFEST.json"
    atomic_text(
        canonical_manifest_path,
        json.dumps(canonical_manifest, indent=2, sort_keys=True) + "\n",
    )
    assembly = {
        "status": "ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION",
        "format": canonical_manifest["format"],
        "canonical_manifest_sha256": sha256sum(canonical_manifest_path),
        "summary": summary,
    }
    atomic_text(ROOT / "ASSEMBLY_READY.json", json.dumps(assembly, indent=2, sort_keys=True) + "\n")
    print(json.dumps(assembly, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
