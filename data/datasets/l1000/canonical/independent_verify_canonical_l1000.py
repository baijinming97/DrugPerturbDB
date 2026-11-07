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
CONFIG = {
    "l3_treated": {
        "source": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_trt_cp_n1805898x12328.gctx"),
        "manifest": "l3_treatment_manifest.parquet",
        "id": "sample_id",
        "row": "treatment_row",
        "source_gene": "l3_treated_source_gene_position",
        "pattern": "l3_treated_full_*-of-*.h5",
        "pert_type": "trt_cp",
        "rows": 1_312_170,
    },
    "l3_control": {
        "source": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_ctl_n188708x12328.gctx"),
        "manifest": "l3_control_manifest.parquet",
        "id": "sample_id",
        "row": "control_row",
        "source_gene": "l3_control_source_gene_position",
        "pattern": "l3_control_full_*-of-*.h5",
        "pert_type": "ctl_vehicle",
        "rows": 76_742,
    },
    "l5": {
        "source": Path("@WORKSPACE@/baicai/l3_l5_978_prep_20260822/raw/level5_beta_trt_cp_n720216x12328.gctx"),
        "manifest": "l5_signature_manifest.parquet",
        "id": "sig_id",
        "row": "signature_row",
        "source_gene": "l5_source_gene_position",
        "pattern": "l5_signatures_full_*-of-*.h5",
        "pert_type": "trt_cp",
        "rows": 510_270,
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(values: np.ndarray) -> np.ndarray:
    return np.isin(np.char.lower(np.char.strip(values.astype(str))), ["1", "1.0", "true", "t", "yes"])


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    if list(ROOT.rglob("*.part")):
        raise RuntimeError("Partial files remain")
    manifest_path = ROOT / "CANONICAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked_bytes = 0
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256sum(path) != entry["sha256"]:
            raise RuntimeError(f"Canonical file hash/size failure: {path}")
        checked_bytes += entry["bytes"]

    axis = pq.read_table(ROOT / "data/gene_axis_978.parquet").to_pandas().sort_values("axis_position")
    if len(axis) != 978 or axis["gene_id"].nunique() != 978:
        raise RuntimeError("Gene axis is not 978 unique genes")
    if not np.array_equal(axis["axis_position"].to_numpy(), np.arange(978)):
        raise RuntimeError("Gene axis positions are not 0..977")

    exact_checks = 0
    summaries = {}
    for dataset, cfg in CONFIG.items():
        metadata = pq.read_table(ROOT / "data" / cfg["manifest"]).to_pandas()
        if len(metadata) != cfg["rows"]:
            raise RuntimeError(f"Manifest count failure: {dataset}")
        if metadata[cfg["id"]].duplicated().any():
            raise RuntimeError(f"Duplicate source IDs: {dataset}")
        if not np.array_equal(metadata[cfg["row"]].to_numpy(), np.arange(cfg["rows"])):
            raise RuntimeError(f"Canonical row failure: {dataset}")
        if set(metadata["pert_type"].astype(str)) != {cfg["pert_type"]}:
            raise RuntimeError(f"Perturbation type failure: {dataset}")
        if not truthy(metadata["qc_pass"].to_numpy()).all():
            raise RuntimeError(f"Non-QC-pass row: {dataset}")
        source_indices_all = metadata["source_profile_index"].to_numpy(dtype=np.int64)
        if np.any(np.diff(source_indices_all) <= 0):
            raise RuntimeError(f"Source row order failure: {dataset}")

        positions = axis[cfg["source_gene"]].to_numpy(dtype=np.int64)
        order = np.argsort(positions)
        sorted_positions = positions[order]
        restore = np.argsort(order)
        expected_start = 0
        rows_seen = 0
        shards = sorted((ROOT / "data").glob(cfg["pattern"]))
        with h5py.File(cfg["source"], "r") as source:
            source_matrix = source["0/DATA/0/matrix"]
            for shard in shards:
                with h5py.File(shard, "r") as handle:
                    start = int(handle.attrs["canonical_start"])
                    stop = int(handle.attrs["canonical_stop"])
                    if start != expected_start or handle["expression"].shape != (stop - start, 978):
                        raise RuntimeError(f"Shard coverage/shape failure: {shard}")
                    canonical_rows = np.asarray(handle["canonical_row"][:], dtype=np.int64)
                    source_rows = np.asarray(handle["source_profile_index"][:], dtype=np.int64)
                    if not np.array_equal(canonical_rows, np.arange(start, stop)):
                        raise RuntimeError(f"Shard canonical binding failure: {shard}")
                    if not np.array_equal(source_rows, source_indices_all[start:stop]):
                        raise RuntimeError(f"Shard source binding failure: {shard}")
                    for local in sorted(set([0, len(canonical_rows) // 2, len(canonical_rows) - 1])):
                        expected = np.asarray(
                            source_matrix[int(source_rows[local]), sorted_positions], dtype=np.float32
                        )[restore]
                        observed = np.asarray(handle["expression"][local], dtype=np.float32)
                        if not np.array_equal(expected, observed):
                            raise RuntimeError(f"Exact source value failure: {shard} row {local}")
                        exact_checks += 1
                expected_start = stop
                rows_seen += stop - start
        if rows_seen != cfg["rows"] or expected_start != cfg["rows"]:
            raise RuntimeError(f"Dataset coverage failure: {dataset}")
        summaries[dataset] = {"rows": rows_seen, "shards": len(shards), "genes": 978}

    treatment = pq.read_table(
        ROOT / "data/l3_treatment_manifest.parquet",
        columns=["control_context_id", "n_controls", "has_exact_qc_vehicle_control"],
    ).to_pandas()
    contexts = pq.read_table(ROOT / "data/l3_control_context_manifest.parquet").to_pandas()
    context_ids = set(map(int, contexts["control_context_id"]))
    valid = treatment["control_context_id"].to_numpy(dtype=np.int64) >= 0
    if not np.array_equal(valid, treatment["has_exact_qc_vehicle_control"].to_numpy(dtype=bool)):
        raise RuntimeError("Treatment control availability flag failure")
    if not set(map(int, treatment.loc[valid, "control_context_id"])).issubset(context_ids):
        raise RuntimeError("Treatment references nonexistent control context")
    if (treatment.loc[valid, "n_controls"].to_numpy(dtype=np.int64) <= 0).any():
        raise RuntimeError("Valid control context has zero controls")
    for row in contexts.itertuples(index=False):
        if int(row.n_controls) != len(row.control_rows):
            raise RuntimeError("Control context list/count mismatch")
        if any(int(value) < 0 or int(value) >= CONFIG["l3_control"]["rows"] for value in row.control_rows):
            raise RuntimeError("Control context row out of range")

    verification = {
        "status": "PASS",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "format": manifest["format"],
        "canonical_manifest_sha256": sha256sum(manifest_path),
        "files_hashed": len(manifest["files"]),
        "bytes_hashed": checked_bytes,
        "exact_source_value_checks": exact_checks,
        "datasets": summaries,
        "l3_treatments_with_exact_controls": int(valid.sum()),
        "l3_treatments_without_exact_controls": int((~valid).sum()),
    }
    atomic_json(ROOT / "INDEPENDENT_VERIFICATION.json", verification)
    ready = {
        "status": "READY",
        "format": manifest["format"],
        "canonical_manifest_sha256": verification["canonical_manifest_sha256"],
        "independent_verification_sha256": sha256sum(ROOT / "INDEPENDENT_VERIFICATION.json"),
        "datasets": summaries,
    }
    atomic_json(ROOT / "READY.json", ready)
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

