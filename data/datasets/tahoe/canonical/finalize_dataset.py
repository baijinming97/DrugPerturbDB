from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
REFERENCE = Path("@WORKSPACE@/baicai/tahoe_official_drug_signal_audit_20260823")
RAW_REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
PDEX_REVISION = "c7963cf334bec0683225d41c9586d900ca6303a2"
N_CONDITIONS = 52886
N_GENES = 62710


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    if path.exists():
        existing = pq.read_table(path)
        if not existing.equals(table):
            raise RuntimeError(f"Existing finalized Parquet differs from expected content: {path}")
        return
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def download_official_metadata(name: str) -> Path:
    path = ROOT / "data" / f"official_{name}"
    if path.exists():
        pq.read_schema(path)
        return path
    tmp = path.with_name(path.name + ".part")
    url = (
        f"https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/{RAW_REVISION}/"
        f"metadata/{name}?download=true"
    )
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
    pq.read_schema(tmp)
    os.replace(tmp, path)
    return path


ready = ROOT / "READY.json"
if ready.exists():
    raise FileExistsError(f"Refuse overwrite of existing READY marker: {ready}")
if list(ROOT.rglob("*.part")):
    raise RuntimeError("Partial files exist; refuse to finalize")

qc_conditions = pq.read_table(ROOT / "audit/qc_condition_manifest.parquet").to_pandas()
if len(qc_conditions) != N_CONDITIONS or qc_conditions["condition_row"].nunique() != N_CONDITIONS:
    raise RuntimeError("QC condition manifest is not the frozen 52,886-row universe")
if qc_conditions["cell_line"].nunique() != 47 or qc_conditions["drug"].nunique() != 379:
    raise RuntimeError("QC cell-line/drug counts are not 47/379")
if qc_conditions[["drug", "dose"]].drop_duplicates().shape[0] != 1135:
    raise RuntimeError("QC drug-dose count is not 1,135")

edges = np.linspace(0, N_CONDITIONS, 17, dtype=np.int64)
qc_conditions = qc_conditions.sort_values("condition_row").reset_index(drop=True)
qc_conditions["treated_shard"] = np.searchsorted(edges[1:], qc_conditions["condition_row"], side="right")
qc_conditions["treated_row_in_shard"] = qc_conditions.apply(
    lambda row: int(row["condition_row"] - edges[int(row["treated_shard"])]), axis=1
)
condition_path = ROOT / "data/condition_manifest.parquet"
write_parquet_atomic(pa.Table.from_pandas(qc_conditions, preserve_index=False), condition_path)

gene_axis_path = ROOT / "data/gene_axis.parquet"
write_parquet_atomic(pq.read_table(ROOT / "audit/gene_axis.parquet"), gene_axis_path)
drug_metadata_path = ROOT / "data/drug_metadata.parquet"
write_parquet_atomic(
    pq.read_table(REFERENCE / "audit/drug_structure_registry.parquet"), drug_metadata_path
)
official_paths = [
    download_official_metadata("cell_line_metadata.parquet"),
    download_official_metadata("drug_metadata.parquet"),
    download_official_metadata("gene_metadata.parquet"),
]

treated_verifications = [
    json.loads((ROOT / "audit" / f"verify_treated_{rank:02d}-of-16.json").read_text(encoding="utf-8"))
    for rank in range(16)
]
if any(item["status"] != "PASS" for item in treated_verifications):
    raise RuntimeError("At least one treated-shard verification failed")
if sum(int(item["conditions"]) for item in treated_verifications) != N_CONDITIONS:
    raise RuntimeError("Treated-shard verification condition count mismatch")

cell_checks = [
    json.loads((ROOT / "audit" / f"cell_eval_directional_{rank:02d}-of-16.json").read_text(encoding="utf-8"))
    for rank in range(16)
]
cell_totals = {
    group: {
        "conditions": 0,
        "values": 0,
        "sse": 0.0,
        "max_abs": 0.0,
        "new_sum": 0.0,
        "old_sum": 0.0,
        "new_sumsq": 0.0,
        "old_sumsq": 0.0,
        "dot": 0.0,
        "condition_cosine_defined": 0,
        "condition_cosine_positive": 0,
        "condition_cosine_sum": 0.0,
        "condition_cosine_min": 1.0,
        "condition_cosine_max": -1.0,
    }
    for group in ["single", "multi"]
}
for item in cell_checks:
    if item["status"] != "PASS":
        raise RuntimeError("At least one official cell_eval concordance shard failed")
    for group in cell_totals:
        source = item["metrics"][group]
        cell_totals[group]["conditions"] += int(source["conditions"])
        cell_totals[group]["values"] += int(source["values"])
        cell_totals[group]["sse"] += float(source["sse"])
        cell_totals[group]["max_abs"] = max(
            cell_totals[group]["max_abs"], float(source["max_abs"])
        )
        for key in [
            "new_sum",
            "old_sum",
            "new_sumsq",
            "old_sumsq",
            "dot",
            "condition_cosine_sum",
        ]:
            cell_totals[group][key] += float(source[key])
        for key in ["condition_cosine_defined", "condition_cosine_positive"]:
            cell_totals[group][key] += int(source[key])
        if source["condition_cosine_min"] is not None:
            cell_totals[group]["condition_cosine_min"] = min(
                cell_totals[group]["condition_cosine_min"],
                float(source["condition_cosine_min"]),
            )
            cell_totals[group]["condition_cosine_max"] = max(
                cell_totals[group]["condition_cosine_max"],
                float(source["condition_cosine_max"]),
            )
for values in cell_totals.values():
    n = values["values"]
    values["rmse"] = float(np.sqrt(values["sse"] / n)) if n else None
    centered_dot = values["dot"] - values["new_sum"] * values["old_sum"] / n
    centered_new = values["new_sumsq"] - values["new_sum"] ** 2 / n
    centered_old = values["old_sumsq"] - values["old_sum"] ** 2 / n
    values["flat_pearson"] = float(
        centered_dot / np.sqrt(max(centered_new, 0.0) * max(centered_old, 0.0))
    )
    values["flat_cosine"] = float(
        values["dot"] / np.sqrt(values["new_sumsq"] * values["old_sumsq"])
    )
    defined = values["condition_cosine_defined"]
    values["mean_condition_cosine"] = values["condition_cosine_sum"] / defined
    values["positive_condition_cosine_fraction"] = (
        values["condition_cosine_positive"] / defined
    )
if sum(values["conditions"] for values in cell_totals.values()) != N_CONDITIONS:
    raise RuntimeError("Official cell_eval concordance condition count mismatch")
for group, values in cell_totals.items():
    if values["flat_pearson"] < 0.5:
        raise RuntimeError(f"Weak PDEX/cell_eval flat correlation for {group}: {values}")
    if values["mean_condition_cosine"] < 0.5:
        raise RuntimeError(f"Weak PDEX/cell_eval mean condition cosine for {group}: {values}")
    if values["positive_condition_cosine_fraction"] < 0.8:
        raise RuntimeError(f"Too few positive PDEX/cell_eval condition cosines for {group}: {values}")

control_audit = json.loads((ROOT / "audit/control_merge_audit.json").read_text(encoding="utf-8"))
if control_audit["status"] != "PASS":
    raise RuntimeError("Control merge audit did not pass")
with h5py.File(ROOT / "data/controls.h5", "r") as handle:
    if handle["controls"].shape[1] != N_GENES:
        raise RuntimeError("Canonical control gene dimension mismatch")
    controls = np.asarray(handle["controls"][:], dtype=np.float32)
    if not np.isfinite(controls).all():
        raise RuntimeError("Canonical controls contain non-finite values")

final_verification = {
    "status": "PASS",
    "conditions": N_CONDITIONS,
    "genes": N_GENES,
    "treated_shards": 16,
    "treated_range": [
        min(item["value_range"][0] for item in treated_verifications),
        max(item["value_range"][1] for item in treated_verifications),
    ],
    "treated_zero_count": sum(int(item["zero_count"]) for item in treated_verifications),
    "official_cell_eval_2k_directional_crosscheck": cell_totals,
    "official_cell_eval_rule": {
        "endpoint_identity": "PDEX target_mean/ref_mean and cell_eval are distinct official endpoints; numerical identity is neither expected nor claimed.",
        "gate": "Exact condition/gene binding plus flat Pearson >=0.5, mean condition cosine >=0.5, and >=80% positive condition cosines, separately for single- and multi-plate conditions.",
        "multi_plate": "Old cell_eval used equal plate weights; canonical PDEX uses raw full-QC treated-cell-count weights.",
    },
    "control_merge": control_audit,
}
verification_path = ROOT / "audit/final_verification.json"
verification_path.write_text(json.dumps(final_verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

card = f"""# Tahoe-PDEX-full-QC-v1

Canonical compact pseudobulk source for future Tahoe perturbation modelling.

## Frozen universe

- 52,886 unique cell-line x drug x dose conditions.
- 47 cell lines; NCI-H661 (CVCL_1577), NCI-H596 (CVCL_1571), and NCI-H2122 (CVCL_1531) are excluded.
- 379 drugs, 1,135 observed drug-dose pairs, doses 0.05/0.5/5.0 uM.
- Full-QC (`pass_filter == full`) is applied at cell level, then each plate x cell-line x drug-dose profile must contain at least 50 cells before replicate plates are pooled.
- The 52,886 unique conditions are obtained only after that plate-profile filter. Five conditions that would reach 50 only by summing multiple individually sub-50 plate profiles are excluded and recorded in the audit.
- 62,710 official PDEX features in the frozen `gene_axis.parquet` order.

## Values and controls

- `treated_00-of-16.h5` ... `treated_15-of-16.h5`: float32 official PDEX `target_mean`, with retained replicate plates pooled by raw full-QC treated-cell-count weights.
- `controls.h5`: deduplicated official plate-matched DMSO `ref_mean` vectors.
- `condition_control_map.parquet`: maps each condition to one or more control vectors and weights. Weights sum to one.
- Delta is intentionally not duplicated. Reconstruct it as `treated - weighted_control`.
- Expression is on the official linear normalized scale; it is not log fold change.
- Official `cell_eval` is retained only as a directional cross-endpoint check. It is not numerically interchangeable with PDEX and is not used to rescale the canonical values.

## Provenance

- PDEX source: `tahoebio/tahoe-de-rhaister`, revision `{PDEX_REVISION}`.
- Tahoe metadata: `tahoebio/Tahoe-100M`, revision `{RAW_REVISION}`.
- The 40.6 GB source Parquet was range-read and is not retained locally.
- Plate provenance, target/ref memberships, chemistry identifiers, and aggregation weights are retained in the manifests.

## Intended use

This is the single canonical raw pseudobulk source. Create experiment-specific high-speed subsets from it; do not duplicate the entire matrix. The 52,886 rows are condition-level observations, not 52,886 drugs and not statistically independent biological samples; likewise, the 95.6 million full-QC cells are not 95.6 million independent perturbations.
"""
card_path = ROOT / "DATASET_CARD.md"
if card_path.exists():
    raise FileExistsError(f"Refuse overwrite: {card_path}")
card_path.write_text(card, encoding="utf-8")

data_files = sorted(path for path in (ROOT / "data").iterdir() if path.is_file())
manifest_rows = []
for path in data_files:
    manifest_rows.append(
        {
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256sum(path),
        }
    )
manifest_path = ROOT / "audit/file_manifest.tsv"
if manifest_path.exists():
    raise FileExistsError(f"Refuse overwrite: {manifest_path}")
pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)

ready_payload = {
    "status": "READY",
    "dataset": "Tahoe-PDEX-full-QC-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "conditions": N_CONDITIONS,
    "genes": N_GENES,
    "cell_lines": 47,
    "drugs": 379,
    "drug_doses": 1135,
    "data_files": len(manifest_rows),
    "data_bytes": int(sum(row["bytes"] for row in manifest_rows)),
    "file_manifest": str(manifest_path.relative_to(ROOT)),
    "file_manifest_sha256": sha256sum(manifest_path),
    "final_verification": str(verification_path.relative_to(ROOT)),
    "final_verification_sha256": sha256sum(verification_path),
    "dataset_card_sha256": sha256sum(card_path),
}
ready.write_text(json.dumps(ready_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(ready_payload, indent=2, sort_keys=True))
