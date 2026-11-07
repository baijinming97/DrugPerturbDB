from __future__ import annotations

import json
from pathlib import Path

import h5py
import pyarrow.parquet as pq


TAHOE = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe/data")
L1000 = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000/data")


def parquet_summary(path: Path, sample_columns: list[str] | None = None) -> dict:
    parquet = pq.ParquetFile(path)
    result = {
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": [
            {"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow
        ],
    }
    if sample_columns:
        available = [column for column in sample_columns if column in parquet.schema_arrow.names]
        result["sample"] = pq.read_table(path, columns=available).slice(0, 2).to_pylist()
    return result


def h5_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        result = {"attrs": {key: str(value) for key, value in handle.attrs.items()}, "datasets": {}}
        for name, dataset in handle.items():
            result["datasets"][name] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "chunks": list(dataset.chunks) if dataset.chunks else None,
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
                "shuffle": bool(dataset.shuffle),
                "fletcher32": bool(dataset.fletcher32),
            }
    return result


payload = {
    "tahoe": {
        "h5_treated_example": h5_summary(TAHOE / "treated_00-of-16.h5"),
        "h5_controls": h5_summary(TAHOE / "controls.h5"),
        "parquet": {
            "condition_manifest": parquet_summary(
                TAHOE / "condition_manifest.parquet",
                ["condition_row", "cell_line", "drug", "drug_id", "dose_um", "n_full_qc_cells", "n_retained_plates", "treated_shard", "treated_row_in_shard"],
            ),
            "condition_control_map": parquet_summary(
                TAHOE / "condition_control_map.parquet",
                ["condition_row", "control_row", "control_weight", "plate"],
            ),
            "control_manifest": parquet_summary(TAHOE / "control_manifest.parquet"),
            "control_context_manifest": parquet_summary(TAHOE / "control_context_manifest.parquet"),
            "gene_axis": parquet_summary(
                TAHOE / "gene_axis.parquet", ["gene_position", "feature"]
            ),
            "drug_metadata": parquet_summary(TAHOE / "drug_metadata.parquet"),
        },
    },
    "l1000": {
        "h5_l3_treated_example": h5_summary(L1000 / "l3_treated_full_00-of-32.h5"),
        "h5_l3_control_example": h5_summary(L1000 / "l3_control_full_00-of-04.h5"),
        "h5_l5_example": h5_summary(L1000 / "l5_signatures_full_00-of-16.h5"),
        "parquet": {
            "l3_treatment_manifest": parquet_summary(
                L1000 / "l3_treatment_manifest.parquet",
                ["treatment_row", "sample_id", "pert_id", "cell_iname", "pert_dose", "pert_dose_unit", "dose_uM", "pert_time", "pert_time_unit", "duration_h", "qc_pass", "control_context_id", "n_controls", "has_exact_qc_vehicle_control"],
            ),
            "l3_control_manifest": parquet_summary(
                L1000 / "l3_control_manifest.parquet",
                ["control_row", "sample_id", "cell_iname", "pert_time", "pert_time_unit", "duration_h", "project_code", "det_plate", "rna_plate", "qc_pass"],
            ),
            "l3_control_context_manifest": parquet_summary(
                L1000 / "l3_control_context_manifest.parquet"
            ),
            "l5_signature_manifest": parquet_summary(
                L1000 / "l5_signature_manifest.parquet",
                ["signature_row", "sig_id", "pert_id", "cell_iname", "pert_dose", "pert_dose_unit", "dose_uM", "pert_time", "pert_time_unit", "duration_h", "qc_pass"],
            ),
            "gene_axis_978": parquet_summary(
                L1000 / "gene_axis_978.parquet",
                ["axis_position", "gene_id", "gene_symbol", "feature_space"],
            ),
        },
    },
}
print(json.dumps(payload, indent=2, default=str))
