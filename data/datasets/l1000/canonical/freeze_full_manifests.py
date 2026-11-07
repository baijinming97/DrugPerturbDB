from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000")
META = Path("@WORKSPACE@/Noise/LINCS2020/raw/metadata")
OLD_PREP = Path("@WORKSPACE@/baicai/l3_l5_978_prep_20260822")
SOURCES = {
    "l3_treated": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_trt_cp_n1805898x12328.gctx"),
    "l3_control": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_ctl_n188708x12328.gctx"),
    "l5": OLD_PREP / "raw/level5_beta_trt_cp_n720216x12328.gctx",
}
INST = META / "instinfo_beta.txt"
SIG = META / "siginfo_beta.txt"
GENE_AXIS_SOURCE = OLD_PREP / "gene_axis_978.tsv"


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "t", "yes"})


def dose_um(values: pd.Series, units: pd.Series) -> pd.Series:
    unit = (
        units.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("µ", "u", regex=False)
        .str.replace("μ", "u", regex=False)
    )
    factor = unit.map({"um": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6, "pm": 1e-6})
    return pd.to_numeric(values, errors="coerce") * factor


def duration_h(values: pd.Series, units: pd.Series) -> pd.Series:
    unit = units.astype(str).str.strip().str.lower()
    factor = unit.map(
        {
            "h": 1.0,
            "hr": 1.0,
            "hrs": 1.0,
            "hour": 1.0,
            "hours": 1.0,
            "m": 1 / 60,
            "min": 1 / 60,
            "mins": 1 / 60,
            "minute": 1 / 60,
            "minutes": 1 / 60,
            "d": 24.0,
            "day": 24.0,
            "days": 24.0,
        }
    )
    return pd.to_numeric(values, errors="coerce") * factor


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)


def read_gctx_axes(path: Path) -> tuple[dict, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["0/DATA/0/matrix"]
        row_ids = decode(handle["0/META/ROW/id"][...])
        col_ids = decode(handle["0/META/COL/id"][...])
        info = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "matrix_shape": [int(v) for v in matrix.shape],
            "matrix_dtype": str(matrix.dtype),
            "matrix_chunks": list(matrix.chunks) if matrix.chunks else None,
            "row_ids": len(row_ids),
            "col_ids": len(col_ids),
            "row_ids_unique": len(row_ids) == len(set(row_ids)),
            "col_ids_unique": len(col_ids) == len(set(col_ids)),
        }
    if info["matrix_shape"] != [len(col_ids), len(row_ids)]:
        raise RuntimeError(f"Unexpected GCTX orientation: {path} {info}")
    if not info["row_ids_unique"] or not info["col_ids_unique"]:
        raise RuntimeError(f"Non-unique GCTX axis: {path}")
    return info, row_ids, col_ids


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str)
    if {"pert_dose", "pert_dose_unit"}.issubset(frame.columns):
        frame["dose_uM"] = dose_um(frame["pert_dose"], frame["pert_dose_unit"])
    else:
        frame["dose_uM"] = np.nan
    if {"pert_time", "pert_time_unit"}.issubset(frame.columns):
        frame["duration_h"] = duration_h(frame["pert_time"], frame["pert_time_unit"])
    else:
        frame["duration_h"] = np.nan
    return frame


def read_selected_metadata(path: Path, id_column: str, pert_type: str) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=100_000, low_memory=False):
        selected = chunk.loc[
            chunk["pert_type"].astype(str).eq(pert_type) & truthy(chunk["qc_pass"])
        ].copy()
        if len(selected):
            pieces.append(normalize_metadata(selected))
    frame = pd.concat(pieces, ignore_index=True)
    if frame[id_column].duplicated().any():
        raise RuntimeError(f"Selected metadata has duplicate {id_column}: {path}")
    return frame


for directory in [ROOT, ROOT / "data", ROOT / "audit", ROOT / "scripts", ROOT / "jobs", ROOT / "logs", ROOT / "tmp"]:
    directory.mkdir(parents=True, exist_ok=True)

source_info = {}
source_rows = {}
source_profiles = {}
for name, path in SOURCES.items():
    info, rows, profiles = read_gctx_axes(path)
    source_info[name] = info
    source_rows[name] = rows
    source_profiles[name] = profiles

axis = pd.read_csv(GENE_AXIS_SOURCE, sep="\t", dtype=str)
if len(axis) != 978 or axis["gene_id"].nunique() != 978:
    raise RuntimeError("Prior directly measured gene axis is not 978 unique gene IDs")
axis["axis_position"] = pd.to_numeric(axis["axis_position"], errors="raise").astype(np.int64)
axis = axis.sort_values("axis_position").reset_index(drop=True)
if not np.array_equal(axis["axis_position"].to_numpy(), np.arange(978)):
    raise RuntimeError("Canonical 978 gene positions are not exact 0..977")

bindings = axis[["axis_position", "gene_id", "gene_symbol", "feature_space"]].copy()
for name in SOURCES:
    lookup = {gene: position for position, gene in enumerate(source_rows[name])}
    positions = bindings["gene_id"].map(lookup)
    if positions.isna().any():
        raise RuntimeError(f"Canonical genes missing from {name}: {bindings.loc[positions.isna(), 'gene_id'].tolist()[:20]}")
    bindings[f"{name}_source_gene_position"] = positions.astype(np.int64)
write_parquet_atomic(bindings, ROOT / "data/gene_axis_978.parquet")

l3_treated = read_selected_metadata(INST, "sample_id", "trt_cp")
l3_control = read_selected_metadata(INST, "sample_id", "ctl_vehicle")
l5 = read_selected_metadata(SIG, "sig_id", "trt_cp")

def bind_profiles(frame: pd.DataFrame, id_column: str, source_name: str, row_name: str) -> pd.DataFrame:
    lookup = pd.Series(
        np.arange(len(source_profiles[source_name]), dtype=np.int64),
        index=pd.Index(source_profiles[source_name], name=id_column),
    )
    frame["source_profile_index"] = frame[id_column].map(lookup)
    if frame["source_profile_index"].isna().any():
        missing = frame.loc[frame["source_profile_index"].isna(), id_column].tolist()
        raise RuntimeError(f"QC-pass metadata rows absent from {source_name}: {missing[:20]}")
    frame["source_profile_index"] = frame["source_profile_index"].astype(np.int64)
    frame = frame.sort_values("source_profile_index", kind="mergesort").reset_index(drop=True)
    frame.insert(0, row_name, np.arange(len(frame), dtype=np.int64))
    return frame


l3_treated = bind_profiles(l3_treated, "sample_id", "l3_treated", "treatment_row")
l3_control = bind_profiles(l3_control, "sample_id", "l3_control", "control_row")
l5 = bind_profiles(l5, "sig_id", "l5", "signature_row")

# Exact technical-context mapping; expression remains at raw replicate level.
context_columns = ["cell_iname", "project_code", "det_plate", "rna_plate"]
for frame in [l3_treated, l3_control]:
    duration_key = frame["duration_h"].map(lambda value: f"{value:.12g}" if pd.notna(value) else "")
    raw_time = frame.get("pert_itime", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    frame["control_match_time"] = np.where(duration_key.ne(""), duration_key, raw_time)
context_columns.append("control_match_time")

control_context = (
    l3_control.groupby(context_columns, sort=True, dropna=False)
    .agg(
        n_controls=("control_row", "size"),
        control_rows=("control_row", lambda x: [int(v) for v in x]),
        control_sample_ids=("sample_id", lambda x: list(map(str, x))),
    )
    .reset_index()
)
control_context.insert(0, "control_context_id", np.arange(len(control_context), dtype=np.int64))
context_lookup = control_context[context_columns + ["control_context_id", "n_controls"]]
l3_treated = l3_treated.merge(
    context_lookup, on=context_columns, how="left", validate="many_to_one"
)
l3_treated["has_exact_qc_vehicle_control"] = l3_treated["control_context_id"].notna()
l3_treated["control_context_id"] = l3_treated["control_context_id"].fillna(-1).astype(np.int64)
l3_treated["n_controls"] = l3_treated["n_controls"].fillna(0).astype(np.int64)
l3_treated = l3_treated.sort_values("treatment_row").reset_index(drop=True)

write_parquet_atomic(l3_treated, ROOT / "data/l3_treatment_manifest.parquet")
write_parquet_atomic(l3_control, ROOT / "data/l3_control_manifest.parquet")
write_parquet_atomic(control_context, ROOT / "data/l3_control_context_manifest.parquet")
write_parquet_atomic(l5, ROOT / "data/l5_signature_manifest.parquet")

def coverage(frame: pd.DataFrame, id_column: str) -> dict:
    condition_columns = [id_column, "cell_iname", "dose_uM", "duration_h"]
    return {
        "rows": int(len(frame)),
        "unique_pert_ids": int(frame[id_column].nunique()),
        "unique_cells": int(frame["cell_iname"].nunique()),
        "unique_conditions_pert_cell_dose_time": int(frame[condition_columns].drop_duplicates().shape[0]),
        "dose_uM_nonmissing_rows": int(frame["dose_uM"].notna().sum()),
        "duration_h_nonmissing_rows": int(frame["duration_h"].notna().sum()),
        "dose_uM_values": sorted(float(v) for v in frame["dose_uM"].dropna().unique()),
        "duration_h_values": sorted(float(v) for v in frame["duration_h"].dropna().unique()),
    }


audit = {
    "status": "PASS",
    "classification": "ALL_OFFICIAL_QC_PASS_TRT_CP_L3_REPLICATES_AND_L5_SIGNATURES_WITH_QC_VEHICLE_CONTROLS",
    "selection": {
        "l3_treated": "instinfo pert_type == trt_cp and official qc_pass truthy; one row per original sample_id; no condition or replicate pooling",
        "l3_control": "instinfo pert_type == ctl_vehicle and official qc_pass truthy; one row per original sample_id; no pooling",
        "l5": "siginfo pert_type == trt_cp and official qc_pass truthy; one row per original sig_id; no cell/drug/dose/time pooling; Level5 already encodes a moderated differential signature and therefore has no separate control matrix",
        "genes": "978 directly measured landmark genes only, in the previously frozen canonical order",
        "structure_filter": "none",
        "dose_time_filter": "none",
    },
    "gene_axis": {
        "genes": 978,
        "source_sha256": sha256sum(GENE_AXIS_SOURCE),
        "canonical_sha256": sha256sum(ROOT / "data/gene_axis_978.parquet"),
    },
    "source_gctx": source_info,
    "l3_treated": {
        **coverage(l3_treated, "pert_id"),
        "rows_with_exact_qc_vehicle_control": int(l3_treated["has_exact_qc_vehicle_control"].sum()),
        "rows_without_exact_qc_vehicle_control": int((~l3_treated["has_exact_qc_vehicle_control"]).sum()),
        "exact_control_contexts": int(len(control_context)),
    },
    "l3_control": {
        "rows": int(len(l3_control)),
        "unique_cells": int(l3_control["cell_iname"].nunique()),
        "exact_control_contexts": int(len(control_context)),
    },
    "l5": coverage(l5, "pert_id"),
    "manifests": {},
}
for name in [
    "gene_axis_978.parquet",
    "l3_treatment_manifest.parquet",
    "l3_control_manifest.parquet",
    "l3_control_context_manifest.parquet",
    "l5_signature_manifest.parquet",
]:
    path = ROOT / "data" / name
    audit["manifests"][name] = {"bytes": path.stat().st_size, "sha256": sha256sum(path)}

out = ROOT / "audit/selection_audit.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
