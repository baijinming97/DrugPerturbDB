from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
REFERENCE = Path("@WORKSPACE@/baicai/tahoe_official_drug_signal_audit_20260823")
EXCLUDED = {
    "CVCL_1577": "NCI-H661",
    "CVCL_1571": "NCI-H596",
    "CVCL_1531": "NCI-H2122",
}
EXPECTED = {
    "conditions": 52886,
    "cell_lines": 47,
    "drugs": 379,
    "drug_doses": 1135,
    "genes": 62710,
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_target(value: str):
    parsed = ast.literal_eval(str(value))
    if len(parsed) != 1 or len(parsed[0]) != 3:
        raise RuntimeError(f"Unexpected target: {value}")
    return str(parsed[0][0]), float(parsed[0][1]), str(parsed[0][2])


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse overwrite: {path}")
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)


scan_paths = sorted(ROOT.glob("tmp/pdex_profile_scan_full16_*-of-16.parquet"))
if len(scan_paths) != 16:
    raise RuntimeError(f"Expected 16 scan shards, found {len(scan_paths)}")

scan = pd.concat([pq.read_table(path).to_pandas() for path in scan_paths], ignore_index=True)
if scan["row_group"].nunique() != 13045:
    raise RuntimeError(f"Incomplete row-group scan: {scan['row_group'].nunique()} != 13045")
if (scan["target_membership_min"] != scan["target_membership_max"]).any():
    raise RuntimeError("target_membership varies within a profile chunk")
if (scan["ref_membership_min"] != scan["ref_membership_max"]).any():
    raise RuntimeError("ref_membership varies within a profile chunk")

keys = ["plate", "cell_line", "target"]
profiles = (
    scan.groupby(keys, sort=False, as_index=False)
    .agg(
        target_membership=("target_membership_min", "first"),
        target_membership_max=("target_membership_max", "max"),
        ref_membership=("ref_membership_min", "first"),
        ref_membership_max=("ref_membership_max", "max"),
        feature_rows=("feature_rows", "sum"),
        row_group_count=("row_group", "nunique"),
        row_groups=("row_group", lambda x: ",".join(str(int(v)) for v in sorted(set(x)))),
    )
)
if (profiles["target_membership"] != profiles["target_membership_max"]).any():
    raise RuntimeError("target_membership varies across profile chunks")
if (profiles["ref_membership"] != profiles["ref_membership_max"]).any():
    raise RuntimeError("ref_membership varies across profile chunks")
if len(profiles) != 65218:
    raise RuntimeError(f"PDEX profile key count {len(profiles)} != 65218")
if not (profiles["feature_rows"] == EXPECTED["genes"]).all():
    bad = profiles.loc[profiles["feature_rows"] != EXPECTED["genes"], keys + ["feature_rows"]]
    raise RuntimeError(f"Incomplete full-gene profiles: {bad.head(20).to_dict('records')}")

plate = pq.read_table(REFERENCE / "audit/cell_eval_plate_row_manifest.parquet").to_pandas()
plate = plate.rename(columns={"treatment": "target"})
if plate.duplicated(keys).any() or len(plate) != 65218:
    raise RuntimeError("Reference plate manifest is not a unique 65,218-key table")
profiles = profiles.merge(plate, on=keys, how="outer", validate="one_to_one", indicator=True)
if not (profiles["_merge"] == "both").all():
    raise RuntimeError(f"PDEX/cell_eval key mismatch: {profiles['_merge'].value_counts().to_dict()}")
profiles = profiles.drop(columns=["_merge", "target_membership_max", "ref_membership_max"])

full_qc_counts = pq.read_table(ROOT / "audit/full_qc_plate_counts.parquet").to_pandas()
count_columns = keys + ["full_qc_cells"]
if full_qc_counts.duplicated(keys).any() or len(full_qc_counts) != 65218:
    raise RuntimeError("Full-QC plate count manifest is not a unique 65,218-key table")
profiles = profiles.merge(
    full_qc_counts[count_columns], on=keys, how="left", validate="one_to_one"
)
if profiles["full_qc_cells"].isna().any():
    raise RuntimeError("Some official plate profiles lack a full-QC count record")
profiles["full_qc_cells"] = profiles["full_qc_cells"].astype(np.int64)

parsed = profiles["target"].map(parse_target)
profiles["target_drug_label"] = parsed.map(lambda x: x[0])
drug_label_mismatch = profiles.loc[
    profiles["drug"].astype(str) != profiles["target_drug_label"].astype(str),
    ["target_drug_label", "drug", "target"],
].drop_duplicates()
if not np.allclose(profiles["dose"].to_numpy(), parsed.map(lambda x: x[1]).to_numpy()):
    raise RuntimeError("Parsed dose disagrees with the official plate manifest")

profiles["excluded_cell_line"] = profiles["cell_line"].isin(EXCLUDED)
eligible_profiles = profiles.loc[
    (~profiles["excluded_cell_line"]) & (profiles["full_qc_cells"] >= 50)
].copy()
condition_keys = ["cell_line", "drug", "dose", "dose_unit"]
condition_counts = eligible_profiles.groupby(
    condition_keys, sort=True, as_index=False
).agg(target_cells=("full_qc_cells", "sum"))
contributing_profiles = eligible_profiles.loc[eligible_profiles["full_qc_cells"] > 0]
profile_summary = (
    contributing_profiles.groupby(condition_keys, sort=True, as_index=False)
    .agg(
        source_plate_profiles=("plate", "size"),
        plates=("plate", lambda x: ",".join(str(int(v)) for v in sorted(set(x)))),
        ref_cells_sum=("ref_membership", "sum"),
    )
)
condition_counts = condition_counts.merge(
    profile_summary, on=condition_keys, how="left", validate="one_to_one"
)
condition_counts["source_plate_profiles"] = (
    condition_counts["source_plate_profiles"].fillna(0).astype(np.int64)
)
condition_counts["plates"] = condition_counts["plates"].fillna("")
condition_counts["ref_cells_sum"] = condition_counts["ref_cells_sum"].fillna(0).astype(np.int64)
conditions = condition_counts.loc[condition_counts["target_cells"] >= 50].copy()

observed = {
    "conditions": int(len(conditions)),
    "cell_lines": int(conditions["cell_line"].nunique()),
    "drugs": int(conditions["drug"].nunique()),
    "drug_doses": int(conditions[["drug", "dose"]].drop_duplicates().shape[0]),
    "doses": sorted(float(v) for v in conditions["dose"].unique()),
    "median_cells_per_condition": float(conditions["target_cells"].median()),
    "eligible_plate_profiles": int(len(eligible_profiles)),
    "median_cells_per_eligible_plate_profile": float(eligible_profiles["full_qc_cells"].median()),
}
for key in ["conditions", "cell_lines", "drugs", "drug_doses"]:
    if observed[key] != EXPECTED[key]:
        alternate = {
            "plate_profiles_ge50_after_cell_exclusion": int(
                ((~profiles["excluded_cell_line"]) & (profiles["full_qc_cells"] >= 50)).sum()
            ),
            "all_conditions_after_cell_exclusion": int(len(condition_counts)),
            "conditions_ge50": observed["conditions"],
        }
        raise RuntimeError(f"Official QC universe mismatch: observed={observed}, alternate={alternate}")

drug = pq.read_table(REFERENCE / "audit/drug_structure_registry.parquet").to_pandas()
drug_columns = [
    "drug_metadata_row",
    "drug",
    "chemical_entity_id",
    "canonical_smiles",
    "structure_parse_ok",
    "parent_connectivity_key",
    "murcko_scaffold_id",
    "chemistry_component",
]
conditions = conditions.merge(drug[drug_columns], on="drug", how="left", validate="many_to_one")
if conditions["drug_metadata_row"].isna().any():
    raise RuntimeError("Some QC conditions are absent from the 379-drug registry")
conditions = conditions.sort_values(condition_keys, kind="mergesort").reset_index(drop=True)
conditions.insert(0, "condition_row", np.arange(len(conditions), dtype=np.int64))

keep = eligible_profiles.merge(
    conditions[condition_keys + ["condition_row", "target_cells"]],
    on=condition_keys,
    how="inner",
    validate="many_to_one",
)
keep = keep.loc[keep["full_qc_cells"] > 0].copy()
keep["aggregation_weight"] = keep["full_qc_cells"] / keep["target_cells"]
weight_check = keep.groupby("condition_row")["aggregation_weight"].sum().to_numpy()
if not np.allclose(weight_check, 1.0, atol=1e-7):
    raise RuntimeError("Plate aggregation weights do not sum to one")
keep = keep.sort_values(["condition_row", "plate"], kind="mergesort").reset_index(drop=True)
keep.insert(0, "profile_row", np.arange(len(keep), dtype=np.int64))

condition_path = ROOT / "audit/qc_condition_manifest.parquet"
profile_path = ROOT / "audit/qc_plate_profile_manifest.parquet"
write_parquet_atomic(conditions, condition_path)
write_parquet_atomic(keep, profile_path)

audit = {
    "status": "PASS",
    "pdex_revision": "c7963cf334bec0683225d41c9586d900ca6303a2",
    "pdex_profiles_all": int(len(profiles)),
    "conditions_before_cell_count_filter_after_cell_exclusion": int(len(condition_counts)),
    "qc_plate_profiles_contributing": int(len(keep)),
    "excluded_cell_lines": EXCLUDED,
    "pdex_target_to_official_drug_label": {
        "mismatched_plate_profiles": int(
            (profiles["drug"].astype(str) != profiles["target_drug_label"].astype(str)).sum()
        ),
        "unique_label_mappings": int(len(drug_label_mismatch)),
        "examples": drug_label_mismatch.head(20).to_dict("records"),
        "binding_rule": "Exact (plate, cell_line, target) key from the previously verified official cell_eval manifest; official 379-drug label is authoritative.",
    },
    "criteria": {
        "minimum_full_qc_cells_per_plate_cell_drug_dose_profile": 50,
        "threshold_order": "Filter plate profiles first, then pool retained replicate plates into unique cell/drug/dose conditions.",
        "condition_cell_count": "sum of raw obs_metadata cells with pass_filter == full across contributing plate profiles",
        "plate_aggregation_weight": "plate full-QC treated cells / total condition full-QC treated cells",
        "pdex_target_membership": "retained as provenance only; not used for the paper full-QC threshold or weights",
    },
    "observed": observed,
    "expected": EXPECTED,
    "condition_manifest": str(condition_path),
    "condition_manifest_sha256": sha256sum(condition_path),
    "plate_profile_manifest": str(profile_path),
    "plate_profile_manifest_sha256": sha256sum(profile_path),
}
audit_path = ROOT / "audit/qc_universe_audit.json"
if audit_path.exists():
    raise FileExistsError(f"Refuse overwrite: {audit_path}")
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
